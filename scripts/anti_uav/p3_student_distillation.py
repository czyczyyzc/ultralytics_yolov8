"""P2-teacher distillation, auxiliary-P2 supervision and RKNN INT8 QAT for a P3 student."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import weakref

import torch
import torch.nn.functional as F
from torch import nn

from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayAddOnTrainer, FixedShapeGrayP3Trainer
from scripts.anti_uav.rknn_qat import (
    prepare_rknn_qat,
    qat_copy,
    set_rknn_qat,
    sync_rknn_qat_observer_flags,
)
from scripts.anti_uav.tiny_aware_loss import TinyAwareAddOnP2DetectionLoss, TinyAwareDetectionLoss
from ultralytics import YOLO
from ultralytics.nn.modules import FrozenP3AddOnP2Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import de_parallel


class DetectInputCapture:
    """Retain the current detector-neck inputs for localized feature transfer."""

    def __init__(self, detector: nn.Module):
        self.features = None
        owner = weakref.ref(self)

        def capture(_module, inputs):
            value = inputs[0]
            if not isinstance(value, (list, tuple)):
                raise TypeError("Expected a list of detector feature maps")
            current = owner()
            if current is not None:
                current.features = tuple(value)

        self.handle = detector.register_forward_pre_hook(capture)


def attach_tiny_feature_projection(model: DetectionModel) -> nn.Conv2d:
    """Attach the disposable teacher-P2 to student-P3 projection before optimizer creation."""
    if hasattr(model, "tiny_feature_projection"):
        return model.tiny_feature_projection
    detector = model.model[-1]
    if not isinstance(detector, FrozenP3AddOnP2Detect):
        raise TypeError("Feature projection requires a temporary P2+P3 graph")
    p2_channels = detector.cv2[0][0].conv.in_channels
    p3_channels = detector.cv2[1][0].conv.in_channels
    projection = nn.Conv2d(p2_channels, p3_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(projection.weight, mode="fan_out", nonlinearity="linear")
    projection.to(next(model.parameters()).device)
    model.add_module("tiny_feature_projection", projection)
    return projection


class P3StudentDistillationLoss:
    """Supervise P3 normally and transfer high-confidence P2/P3 teacher evidence."""

    def __init__(
        self,
        model: DetectionModel,
        teacher: DetectionModel,
        auxiliary_weight: float = 0.25,
        p3_cls_weight: float = 0.10,
        p3_dfl_weight: float = 0.05,
        p2_cls_weight: float = 0.25,
        p2_box_weight: float = 0.10,
        feature_weight: float = 0.10,
        tiny_multiplier: float = 1.75,
        temperature: float = 2.0,
    ):
        detector = model.model[-1]
        if not isinstance(detector, FrozenP3AddOnP2Detect):
            raise TypeError("The student must use FrozenP3AddOnP2Detect during training")
        self.model = model
        self.teacher = teacher
        self.primary = TinyAwareDetectionLoss(model, tiny_multiplier=tiny_multiplier)
        self.primary.stride = detector.stride[detector.legacy_start_index :]
        self.auxiliary = v8DetectionLoss(model)
        self.auxiliary.stride = detector.stride[: detector.legacy_start_index]
        self.reg_max = detector.reg_max
        self.nc = detector.nc
        self.weights = dict(
            auxiliary=auxiliary_weight,
            p3_cls=p3_cls_weight,
            p3_dfl=p3_dfl_weight,
            p2_cls=p2_cls_weight,
            p2_box=p2_box_weight,
            feature=feature_weight,
        )
        self.temperature = temperature
        self.projection = attach_tiny_feature_projection(model)
        self.student_capture = DetectInputCapture(model.model[-1])
        self.teacher_capture = DetectInputCapture(teacher.model[-1])

    def _split(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boundary = self.reg_max * 4
        return raw[:, :boundary].float(), raw[:, boundary:].float()

    @staticmethod
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(1)
        return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(1e-6)

    def _classification_kd(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature
        target = (teacher.detach() / temperature).sigmoid()
        confidence = teacher.detach().sigmoid().amax(dim=1, keepdim=True).pow(2)
        loss = F.binary_cross_entropy_with_logits(student / temperature, target, reduction="none")
        return self._weighted_mean(loss, confidence) * temperature**2

    def _dfl_kd(self, student: torch.Tensor, teacher: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = student.shape
        temperature = self.temperature
        student = student.view(batch, 4, self.reg_max, height, width).float()
        teacher = teacher.detach().view(batch, 4, self.reg_max, height, width).float()
        teacher_probability = (teacher / temperature).softmax(dim=2)
        loss = F.kl_div(
            (student / temperature).log_softmax(dim=2),
            teacher_probability,
            reduction="none",
        ).sum(dim=2)
        return self._weighted_mean(loss, confidence.detach().pow(2)) * temperature**2

    def _decode_boxes(self, distribution: torch.Tensor, stride: float) -> torch.Tensor:
        batch, _, height, width = distribution.shape
        distribution = distribution.view(batch, 4, self.reg_max, height, width).float().softmax(dim=2)
        bins = torch.arange(self.reg_max, device=distribution.device, dtype=distribution.dtype).view(1, 1, -1, 1, 1)
        distance = (distribution * bins).sum(dim=2)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=distribution.device, dtype=distribution.dtype) + 0.5,
            torch.arange(width, device=distribution.device, dtype=distribution.dtype) + 0.5,
            indexing="ij",
        )
        center = torch.stack((grid_x, grid_y, grid_x, grid_y)).unsqueeze(0)
        signs = distribution.new_tensor((-1.0, -1.0, 1.0, 1.0)).view(1, 4, 1, 1)
        return (center + signs * distance) * float(stride)

    def _p2_to_p3(self, student_p3: torch.Tensor, teacher_p2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        student_box, student_cls = self._split(student_p3)
        teacher_box, teacher_cls = self._split(teacher_p2)
        _, _, p2_height, p2_width = teacher_p2.shape
        _, _, p3_height, p3_width = student_p3.shape
        if (p2_height, p2_width) != (p3_height * 2, p3_width * 2):
            raise ValueError(f"P2/P3 maps must have a 2x ratio, got {(p2_height, p2_width)} and {(p3_height, p3_width)}")

        teacher_probability = teacher_cls.detach().sigmoid()
        target_cls = F.max_pool2d(teacher_probability, kernel_size=2, stride=2)
        cls_weight = target_cls.amax(dim=1, keepdim=True).pow(2)
        cls_loss = self._weighted_mean(
            F.binary_cross_entropy_with_logits(student_cls, target_cls, reduction="none"), cls_weight
        )

        teacher_confidence = teacher_probability.amax(dim=1, keepdim=True)
        teacher_boxes = self._decode_boxes(teacher_box.detach(), stride=4.0)
        grouped_confidence = teacher_confidence.view(
            teacher_confidence.shape[0], 1, p3_height, 2, p3_width, 2
        ).permute(0, 1, 2, 4, 3, 5).reshape(teacher_confidence.shape[0], 1, p3_height, p3_width, 4)
        grouped_boxes = teacher_boxes.view(
            teacher_boxes.shape[0], 4, p3_height, 2, p3_width, 2
        ).permute(0, 1, 2, 4, 3, 5).reshape(teacher_boxes.shape[0], 4, p3_height, p3_width, 4)
        selected = grouped_confidence.argmax(dim=-1, keepdim=True)
        target_boxes = grouped_boxes.gather(4, selected.expand(-1, 4, -1, -1, -1)).squeeze(4)
        target_confidence = grouped_confidence.gather(4, selected).squeeze(4).pow(2)
        student_boxes = self._decode_boxes(student_box, stride=8.0)
        box_delta = F.smooth_l1_loss(student_boxes, target_boxes, reduction="none") / 8.0
        box_loss = self._weighted_mean(box_delta, target_confidence)
        return cls_loss, box_loss

    def _localized_feature_kd(self, batch) -> torch.Tensor:
        student_features = self.student_capture.features
        teacher_features = self.teacher_capture.features
        if student_features is None or teacher_features is None:
            raise RuntimeError("Detector feature capture did not run")
        student_p3 = student_features[1]
        teacher_p2 = teacher_features[0].detach()
        target = F.adaptive_avg_pool2d(teacher_p2, student_p3.shape[-2:])
        target = self.projection(target)

        batch_size, _, height, width = student_p3.shape
        mask = student_p3.new_zeros((batch_size, 1, height, width))
        image_height, image_width = batch["img"].shape[-2:]
        boxes = batch["bboxes"]
        long_edges = torch.maximum(boxes[:, 2] * image_width, boxes[:, 3] * image_height)
        selected = (long_edges >= 4.0) & (long_edges <= 8.0)
        regions = torch.cat((batch["batch_idx"].view(-1, 1), boxes), dim=1)[selected].detach().cpu().tolist()
        for index, cx, cy, bw, bh in regions:
            x1 = max(int((cx - bw / 2) * width) - 1, 0)
            y1 = max(int((cy - bh / 2) * height) - 1, 0)
            x2 = min(int((cx + bw / 2) * width + 0.9999) + 1, width)
            y2 = min(int((cy + bh / 2) * height + 0.9999) + 1, height)
            index = int(index)
            mask[index, :, y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)] = 1
        if not mask.any():
            return student_p3.sum() * 0.0
        student_normalized = F.normalize(student_p3.float(), dim=1)
        target_normalized = F.normalize(target.float(), dim=1)
        delta = F.smooth_l1_loss(student_normalized, target_normalized, reduction="none").mean(dim=1, keepdim=True)
        return (delta * mask).sum() / mask.sum().clamp_min(1.0)

    def __call__(self, predictions, batch):
        student_raw = predictions[1] if isinstance(predictions, tuple) else predictions
        if len(student_raw) != 4:
            raise ValueError(f"Expected four student levels during training, received {len(student_raw)}")
        primary_total, primary_items = self.primary(student_raw[1:], batch)
        auxiliary_total, auxiliary_items = self.auxiliary(student_raw[:1], batch)

        with torch.no_grad():
            teacher_output = self.teacher(batch["img"])
            teacher_raw = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
        if len(teacher_raw) != 4:
            raise ValueError(f"Expected four teacher levels, received {len(teacher_raw)}")

        p3_cls = student_raw[0].new_zeros(())
        p3_dfl = student_raw[0].new_zeros(())
        for student_level, teacher_level in zip(student_raw[1:], teacher_raw[1:]):
            student_box, student_cls = self._split(student_level)
            teacher_box, teacher_cls = self._split(teacher_level)
            p3_cls = p3_cls + self._classification_kd(student_cls, teacher_cls)
            confidence = teacher_cls.detach().sigmoid().amax(dim=1)
            p3_dfl = p3_dfl + self._dfl_kd(student_box, teacher_box, confidence)
        p3_cls = p3_cls / 3.0
        p3_dfl = p3_dfl / 3.0
        p2_cls, p2_box = self._p2_to_p3(student_raw[1], teacher_raw[0])
        feature = self._localized_feature_kd(batch)

        distillation = (
            self.weights["p3_cls"] * p3_cls
            + self.weights["p3_dfl"] * p3_dfl
            + self.weights["p2_cls"] * p2_cls
            + self.weights["p2_box"] * p2_box
            + self.weights["feature"] * feature
        )
        batch_size = student_raw[0].shape[0]
        total = primary_total + self.weights["auxiliary"] * auxiliary_total + batch_size * distillation
        items = primary_items + self.weights["auxiliary"] * auxiliary_items
        items[0] += self.weights["p2_box"] * p2_box.detach()
        items[1] += (
            self.weights["p3_cls"] * p3_cls
            + self.weights["p2_cls"] * p2_cls
            + self.weights["feature"] * feature
        ).detach()
        items[2] += self.weights["p3_dfl"] * p3_dfl.detach()
        return total, items


class P3StudentDistillationTrainer(FixedShapeGrayP3Trainer):
    """Single-GPU trainer configured by the experiment runner before construction."""

    teacher_weights: Path | None = None
    qat_start_epoch = 0
    qat_observer_freeze_epoch = 12
    enable_qat = True
    loss_options: dict = {}

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        detector = model.model[-1]
        if not isinstance(detector, FrozenP3AddOnP2Detect):
            raise TypeError("Distillation trainer requires the temporary four-scale student graph")
        detector.auxiliary_training_only = True
        attach_tiny_feature_projection(model)
        if self.enable_qat:
            wrapped = prepare_rknn_qat(model)
            if wrapped == 0:
                raise RuntimeError("No convolution was prepared for RKNN INT8 QAT")
            LOGGER.info("Prepared %d Conv2d layers for RKNN INT8 QAT", wrapped)
        return model

    def _setup_train(self, world_size):
        if world_size != 1:
            raise ValueError("P2-teacher distillation is intentionally restricted to one GPU")
        super()._setup_train(world_size)
        if not self.teacher_weights:
            raise ValueError("Configure P3StudentDistillationTrainer.teacher_weights before training")
        teacher = YOLO(str(self.teacher_weights)).model.float().to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        detector = teacher.model[-1]
        if not isinstance(detector, FrozenP3AddOnP2Detect):
            raise TypeError("Teacher checkpoint is not a P2+P3 model")
        detector.auxiliary_training_only = False
        self.teacher = teacher
        student = de_parallel(self.model)
        student.criterion = P3StudentDistillationLoss(student, teacher, **self.loss_options)
        LOGGER.info("Loaded P2+P3 teacher from %s", self.teacher_weights)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.enable_qat:
            enabled = self.epoch >= self.qat_start_epoch
            observe = self.epoch < self.qat_observer_freeze_epoch
            set_rknn_qat(de_parallel(self.model), enabled=enabled, observer_enabled=observe)
            if self.ema:
                set_rknn_qat(self.ema.ema, enabled=enabled, observer_enabled=False)
        self.teacher.eval()
        return batch

    def optimizer_step(self):
        super().optimizer_step()
        if self.enable_qat and self.ema:
            sync_rknn_qat_observer_flags(de_parallel(self.model), self.ema.ema)


class P3StudentFP32DistillationTrainer(P3StudentDistillationTrainer):
    """Distill without fake-quant noise; QAT is a later, short stage."""

    enable_qat = False


class TinyAwareAddOnTrainer(FixedShapeGrayAddOnTrainer):
    """Train the new Teacher P2 branch with bounded tiny-target weighting."""

    tiny_multiplier = 1.75

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        model = de_parallel(self.model)
        model.criterion = TinyAwareAddOnP2DetectionLoss(model, tiny_multiplier=self.tiny_multiplier)


class P3QATFineTuneTrainer(FixedShapeGrayP3Trainer):
    """Short pure-P3 QAT stage after FP32 distillation has converged."""

    tiny_multiplier = 1.75
    observer_freeze_epoch = 3

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        wrapped = prepare_rknn_qat(model)
        if wrapped == 0:
            raise RuntimeError("No convolution was prepared for pure-P3 QAT")
        return model

    def _setup_train(self, world_size):
        if world_size != 1:
            raise ValueError("RKNN QAT is intentionally restricted to one GPU")
        super()._setup_train(world_size)
        model = de_parallel(self.model)
        model.criterion = TinyAwareDetectionLoss(model, tiny_multiplier=self.tiny_multiplier)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        observe = self.epoch < self.observer_freeze_epoch
        set_rknn_qat(de_parallel(self.model), enabled=True, observer_enabled=observe)
        if self.ema:
            set_rknn_qat(self.ema.ema, enabled=True, observer_enabled=False)
        return batch

    def optimizer_step(self):
        super().optimizer_step()
        if self.ema:
            sync_rknn_qat_observer_flags(de_parallel(self.model), self.ema.ema)


def copy_teacher_auxiliary(teacher: DetectionModel, student: DetectionModel) -> dict[str, int]:
    """Initialize the disposable auxiliary branch from the trained teacher."""
    source = teacher.model[-1]
    target = student.model[-1]
    if not all(isinstance(value, FrozenP3AddOnP2Detect) for value in (source, target)):
        raise TypeError("Teacher and student must both use FrozenP3AddOnP2Detect")
    student.model[-2].load_state_dict(deepcopy(teacher.model[-2].state_dict()))
    target.cv2[0].load_state_dict(deepcopy(source.cv2[0].state_dict()))
    target.cv3[0].load_state_dict(deepcopy(source.cv3[0].state_dict()))
    return {
        "adapter_tensors": len(student.model[-2].state_dict()),
        "p2_box_tensors": len(target.cv2[0].state_dict()),
        "p2_cls_tensors": len(target.cv3[0].state_dict()),
    }


def transfer_student_to_p3(source: DetectionModel, target: DetectionModel) -> dict[str, int]:
    """Remove the temporary P2 graph and copy the trained P3 backbone/head exactly."""
    source = qat_copy(source, bake_weights=True).cpu().float()
    source_state = source.state_dict()
    target_state = target.state_dict()
    source_head = len(source.model) - 1
    target_head = len(target.model) - 1
    transferred = {
        key: value for key, value in source_state.items() if key in target_state and value.shape == target_state[key].shape
    }
    for source_key, value in source_state.items():
        prefix = f"model.{source_head}."
        if not source_key.startswith(prefix):
            continue
        parts = source_key[len(prefix) :].split(".")
        if parts[0] in {"cv2", "cv3"} and len(parts) >= 2 and int(parts[1]) >= 1:
            parts[1] = str(int(parts[1]) - 1)
            target_key = f"model.{target_head}." + ".".join(parts)
        elif parts[0] == "dfl":
            target_key = f"model.{target_head}." + ".".join(parts)
        else:
            continue
        if target_key in target_state and target_state[target_key].shape == value.shape:
            transferred[target_key] = value
    missing = set(target_state).difference(transferred)
    if missing:
        raise RuntimeError(f"Pure P3 extraction missed {len(missing)} tensors: {sorted(missing)[:3]}")
    target.load_state_dict(transferred, strict=True)
    return {"transferred_tensors": len(transferred), "target_tensors": len(target_state)}


def export_pure_p3_student(qat_checkpoint: Path, output: Path, p3_cfg: Path) -> dict[str, object]:
    """Write a portable standard YOLOv8 P3 checkpoint with baked QAT weights."""
    source_wrapper = YOLO(str(qat_checkpoint))
    source = source_wrapper.model.float().cpu().eval()
    source.model[-1].auxiliary_training_only = True
    target = DetectionModel(str(p3_cfg), nc=source.nc, verbose=False).float().cpu().eval()
    target.names = source.names
    target.args = dict(source.args)
    transfer = transfer_student_to_p3(source, target)

    torch.manual_seed(20261009)
    sample = torch.rand(1, 3, 64, 64)
    clean_source = qat_copy(source, bake_weights=True).eval()
    with torch.inference_mode():
        source_raw = clean_source(sample)[1]
        target_raw = target(sample)[1]
    extraction_errors = [float((left - right).abs().max()) for left, right in zip(source_raw, target_raw)]
    if len(source_raw) != 3 or len(target_raw) != 3 or any(error != 0.0 for error in extraction_errors):
        raise RuntimeError(
            f"Pure P3 extraction regression: source={len(source_raw)}, target={len(target_raw)}, "
            f"errors={extraction_errors}"
        )

    wrapper = YOLO(str(p3_cfg))
    wrapper.model = target
    wrapper.ckpt = {
        "train_args": dict(target.args),
        "source_qat_checkpoint": str(qat_checkpoint.resolve()),
        "training_method": "P2 teacher -> P3 student + auxiliary P2 + RKNN INT8 QAT",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(output)

    reloaded = YOLO(str(output)).model.float().cpu().eval()
    # YOLO.save serializes weights as FP16. Compare against the same explicit
    # round trip rather than mistaking expected FP16 rounding for a graph error.
    serialized_reference = deepcopy(target).half().float().eval()
    with torch.inference_mode():
        reference_raw = serialized_reference(sample)[1]
        reloaded_raw = reloaded(sample)[1]
    serialization_errors = [float((left - right).abs().max()) for left, right in zip(reference_raw, reloaded_raw)]
    if len(reference_raw) != 3 or len(reloaded_raw) != 3 or any(error != 0.0 for error in serialization_errors):
        raise RuntimeError(
            f"Pure P3 serialization regression: reference={len(reference_raw)}, reloaded={len(reloaded_raw)}, "
            f"errors={serialization_errors}"
        )
    return {
        "source": str(qat_checkpoint.resolve()),
        "output": str(output.resolve()),
        "transfer": transfer,
        "levels": 3,
        "extraction_max_abs_error": extraction_errors,
        "serialization_max_abs_error": serialization_errors,
        "parameters": sum(parameter.numel() for parameter in reloaded.parameters()),
    }


def export_standard_p3_qat(qat_checkpoint: Path, output: Path) -> dict[str, object]:
    """Bake fake-quantized weights from a standard three-scale P3 checkpoint."""
    source_wrapper = YOLO(str(qat_checkpoint))
    source = source_wrapper.model.float().cpu().eval()
    if len(source.model[-1].stride) != 3:
        raise ValueError("Expected a standard three-scale P3 checkpoint")
    target = qat_copy(source, bake_weights=True).eval()
    wrapper = YOLO(str(qat_checkpoint))
    wrapper.model = target
    wrapper.ckpt = {
        "train_args": dict(target.args),
        "source_qat_checkpoint": str(qat_checkpoint.resolve()),
        "training_method": "FP32 P2-to-P3 distillation followed by short pure-P3 RKNN INT8 QAT",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(output)

    torch.manual_seed(20261010)
    image = torch.rand(1, 3, 64, 64)
    reference = deepcopy(target).half().float().eval()
    reloaded = YOLO(str(output)).model.float().cpu().eval()
    with torch.inference_mode():
        reference_raw = reference(image)[1]
        reloaded_raw = reloaded(image)[1]
    errors = [float((left - right).abs().max()) for left, right in zip(reference_raw, reloaded_raw)]
    if len(reference_raw) != 3 or len(reloaded_raw) != 3 or any(error != 0.0 for error in errors):
        raise RuntimeError(f"Pure P3 QAT serialization regression: {errors}")
    return {
        "source": str(qat_checkpoint.resolve()),
        "output": str(output.resolve()),
        "levels": 3,
        "serialization_max_abs_error": errors,
        "parameters": sum(parameter.numel() for parameter in reloaded.parameters()),
    }
