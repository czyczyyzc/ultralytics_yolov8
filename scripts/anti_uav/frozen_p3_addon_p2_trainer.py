"""Trainer utilities for a Frozen-P3 + Add-on P2 detector."""

from __future__ import annotations

import torch

from scripts.anti_uav.lovo_detection_trainer import LovoDetectionTrainer
from ultralytics.nn.modules import FrozenP3AddOnP2Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel


def add_on_parameter_ids(model: DetectionModel) -> set[int]:
    detector = model.model[-1]
    if not isinstance(detector, FrozenP3AddOnP2Detect):
        raise TypeError(f"Expected FrozenP3AddOnP2Detect, received {type(detector).__name__}")
    adapter = model.model[-2]
    return {
        id(parameter)
        for module in (adapter, detector.cv2[detector.addon_index], detector.cv3[detector.addon_index])
        for parameter in module.parameters()
    }


def is_add_on_state(name: str, model: DetectionModel) -> bool:
    adapter_index = len(model.model) - 2
    detector_index = len(model.model) - 1
    return name.startswith(f"model.{adapter_index}.") or name.startswith(
        (f"model.{detector_index}.cv2.0.", f"model.{detector_index}.cv3.0.")
    )


class FrozenP3AddOnP2Trainer(LovoDetectionTrainer):
    """Optimize only the P2 adapter and its decoupled box/classification towers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This run is fully local. Older W&B versions ignore WANDB_MODE=disabled
        # and reject long filesystem project paths before training can start.
        for event, event_callbacks in self.callbacks.items():
            self.callbacks[event] = [
                callback
                for callback in event_callbacks
                if callback.__module__ != "ultralytics.utils.callbacks.wb"
            ]

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        base_model = de_parallel(model)
        trainable_ids = add_on_parameter_ids(base_model)
        trainable_names = []
        trainable_count = 0
        total_count = 0
        for parameter_name, parameter in base_model.named_parameters():
            parameter.requires_grad = id(parameter) in trainable_ids
            total_count += parameter.numel()
            if parameter.requires_grad:
                trainable_names.append(parameter_name)
                trainable_count += parameter.numel()
        if not trainable_names:
            raise RuntimeError("No Add-on P2 parameters were selected")
        LOGGER.info(
            "Frozen-P3 + Add-on P2: %d/%d parameters trainable across %d tensors",
            trainable_count,
            total_count,
            len(trainable_names),
        )
        LOGGER.info("Trainable tensors: %s", ", ".join(trainable_names))
        return super().build_optimizer(model, name, lr, momentum, decay, iterations)

    def preprocess_batch(self, batch):
        # BaseTrainer calls model.train() each epoch. Keep every frozen BN fixed while
        # enabling raw Detect output and training state only for the new P2 modules.
        model = de_parallel(self.model)
        model.eval()
        adapter = model.model[-2]
        detector = model.model[-1]
        adapter.train()
        detector.training = True
        detector.cv2[detector.addon_index].train()
        detector.cv3[detector.addon_index].train()
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        super().optimizer_step()
        # EMA arithmetic can move an unchanged FP32 tensor by one ULP. Restore all
        # frozen parameters and buffers so saved checkpoints preserve P3 bit-for-bit.
        if self.ema:
            model = de_parallel(self.model)
            model_state = model.state_dict()
            ema_state = self.ema.ema.state_dict()
            with torch.no_grad():
                for name, value in model_state.items():
                    if not is_add_on_state(name, model):
                        ema_state[name].copy_(value)
