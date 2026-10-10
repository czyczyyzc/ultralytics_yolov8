from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.anti_uav.p3_student_distillation import (
    P3StudentDistillationLoss,
    transfer_student_to_p3,
)
from scripts.anti_uav.tiny_aware_loss import TinyAwareAddOnP2DetectionLoss, TinyAwareDetectionLoss
from scripts.anti_uav.rknn_qat import (
    RKNNFakeQuantConv2d,
    prepare_rknn_qat,
    qat_copy,
    set_rknn_qat,
    strip_rknn_qat,
    sync_rknn_qat_observer_flags,
)
from scripts.anti_uav.train_frozen_p3_addon_p2 import (
    initialize_addon_model,
    transfer_frozen_p3_weights,
    verify_legacy_outputs,
)
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]


def build_pair():
    p3 = DetectionModel(ROOT / "ultralytics/cfg/models/v8/yolov8.yaml", nc=1, verbose=False)
    addon = DetectionModel(
        ROOT / "ultralytics/cfg/models/v8/yolov8-frozen-p3-addon-p2.yaml", nc=1, verbose=False
    )
    transfer_frozen_p3_weights(p3, addon)
    return p3, addon


def test_auxiliary_p2_is_not_used_for_student_validation():
    p3, addon = build_pair()
    addon.model[-1].auxiliary_training_only = True
    p3.eval()
    addon.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        p3_output, p3_raw = p3(image)
        addon_output, addon_raw = addon(image)
    assert len(addon_raw) == 3
    torch.testing.assert_close(addon_output, p3_output, rtol=0, atol=0)
    for left, right in zip(addon_raw, p3_raw):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_legacy_regression_temporarily_unfolds_auxiliary_p2():
    p3, addon = build_pair()
    addon.model[-1].auxiliary_training_only = True
    report = verify_legacy_outputs(p3, addon, size=64)
    assert report["reference_levels"] == 3
    assert report["addon_levels"] == 4
    assert report["bit_exact"]
    assert addon.model[-1].auxiliary_training_only is True


def test_addon_initialization_reads_class_count_from_standard_detect_head(tmp_path):
    p3, _ = build_pair()
    assert not hasattr(p3, "nc")
    wrapper = YOLO(str(ROOT / "ultralytics/cfg/models/v8/yolov8.yaml"))
    wrapper.model = p3
    wrapper.ckpt = {}
    source = tmp_path / "p3.pt"
    wrapper.save(source)
    initialized, report = initialize_addon_model(
        source,
        ROOT / "ultralytics/cfg/models/v8/yolov8-frozen-p3-addon-p2.yaml",
        tmp_path / "addon.pt",
    )
    assert initialized.model.model[-1].nc == 1
    assert report["legacy_regression"]["bit_exact"]


def test_rknn_qat_wraps_toggles_and_strips_convolutions():
    p3, _ = build_pair()
    original = p3.model[0].conv.weight.detach().clone()
    count = prepare_rknn_qat(p3)
    assert count > 0 and isinstance(p3.model[0].conv, RKNNFakeQuantConv2d)
    set_rknn_qat(p3, enabled=True)
    p3.train()(torch.rand(1, 3, 64, 64))
    assert bool(p3.model[0].conv.observer_initialized)
    ema = __import__("copy").deepcopy(p3)
    ema.model[0].conv.observer_initialized.fill_(False)
    sync_rknn_qat_observer_flags(p3, ema)
    assert bool(ema.model[0].conv.observer_initialized)
    strip_rknn_qat(p3, bake_weights=True)
    assert not any(isinstance(module, RKNNFakeQuantConv2d) for module in p3.modules())
    assert not torch.equal(p3.model[0].conv.weight, original)


def test_distillation_loss_backpropagates_and_pure_p3_extraction_is_exact():
    p3, student = build_pair()
    _, teacher = build_pair()
    student.train()
    teacher.eval()
    student.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    criterion = P3StudentDistillationLoss(student, teacher)
    image = torch.rand(1, 3, 64, 64)
    prediction = student(image)
    batch = dict(
        img=image,
        batch_idx=torch.tensor([0.0]),
        cls=torch.tensor([[0.0]]),
        bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    )
    loss, items = criterion(prediction, batch)
    assert torch.isfinite(loss) and items.shape == (3,)
    loss.backward()
    assert student.model[0].conv.weight.grad is not None

    prepare_rknn_qat(student)
    set_rknn_qat(student, enabled=True)
    student.eval()
    student.model[-1].auxiliary_training_only = True
    target = DetectionModel(ROOT / "ultralytics/cfg/models/v8/yolov8.yaml", nc=1, verbose=False).eval()
    report = transfer_student_to_p3(student, target)
    assert report["transferred_tensors"] == report["target_tensors"]
    clean_student = qat_copy(student, bake_weights=True).eval()
    with torch.inference_mode():
        source_raw = clean_student(image)[1]
        target_raw = target(image)[1]
    assert len(source_raw) == len(target_raw) == 3
    for source_level, target_level in zip(source_raw, target_raw):
        torch.testing.assert_close(source_level, target_level, rtol=0, atol=0)


def test_tiny_aware_loss_is_bounded_and_backpropagates():
    p3, _ = build_pair()
    p3.train()
    p3.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    criterion = TinyAwareDetectionLoss(p3, tiny_multiplier=1.75)
    image = torch.rand(2, 3, 64, 64)
    prediction = p3(image)
    batch = dict(
        img=image,
        batch_idx=torch.tensor([0.0, 1.0]),
        cls=torch.tensor([[0.0], [0.0]]),
        bboxes=torch.tensor([[.5, .5, 6 / 64, 6 / 64], [.5, .5, .25, .25]]),
    )
    loss, items = criterion(prediction, batch)
    assert torch.isfinite(loss) and items.shape == (3,)
    loss.backward()
    assert p3.model[0].conv.weight.grad is not None


def test_tiny_aware_addon_loss_uses_only_p2_level():
    _, addon = build_pair()
    addon.train()
    addon.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    image = torch.rand(1, 3, 64, 64)
    batch = dict(
        img=image,
        batch_idx=torch.tensor([0.0]),
        cls=torch.tensor([[0.0]]),
        bboxes=torch.tensor([[.5, .5, 6 / 64, 6 / 64]]),
    )
    loss, items = TinyAwareAddOnP2DetectionLoss(addon)(addon(image), batch)
    assert torch.isfinite(loss) and items.shape == (3,)
