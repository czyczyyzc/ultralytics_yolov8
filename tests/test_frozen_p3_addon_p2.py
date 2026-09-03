from pathlib import Path

from scripts.anti_uav.train_frozen_p3_addon_p2 import (
    add_on_parameter_ids,
    transfer_frozen_p3_weights,
    verify_legacy_outputs,
)
from ultralytics.nn.modules import FrozenP3AddOnP2Detect
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]


def build_models():
    source = DetectionModel(ROOT / "ultralytics/cfg/models/v8/yolov8.yaml", nc=1, verbose=False)
    addon = DetectionModel(
        ROOT / "ultralytics/cfg/models/v8/yolov8-frozen-p3-addon-p2.yaml", nc=1, verbose=False
    )
    transfer_frozen_p3_weights(source, addon)
    return source, addon


def test_frozen_p3_outputs_are_bit_exact_after_transfer():
    source, addon = build_models()
    report = verify_legacy_outputs(source, addon, size=64)
    assert report["bit_exact"]
    assert report["legacy_max_abs_error"] == [0.0, 0.0, 0.0]


def test_only_adapter_and_p2_towers_are_selected_for_training():
    _, addon = build_models()
    detector = addon.model[-1]
    assert isinstance(detector, FrozenP3AddOnP2Detect)
    selected = add_on_parameter_ids(addon)
    expected = {
        id(parameter)
        for module in (addon.model[-2], detector.cv2[0], detector.cv3[0])
        for parameter in module.parameters()
    }
    legacy = {
        id(parameter)
        for module in (detector.cv2[1:], detector.cv3[1:])
        for parameter in module.parameters()
    }
    assert selected == expected
    assert selected.isdisjoint(legacy)
