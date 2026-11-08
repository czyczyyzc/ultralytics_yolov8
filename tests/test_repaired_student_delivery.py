import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.anti_uav.deliver_regression_repair_student import (
    evaluated_binary, metric_table, ready_state, sha256, validate_inputs,
)
from scripts.anti_uav.prepare_stratified_rknn_calibration import calibration_pixels, letterbox


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_gray_calibration_matches_gray_after_letterbox_and_preserves_padding():
    image = np.zeros((10, 40, 3), dtype=np.uint8)
    image[:, :20] = [10, 90, 240]
    image[:, 20:] = [220, 100, 30]
    canvas = calibration_pixels(image, 32, 64, "gray3")
    expected = cv2.cvtColor(letterbox(image, 32, 64), cv2.COLOR_BGR2GRAY)
    assert canvas.shape == (32, 64, 3)
    assert np.array_equal(canvas[..., 0], expected)
    assert np.array_equal(canvas[..., 0], canvas[..., 1])
    assert np.array_equal(canvas[..., 1], canvas[..., 2])
    assert np.all(canvas[:8] == 114)


def test_source_color_calibration_remains_unchanged():
    image = np.full((20, 30, 3), [12, 90, 210], dtype=np.uint8)
    assert np.array_equal(calibration_pixels(image, 32, 64, "source"), letterbox(image, 32, 64))
    with pytest.raises(ValueError, match="input color"):
        calibration_pixels(image, 32, 64, "invalid")


@pytest.mark.parametrize("training,supervisor,ready", [
    ("distill_fp32", "training", False), ("complete", "training", False),
    ("complete", "complete", True),
])
def test_delivery_waits_for_training_and_final_comparison(tmp_path, training, supervisor, ready):
    write(tmp_path / "status.json", dict(stage=training, epoch=9))
    write(tmp_path / "repair_launch_status.json", dict(stage=supervisor))
    assert ready_state(tmp_path)[0] is ready


@pytest.mark.parametrize("failed", ["status.json", "repair_launch_status.json"])
def test_failed_training_or_comparison_cannot_trigger_delivery(tmp_path, failed):
    for name in ("status.json", "repair_launch_status.json"):
        write(tmp_path / name, dict(stage="failed" if name == failed else "complete"))
    with pytest.raises(RuntimeError, match="failed"):
        ready_state(tmp_path)


def test_data_contract_rejects_qat_rgb_or_unpinned_files(tmp_path):
    data = tmp_path / "data.yaml"
    write(data, dict(input_color="gray3", tiny_sampling=dict(policy="fixed_domain_v1")))
    protocol = dict(regression_repair={"options": {}}, qat=False, input_color="gray3",
                    training_videos=74, input_hw=[544, 960], task_specific_initialization=False,
                    data=str(data), trusted_label_caches={})
    write(tmp_path / "protocol.json", protocol)
    write(tmp_path / "experiment_contract.json", dict(same_data=str(data), training_videos=74,
                                                     pinned={str(data): sha256(data)}))
    validate_inputs(tmp_path)
    for updates in (dict(qat=True), dict(input_color="source"), dict(training_videos=73)):
        write(tmp_path / "protocol.json", dict(protocol, **updates))
        with pytest.raises(ValueError, match="curriculum"):
            validate_inputs(tmp_path)
    write(tmp_path / "protocol.json", protocol)
    data.write_text("changed")
    with pytest.raises(ValueError, match="Pinned"):
        validate_inputs(tmp_path)


def quantization_result(tmp_path):
    binary = tmp_path / "rebuilt_for_simulator.rknn"
    binary.write_bytes(b"actually-evaluated-rk3576")
    splits = {key: {"evaluated_frames": 100} for key in ("Video00004_test", "Video00009_selection_subset")}
    protocol = dict(target="rk3576", input_color="gray3", quantized_dtype="w8a8", auto_hybrid=False,
                    rebuilt_sha256=sha256(binary), datasets=splits)
    result = dict(protocol=protocol, raw_export_max_error=dict(class_score=1e-6),
                  backends={backend: {split: {"native/c0.03/FRAMES": 100} for split in splits}
                            for backend in ("pt_fp32", "onnx_fp32", "rknn_int8_simulator")})
    write(tmp_path / "status.json", dict(stage="complete"))
    write(tmp_path / "results.json", result)
    return result


def test_ship_only_the_binary_actually_evaluated(tmp_path):
    quantization_result(tmp_path)
    binary, _ = evaluated_binary(tmp_path)
    binary.write_bytes(b"different-model")
    with pytest.raises(ValueError, match="artifact"):
        evaluated_binary(tmp_path)


@pytest.mark.parametrize("updates", [dict(target="rk3588"), dict(input_color="source"), dict(auto_hybrid=True)])
def test_reject_wrong_platform_color_and_mixed_precision(tmp_path, updates):
    result = quantization_result(tmp_path)
    result["protocol"].update(updates)
    write(tmp_path / "results.json", result)
    with pytest.raises(ValueError, match="artifact"):
        evaluated_binary(tmp_path)


def test_partial_quantization_or_broken_export_cannot_be_shipped(tmp_path):
    result = quantization_result(tmp_path)
    result["backends"]["rknn_int8_simulator"]["Video00004_test"]["native/c0.03/FRAMES"] = 99
    write(tmp_path / "results.json", result)
    with pytest.raises(ValueError, match="coverage"):
        evaluated_binary(tmp_path)
    result = quantization_result(tmp_path)
    result["raw_export_max_error"]["class_score"] = .1
    write(tmp_path / "results.json", result)
    with pytest.raises(ValueError, match="parity"):
        evaluated_binary(tmp_path)


def test_handover_has_no_claimed_board_speed_and_puts_recall_first():
    template = Path("scripts/anti_uav/templates/RK3576_REPAIR_HANDOVER_ZH.md").read_text()
    assert "尚未进行本版本 RK3576 真机" in template
    assert "--workers 2 --inflight 2 --npu-masks 0,1" in template
    assert "observations" in template
    assert "--detector-only" in template
    assert "--tracking none" not in template
    metrics = {"m": {"splits": {"pooled": {"native/c0.03/R": .7, "native/c0.03/P": .8,
                 "native/c0.03/FP": 5, "native/mAP50": .6, "native/mAP50-95": .4}}}}
    table = metric_table(metrics, "pooled", {"m": "Student"})
    assert "Recall | Precision" in table
    assert "70.00% | 80.00% | 5" in table
