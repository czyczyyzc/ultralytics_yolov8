import csv
import json

import pytest

from scripts.anti_uav.run_labeler_retrain_delivery import augmentation_policy, completed_student_metadata


def completed_run(tmp_path):
    (tmp_path / "audit").mkdir()
    student = tmp_path / "p3_student_distilled_best_fp32.pt"
    student.write_bytes(b"pure-p3-best")
    records = {
        "status.json": {"stage": "complete"},
        "selection.json": {"best_fp32": "student_fp32", "best_fp32_path": str(student)},
        "protocol.json": {"data": "/frozen73/train_legacy52_gray_monitor.yaml", "input_hw": [544, 960],
                          "epochs": dict(foundation=30, teacher=12, student=15, localization=0),
                          "qat": False, "task_specific_initialization": False},
        "audit/student_exports.json": {"best": {"levels": 3, "output": str(student),
            "extraction_max_abs_error": [0, 0, 0], "serialization_max_abs_error": [0, 0, 0]}},
    }
    for name, value in records.items():
        (tmp_path / name).write_text(json.dumps(value))
    for stage, epochs in [("foundation/p3", 30), ("teacher/p2p3", 12), ("distill_fp32/p3_student", 15)]:
        directory = tmp_path / stage
        directory.mkdir(parents=True)
        with (directory / "results.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "native/c0.03/F2", "native/mAP50", "native/mAP50-95"])
            for epoch in range(1, epochs + 1):
                writer.writerow([epoch, .9 if epoch == 3 else .5, .5, .3])
    return tmp_path


def test_completed_export_uses_actual_config_and_epoch_without_training(tmp_path):
    result = completed_student_metadata(completed_run(tmp_path))
    assert result["best_epoch"] == 3
    assert result["data"].endswith("train_legacy52_gray_monitor.yaml")
    assert result["student"].endswith("p3_student_distilled_best_fp32.pt")
    assert len(result["student_sha256"]) == 64


def test_completed_export_rejects_partial_training(tmp_path):
    run = completed_run(tmp_path)
    (run / "status.json").write_text(json.dumps({"stage": "distill_fp32"}))
    with pytest.raises(ValueError, match="completed"):
        completed_student_metadata(run)


def test_completed_export_rejects_missing_or_duplicate_epochs(tmp_path):
    run = completed_run(tmp_path)
    (run / "teacher/p2p3/results.csv").write_text("epoch\n1\n2\n2\n")
    with pytest.raises(ValueError, match="epochs"):
        completed_student_metadata(run)


def test_completed_export_requires_bit_exact_three_scale_model(tmp_path):
    run = completed_run(tmp_path)
    path = run / "audit/student_exports.json"
    data = json.loads(path.read_text())
    data["best"]["extraction_max_abs_error"] = [.01, 0, 0]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="regression"):
        completed_student_metadata(run)


def test_historical_manifest_must_not_mislabel_actual_legacy_training():
    manifest = {"sampler_audit": {"policy": "balanced_v2"}}
    config = {"tiny_sampling": {"input_hw": [544, 960], "tiny_min_px": 4., "tiny_max_px": 8.}}
    text = augmentation_policy(manifest, {}, config)
    assert "legacy TinyAwareSampler" in text
    assert "不启用balanced_v2" in text
    assert "采样使用balanced_v2" not in text


def test_completed_export_rejects_different_selected_model(tmp_path):
    run = completed_run(tmp_path)
    (run / "selection.json").write_text(json.dumps({"best_fp32": "student_localization_fp32"}))
    with pytest.raises(ValueError, match="completed"):
        completed_student_metadata(run)
