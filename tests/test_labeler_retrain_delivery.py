from pathlib import Path

from scripts.anti_uav.run_labeler_retrain_delivery import metric_table, training_command


def test_training_starts_from_supplied_public_foundation_and_disables_qat_stages():
    command = training_command("python", "data.yaml", "yolov8n.pt", "fresh_run", "0", 73)
    assert command[command.index("--foundation") + 1] == "yolov8n.pt"
    assert command[command.index("--expected-training-videos") + 1] == "73"
    assert command[command.index("--localization-epochs") + 1] == "0"
    assert command[command.index("--foundation-epochs") + 1] == "30"
    assert command[command.index("--teacher-epochs") + 1] == "12"
    assert command[command.index("--student-epochs") + 1] == "15"
    assert not any("baseline" in value or "qat" in value for value in command)


def test_delivery_metrics_recall_first_and_do_not_substitute_old_metrics():
    q = "native/c0.03/"
    values = {q + "R": .75, q + "P": .8, q + "FP": 20, q + "FN": 25,
              "native/mAP50": .7, "native/mAP50-95": .5}
    text = metric_table({"new_student": dict(splits=dict(pooled_native=values))})
    assert "| Model | Recall | Precision |" in text
    assert "| new_student | 75.00% | 80.00% | 20 | 25 | 70.00% | 50.00% |" in text


def test_template_clearly_distinguishes_host_checks_from_board_exposure_measurement():
    root = Path(__file__).resolve().parents[1]
    template = root / "scripts/anti_uav/templates/RK3576_LABELER_HANDOVER_ZH.md"
    if not template.exists():
        root = Path.cwd()
        template = root / "scripts/anti_uav/templates/RK3576_LABELER_HANDOVER_ZH.md"
    text = template.read_text()
    assert "200 帧宿主机量化仿真" in text
    assert "尚未实测 RK3576 FPS" in text
    assert "曝光开始到结果" in text
    assert "--npu-masks 0,1" in text
