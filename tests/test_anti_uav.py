# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from ultralytics import solutions
from scripts.anti_uav.train_nanotrack_local import parse_device_spec


class FakeTracker:
    def __init__(self, responses):
        self.responses = list(responses)
        self.initialized = []
        self.reset_count = 0

    def init(self, frame, bbox):
        del frame
        self.initialized.append(tuple(bbox))

    def update(self, frame):
        del frame
        if self.responses:
            return self.responses.pop(0)
        return False, None, 0.0

    def reset(self):
        self.reset_count += 1


class DummyDetector:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def detect(self, frame, roi=None, prefer_roi=False):
        del frame
        self.calls.append({"roi": roi, "prefer_roi": prefer_roi})
        if self.responses:
            return self.responses.pop(0)
        return []


def test_anti_uav_detects_and_tracks():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(50, 60, 90, 100), confidence=0.9, class_id=0, class_name="drone")],
            [],
        ]
    )
    tracker = FakeTracker([(True, (52, 61, 92, 101), 0.88)])

    system = solutions.AntiUAVSystem(detector, tracker=tracker, detect_interval=10)

    first = system.step(frame)
    second = system.step(frame)

    assert first.status == "detected"
    assert first.bbox == (50.0, 60.0, 90.0, 100.0)
    assert second.status == "tracking"
    assert second.track_score == 0.88
    assert second.age == 2


def test_anti_uav_reacquires_after_tracker_loss():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(40, 40, 80, 80), confidence=0.85, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(44, 44, 84, 84), confidence=0.87, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker([(False, None, 0.05)])

    system = solutions.AntiUAVSystem(detector, tracker=tracker, detect_interval=20)

    first = system.step(frame)
    second = system.step(frame)

    assert first.status == "detected"
    assert second.status == "reacquired"
    assert second.bbox == (44.0, 44.0, 84.0, 84.0)
    assert second.lost_frames == 0
    assert detector.calls[1]["roi"] == (40.0, 40.0, 80.0, 80.0)


def test_anti_uav_drops_target_after_too_many_misses():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = DummyDetector([[solutions.Detection(bbox=(20, 20, 60, 60), confidence=0.9, class_id=0, class_name="drone")], [], []])
    tracker = FakeTracker([(False, None, 0.0), (False, None, 0.0)])

    system = solutions.AntiUAVSystem(detector, tracker=tracker, detect_interval=1, max_lost=1)

    first = system.step(frame)
    second = system.step(frame)
    third = system.step(frame)

    assert first.status == "detected"
    assert second.status == "lost"
    assert third.status == "searching"
    assert third.bbox is None


def test_manual_confirmation_emits_alert_event():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detector = DummyDetector([[solutions.Detection(bbox=(10, 10, 30, 30), confidence=0.95, class_id=0, class_name="drone")]])
    tracker = FakeTracker([])

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=5,
        pending_frames=1,
        min_confirm_detections=1,
        manual_confirmation=True,
    )
    state = system.step(frame)
    event = system.confirm_current_target(True, note="unit_test_confirm")
    events = system.drain_alerts()

    assert state.confirmation_state == "pending"
    assert event is not None
    assert event.event_type == "alert_raised"
    assert events[0].event_type == "alert_raised"


def test_confirmation_requires_multiple_detection_hits():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(10, 10, 30, 30), confidence=0.95, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(11, 10, 31, 30), confidence=0.93, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker(
        [
            (True, (10, 10, 30, 30), 0.88),
            (True, (11, 10, 31, 30), 0.86),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        pending_frames=2,
        min_confirm_detections=2,
        manual_confirmation=True,
    )

    first = system.step(frame)
    second = system.step(frame)
    third = system.step(frame)

    assert first.confirmation_state == "idle"
    assert second.confirmation_state == "idle"
    assert second.confirmation_hits == 2
    assert second.detection_hits == 1
    assert third.status == "redetected"
    assert third.confirmation_state == "pending"
    assert third.detection_hits == 2


def test_association_gate_rejects_far_redetection_and_keeps_track():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(20, 20, 40, 40), confidence=0.95, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(90, 70, 120, 100), confidence=0.98, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker(
        [
            (True, (21, 20, 41, 40), 0.92),
            (True, (22, 20, 42, 40), 0.90),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        tracker_score_thresh=0.4,
        min_confidence=0.45,
        manual_confirmation=False,
    )

    first = system.step(frame)
    second = system.step(frame)
    third = system.step(frame)

    assert first.status == "detected"
    assert second.status == "tracking"
    assert third.status == "tracking"
    assert third.bbox == (22.0, 20.0, 42.0, 40.0)
    assert third.detection_hits == 1
    assert detector.calls[-1]["roi"] == (22.0, 20.0, 42.0, 40.0)


def test_tracker_registry_contains_expected_defaults():
    names = solutions.available_trackers()
    assert "template_match" in names
    assert "opencv" in names
    assert "nanotrack" in names
    assert isinstance(solutions.build_tracker("template_match"), solutions.TemplateMatchTracker)


def test_iter_tiles_covers_full_frame():
    tiles = solutions.iter_tiles((100, 120), tile_size=(64, 64), overlap=0.25)
    assert tiles
    assert tiles[0] == (0.0, 0.0, 64.0, 64.0)
    assert any(tile[2] == 120.0 for tile in tiles)
    assert any(tile[3] == 100.0 for tile in tiles)


def test_convert_anti_uav300_nanotrack_exports_expected_layout(tmp_path):
    source_root = tmp_path / "Anti-UAV300" / "train"
    sequence_root = source_root / "seq001"
    sequence_root.mkdir(parents=True)

    video_path = sequence_root / "rgb.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    for frame_index in range(3):
        frame = np.full((48, 64, 3), frame_index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    annotation_path = sequence_root / "rgb_label.json"
    annotation_path.write_text(json.dumps({"gt_rect": [[10, 12, 8, 6], [11, 12, 8, 6], []]}), encoding="utf-8")

    output_root = tmp_path / "nanotrack_export"
    converter = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "convert_anti_uav300_nanotrack.py"
    subprocess.run(
        [
            sys.executable,
            str(converter),
            "--source-root",
            str(tmp_path / "Anti-UAV300"),
            "--output-root",
            str(output_root),
            "--modalities",
            "rgb",
            "--frame-step",
            "1",
        ],
        check=True,
    )

    train_json = output_root / "rgb" / "train.json"
    val_json = output_root / "rgb" / "val.json"
    crop_path = output_root / "rgb" / "crop511" / "train_seq001" / "000000.00.x.jpg"

    assert train_json.exists() or val_json.exists()
    assert crop_path.exists()
    metadata = json.loads((train_json if train_json.stat().st_size else val_json).read_text(encoding="utf-8"))
    assert "train_seq001" in metadata
    assert "00" in metadata["train_seq001"]
    assert "000000" in metadata["train_seq001"]["00"]


def test_train_nanotrack_local_smoke(tmp_path):
    crop_root = tmp_path / "rgb" / "crop511" / "train_seq001"
    crop_root.mkdir(parents=True)

    frame = np.full((255, 255, 3), 127, dtype=np.uint8)
    for frame_index in range(2):
        cv2.imwrite(str(crop_root / f"{frame_index:06d}.00.x.jpg"), frame)

    train_json = tmp_path / "rgb" / "train.json"
    train_json.write_text(
        json.dumps({"train_seq001": {"00": {"000000": [100, 100, 40, 32], "000001": [102, 101, 40, 32]}}}),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    run_root = tmp_path / "run"
    writer = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "write_nanotrack_config.py"
    trainer = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "train_nanotrack_local.py"

    subprocess.run(
        [
            sys.executable,
            str(writer),
            "--output",
            str(config_path),
            "--dataset-name",
            "ANTIUAV300_RGB",
            "--crop-root",
            str(tmp_path / "rgb" / "crop511"),
            "--train-json",
            str(train_json),
            "--variant",
            "v2",
            "--pretrained",
            "",
            "--snapshot-dir",
            str(run_root / "snapshots"),
            "--log-dir",
            str(run_root / "logs"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--videos-per-epoch",
            "2",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(trainer),
            "--cfg",
            str(config_path),
            "--device",
            "cpu",
            "--save-every",
            "1",
        ],
        check=True,
    )

    assert (run_root / "snapshots" / "best.pth").exists()
    assert (run_root / "logs" / "history.json").exists()


def test_export_nanotrack_rk3588_dry_run(tmp_path):
    crop_root = tmp_path / "rgb" / "crop511"
    crop_root.mkdir(parents=True)
    train_json = tmp_path / "rgb" / "train.json"
    train_json.write_text(json.dumps({}), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    run_root = tmp_path / "run"
    writer = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "write_nanotrack_config.py"
    exporter = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "export_nanotrack_rk3588.py"

    subprocess.run(
        [
            sys.executable,
            str(writer),
            "--output",
            str(config_path),
            "--dataset-name",
            "ANTIUAV300_RGB",
            "--crop-root",
            str(crop_root),
            "--train-json",
            str(train_json),
            "--variant",
            "v2",
            "--pretrained",
            "",
            "--snapshot-dir",
            str(run_root / "snapshots"),
            "--log-dir",
            str(run_root / "logs"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--videos-per-epoch",
            "1",
        ],
        check=True,
    )

    output_dir = tmp_path / "rk3588_onnx"
    subprocess.run(
        [
            sys.executable,
            str(exporter),
            "--cfg",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["template_input_shape_nchw"] == [1, 3, 127, 127]
    assert manifest["search_input_shape_nchw"] == [1, 3, 255, 255]
    assert manifest["head_cls_shape_nchw"][0] == 1
    assert manifest["head_loc_shape_nchw"][1] == 4


def test_parse_nanotrack_device_spec():
    device, ids = parse_device_spec("cuda:0,1,2,3")
    assert str(device) == "cuda:0"
    assert ids == [0, 1, 2, 3]

    device, ids = parse_device_spec("cpu")
    assert str(device) == "cpu"
    assert ids == []
