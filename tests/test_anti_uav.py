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
        self.corrected = []
        self.reinitialized = []
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

    def correct_bbox(self, frame, bbox):
        del frame
        self.corrected.append(tuple(bbox))
        self.initialized.append(tuple(bbox))

    def reinit_from_detection(self, frame, bbox):
        self.reinitialized.append(tuple(bbox))
        self.reset()
        self.init(frame, bbox)


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


def create_mini_anti_uav_sequence(
    root: Path,
    sequence_name: str,
    modality: str,
    boxes: list[list[float]],
    *,
    video_stem: str | None = None,
    label_stem: str | None = None,
) -> Path:
    sequence_root = root / sequence_name
    sequence_root.mkdir(parents=True, exist_ok=True)

    video_stem = video_stem or modality
    label_stem = label_stem or f"{modality}_label"

    video_path = sequence_root / f"{video_stem}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    for frame_index in range(len(boxes)):
        frame = np.full((48, 64, 3), frame_index * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    annotation_path = sequence_root / f"{label_stem}.json"
    annotation_path.write_text(json.dumps({"gt_rect": boxes}), encoding="utf-8")
    return sequence_root


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
    assert tracker.reinitialized[-1] == (44.0, 44.0, 84.0, 84.0)


def test_redetected_uses_soft_correction_instead_of_hard_reinit():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(40, 40, 80, 80), confidence=0.9, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(43, 42, 83, 82), confidence=0.91, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker([(True, (41, 40, 81, 80), 0.92), (True, (42, 41, 82, 81), 0.91)])

    system = solutions.AntiUAVSystem(detector, tracker=tracker, detect_interval=1, manual_confirmation=False)

    first = system.step(frame)
    second = system.step(frame)
    third = system.step(frame)

    assert first.status == "detected"
    assert second.status == "tracking"
    assert third.status == "redetected"
    assert tracker.corrected[-1] == (43.0, 42.0, 83.0, 82.0)
    assert tracker.reinitialized == [(40.0, 40.0, 80.0, 80.0)]


def test_reacquired_uses_hard_reinit_after_loss():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(60, 60, 100, 100), confidence=0.9, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(64, 62, 104, 102), confidence=0.92, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker([(False, None, 0.05)])

    system = solutions.AntiUAVSystem(detector, tracker=tracker, detect_interval=1, manual_confirmation=False)

    first = system.step(frame)
    second = system.step(frame)

    assert first.status == "detected"
    assert second.status == "reacquired"
    assert tracker.reinitialized[-1] == (64.0, 62.0, 104.0, 102.0)
    assert tracker.corrected == []


def test_assist_window_escalates_to_hard_reinit_after_consecutive_disagreement():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(40, 40, 80, 80), confidence=0.9, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(45, 40, 85, 80), confidence=0.91, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(170, 40, 210, 80), confidence=0.92, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(175, 40, 215, 80), confidence=0.93, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker(
        [
            (True, (42, 40, 82, 80), 0.92),
            (True, (46, 40, 86, 80), 0.90),
            (True, (47, 40, 87, 80), 0.88),
            (True, (48, 40, 88, 80), 0.87),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        manual_confirmation=False,
        assist_hard_reinit_streak=2,
    )

    first = system.step(frame)
    second = system.step(frame)
    third = system.step(frame)
    fourth = system.step(frame)
    fifth = system.step(frame)

    assert first.status == "detected"
    assert second.status == "tracking"
    assert third.status == "redetected"
    assert fourth.status == "redetected"
    assert fifth.status == "redetected"
    assert tracker.corrected == [
        (45.0, 40.0, 85.0, 80.0),
        (170.0, 40.0, 210.0, 80.0),
    ]
    assert tracker.reinitialized == [
        (40.0, 40.0, 80.0, 80.0),
        (175.0, 40.0, 215.0, 80.0),
    ]
    assert system.assist_disagreement_streak == 0


def test_assist_window_resets_disagreement_streak_after_consistent_refresh():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(40, 40, 80, 80), confidence=0.9, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(45, 40, 85, 80), confidence=0.91, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(170, 40, 210, 80), confidence=0.92, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(48, 40, 88, 80), confidence=0.90, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(176, 40, 216, 80), confidence=0.93, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker(
        [
            (True, (42, 40, 82, 80), 0.92),
            (True, (46, 40, 86, 80), 0.90),
            (True, (47, 40, 87, 80), 0.89),
            (True, (49, 40, 89, 80), 0.88),
            (True, (50, 40, 90, 80), 0.87),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        manual_confirmation=False,
        assist_hard_reinit_streak=2,
    )

    states = [system.step(frame) for _ in range(6)]

    assert [state.status for state in states] == [
        "detected",
        "tracking",
        "redetected",
        "redetected",
        "redetected",
        "redetected",
    ]
    assert tracker.reinitialized == [(40.0, 40.0, 80.0, 80.0)]
    assert tracker.corrected[-1] == (176.0, 40.0, 216.0, 80.0)
    assert system.assist_disagreement_streak == 1


def test_detector_contradiction_forces_hard_reacquire_when_tracker_false_positive_persists():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(2, 40, 42, 80), confidence=0.94, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(210, 44, 252, 86), confidence=0.82, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(219, 46, 261, 88), confidence=0.80, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker(
        [
            (True, (1, 40, 41, 80), 0.91),
            (True, (0, 41, 40, 81), 0.90),
            (True, (0, 42, 40, 82), 0.89),
            (True, (0, 43, 40, 83), 0.88),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        roi_redetect=False,
        manual_confirmation=False,
        detector_contradiction_consensus_frames=2,
        detector_contradiction_confidence=0.8,
        detector_contradiction_continue_confidence=0.75,
    )

    first = system.step(frame)
    second = system.step(frame)
    third = system.step(frame)
    fourth = system.step(frame)
    fifth = system.step(frame)

    assert first.status == "detected"
    assert second.status == "tracking"
    assert third.status == "tracking"
    assert fourth.status == "tracking"
    assert fifth.status == "reacquired"
    assert fifth.bbox == (219.0, 46.0, 261.0, 88.0)
    assert tracker.reinitialized[-1] == (219.0, 46.0, 261.0, 88.0)
    assert system.requires_detector_refresh is False


def test_detector_contradiction_allows_high_score_non_edge_tracker_to_be_overridden():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(96, 52, 136, 92), confidence=0.94, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(216, 56, 256, 96), confidence=0.93, class_id=0, class_name="drone")],
            [solutions.Detection(bbox=(218, 57, 258, 97), confidence=0.92, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker(
        [
            (True, (98, 52, 138, 92), 0.96),
            (True, (100, 52, 140, 92), 0.97),
            (True, (102, 52, 142, 92), 0.97),
            (True, (104, 52, 144, 92), 0.97),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        roi_redetect=False,
        manual_confirmation=False,
        detector_contradiction_consensus_frames=2,
        detector_contradiction_confidence=0.8,
        detector_contradiction_continue_confidence=0.75,
        detector_contradiction_high_score_confidence=0.9,
        detector_contradiction_high_score_continue_confidence=0.9,
        detector_contradiction_min_center_ratio=2.0,
    )

    first = system.step(frame)
    second = system.step(frame)
    third = system.step(frame)
    fourth = system.step(frame)
    fifth = system.step(frame)

    assert first.status == "detected"
    assert second.status == "tracking"
    assert third.status == "tracking"
    assert fourth.status == "tracking"
    assert fifth.status == "reacquired"
    assert fifth.bbox == (218.0, 57.0, 258.0, 97.0)
    assert tracker.reinitialized[-1] == (218.0, 57.0, 258.0, 97.0)


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


def test_detector_miss_does_not_mark_fake_lost_when_tracker_is_healthy():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(20, 20, 40, 40), confidence=0.95, class_id=0, class_name="drone")],
            [],
        ]
    )
    tracker = FakeTracker(
        [
            (True, (21, 20, 41, 40), 0.94),
            (True, (22, 20, 42, 40), 0.92),
        ]
    )

    system = solutions.AntiUAVSystem(detector, tracker=tracker, detect_interval=1, manual_confirmation=False)

    first = system.step(frame)
    second = system.step(frame)
    third = system.step(frame)

    assert first.status == "detected"
    assert second.status == "tracking"
    assert third.status == "tracking"
    assert third.lost_frames == 0
    assert third.bbox == (22.0, 20.0, 42.0, 40.0)


def test_stale_edge_locked_track_drops_target_and_forces_full_frame_search():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(0, 20, 20, 40), confidence=0.95, class_id=0, class_name="drone")],
            [],
            [],
        ]
    )
    tracker = FakeTracker(
        [
            (False, None, 0.0),
            (False, None, 0.0),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        manual_confirmation=True,
        pending_frames=1,
        min_confirm_detections=1,
        stale_search_lost_frames=2,
        stale_search_low_score_streak=2,
        stale_search_edge_margin_px=4,
    )

    first = system.step(frame)
    system.confirm_current_target(True, note="unit_test_confirm")
    second = system.step(frame)
    third = system.step(frame)

    assert first.status == "detected"
    assert second.status == "lost"
    assert second.confirmation_state == "pending"
    assert third.status == "searching"
    assert third.bbox is None
    assert detector.calls[-1]["roi"] is None


def test_refresh_override_allows_far_full_frame_consensus_hard_reacquire():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    far_detection = solutions.Detection(
        bbox=(110, 70, 138, 98),
        confidence=0.92,
        class_id=0,
        class_name="drone",
        source="full_frame",
    )
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(10, 20, 30, 40), confidence=0.95, class_id=0, class_name="drone")],
            [
                solutions.Detection(bbox=(12, 20, 32, 40), confidence=0.80, class_id=0, class_name="drone", source="roi"),
                far_detection,
            ],
            [
                solutions.Detection(bbox=(13, 20, 33, 40), confidence=0.82, class_id=0, class_name="drone", source="roi"),
                solutions.Detection(
                    bbox=(111, 71, 139, 99),
                    confidence=0.93,
                    class_id=0,
                    class_name="drone",
                    source="full_frame",
                ),
            ],
        ]
    )
    tracker = FakeTracker(
        [
            (False, None, 0.0),
            (False, None, 0.0),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        manual_confirmation=True,
        pending_frames=1,
        min_confirm_detections=1,
        refresh_override_confidence=0.75,
        refresh_override_consensus_frames=2,
        refresh_override_stability_iou=0.25,
        refresh_override_min_center_ratio=2.5,
        full_frame_fallback=False,
    )

    first = system.step(frame)
    system.confirm_current_target(True, note="unit_test_confirm")
    second = system.step(frame)
    third = system.step(frame)

    assert first.status == "detected"
    assert second.status == "lost"
    assert second.bbox == (10.0, 20.0, 30.0, 40.0)
    assert third.status == "reacquired"
    assert third.bbox == (111.0, 71.0, 139.0, 99.0)
    assert tracker.reinitialized[-1] == (111.0, 71.0, 139.0, 99.0)
    assert system.refresh_override_streak == 0


def test_refresh_override_accepts_motion_consistent_far_candidate_with_lower_followup_confidence():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(10, 20, 30, 40), confidence=0.95, class_id=0, class_name="drone")],
            [
                solutions.Detection(bbox=(110, 70, 138, 98), confidence=0.90, class_id=0, class_name="drone", source="full_frame"),
            ],
            [
                solutions.Detection(bbox=(92, 68, 120, 96), confidence=0.66, class_id=0, class_name="drone", source="full_frame"),
            ],
        ]
    )
    tracker = FakeTracker(
        [
            (False, None, 0.0),
            (False, None, 0.0),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        manual_confirmation=True,
        pending_frames=1,
        min_confirm_detections=1,
        refresh_override_confidence=0.75,
        refresh_override_continue_confidence=0.60,
        refresh_override_consensus_frames=2,
        refresh_override_stability_iou=0.25,
        refresh_override_min_center_ratio=2.5,
        refresh_override_motion_center_ratio=5.0,
        full_frame_fallback=False,
    )

    first = system.step(frame)
    system.confirm_current_target(True, note="unit_test_confirm")
    second = system.step(frame)
    third = system.step(frame)

    assert first.status == "detected"
    assert second.status == "lost"
    assert third.status == "reacquired"
    assert third.bbox == (92.0, 68.0, 120.0, 96.0)
    assert tracker.reinitialized[-1] == (92.0, 68.0, 120.0, 96.0)


def test_confirmed_target_requires_detector_refresh_after_lost():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detector = DummyDetector(
        [
            [solutions.Detection(bbox=(20, 20, 40, 40), confidence=0.95, class_id=0, class_name="drone")],
            [],
            [solutions.Detection(bbox=(23, 20, 43, 40), confidence=0.94, class_id=0, class_name="drone")],
        ]
    )
    tracker = FakeTracker(
        [
            (False, None, 0.0),
            (True, (21, 20, 41, 40), 0.93),
            (True, (22, 20, 42, 40), 0.91),
        ]
    )

    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        detect_interval=1,
        pending_frames=1,
        min_confirm_detections=1,
        manual_confirmation=True,
        full_frame_fallback=False,
    )

    first = system.step(frame)
    system.confirm_current_target(True, note="unit_test_confirm")
    second = system.step(frame)
    third = system.step(frame)
    fourth = system.step(frame)

    assert first.status == "detected"
    assert second.status == "lost"
    assert second.confirmation_state == "pending"
    assert third.status == "redetected"
    assert third.confirmation_state == "confirmed"
    assert fourth.status == "tracking"
    assert fourth.confirmation_state == "confirmed"


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
    create_mini_anti_uav_sequence(source_root, "seq001", "rgb", [[10, 12, 8, 6], [11, 12, 8, 6], [], [12, 13, 8, 6], []])

    output_root = tmp_path / "nanotrack_export"
    converter = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "convert_anti_uav300_nanotrack.py"
    hardneg_root = tmp_path / "replay_errors" / "train_seq001"
    hardneg_root.mkdir(parents=True)
    (hardneg_root / "errors.jsonl").write_text(
        json.dumps({"type": "false_positive", "frame_index": 3, "bbox": [4, 5, 14, 13]}) + "\n",
        encoding="utf-8",
    )
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
            "--background-frame-step",
            "1",
            "--distractor-frame-step",
            "1",
            "--transition-window",
            "2",
            "--hard-negative-errors",
            str(tmp_path / "replay_errors"),
        ],
        check=True,
    )

    train_json = output_root / "rgb" / "train.json"
    val_json = output_root / "rgb" / "val.json"
    crop_path = output_root / "rgb" / "crop511" / "train_seq001" / "000000.00.x.jpg"
    split_manifest = output_root / "rgb" / "split_manifest.json"

    assert train_json.exists() or val_json.exists()
    assert crop_path.exists()
    assert split_manifest.exists()
    metadata = json.loads((train_json if train_json.stat().st_size else val_json).read_text(encoding="utf-8"))
    assert "train_seq001" in metadata
    assert "00" in metadata["train_seq001"]
    assert "000000" in metadata["train_seq001"]["00"]
    assert "__neg__" in metadata["train_seq001"]
    assert "__bg__" in metadata["train_seq001"]
    assert "__bg_transition__" in metadata["train_seq001"]
    assert "__hardneg__" in metadata["train_seq001"]


def test_convert_anti_uav300_nanotrack_supports_visible_infrared_train_layout(tmp_path):
    source_root = tmp_path / "Anti-UAV300" / "train"
    create_mini_anti_uav_sequence(
        source_root,
        "seq001",
        "rgb",
        [[10, 12, 8, 6], [11, 12, 8, 6], []],
        video_stem="visible",
        label_stem="visible",
    )
    create_mini_anti_uav_sequence(
        source_root,
        "seq001",
        "ir",
        [[13, 14, 7, 5], [], [14, 15, 7, 5]],
        video_stem="infrared",
        label_stem="infrared",
    )

    output_root = tmp_path / "nanotrack_export"
    converter = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "convert_anti_uav300_nanotrack.py"
    subprocess.run(
        [
            sys.executable,
            str(converter),
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--modalities",
            "rgb",
            "ir",
        ],
        check=True,
    )

    rgb_train = json.loads((output_root / "rgb" / "train.json").read_text(encoding="utf-8"))
    ir_train = json.loads((output_root / "ir" / "train.json").read_text(encoding="utf-8"))
    rgb_manifest = json.loads((output_root / "rgb" / "split_manifest.json").read_text(encoding="utf-8"))
    ir_manifest = json.loads((output_root / "ir" / "split_manifest.json").read_text(encoding="utf-8"))

    assert "seq001" in rgb_train
    assert "seq001" in ir_train
    assert rgb_manifest["train"][0]["video"].endswith("visible.mp4")
    assert rgb_manifest["train"][0]["label"].endswith("visible.json")
    assert ir_manifest["train"][0]["video"].endswith("infrared.mp4")
    assert ir_manifest["train"][0]["label"].endswith("infrared.json")


def test_nanotrack_dataset_prefers_transition_templates_and_absent_search_tracks(tmp_path):
    nanotrack_root = Path(__file__).resolve().parents[1] / "third_party" / "nanotrack_vendor"
    if str(nanotrack_root) not in sys.path:
        sys.path.insert(0, str(nanotrack_root))

    from nanotrack.core.config import cfg
    from nanotrack.datasets.dataset import SubDataset

    crop_root = tmp_path / "rgb" / "crop511" / "train_seq001"
    crop_root.mkdir(parents=True)
    frame = np.full((255, 255, 3), 127, dtype=np.uint8)
    track_names = ("00", "__neg__", "__bg__", "__bg_transition__", "__hardneg__")
    frame_ids = [f"{frame_index:06d}" for frame_index in range(16)]
    for frame_id in frame_ids:
        for track_name in track_names:
            cv2.imwrite(str(crop_root / f"{frame_id}.{track_name}.x.jpg"), frame)

    train_json = tmp_path / "rgb" / "train.json"
    train_json.write_text(
        json.dumps(
            {
                "train_seq001": {
                    "00": {frame_id: [100, 100, 40, 32] for frame_id in frame_ids},
                    "__neg__": {"000004": [0, 0, 40, 32]},
                    "__bg__": {"000006": [0, 0, 40, 32]},
                    "__bg_transition__": {"000007": [0, 0, 40, 32]},
                    "__hardneg__": {"000008": [0, 0, 40, 32]},
                }
            }
        ),
        encoding="utf-8",
    )

    original_values = {
        "NEG_SAME_SEQ_PROB": getattr(cfg.DATASET, "NEG_SAME_SEQ_PROB", 0.75),
        "NEG_HARD_PROB": getattr(cfg.DATASET, "NEG_HARD_PROB", 0.25),
        "NEG_TRANSITION_PROB": getattr(cfg.DATASET, "NEG_TRANSITION_PROB", 0.25),
        "NEG_BACKGROUND_PROB": getattr(cfg.DATASET, "NEG_BACKGROUND_PROB", 0.20),
        "TRANSITION_TEMPLATE_PROB": getattr(cfg.DATASET, "TRANSITION_TEMPLATE_PROB", 0.5),
        "TRANSITION_FRAME_WINDOW": getattr(cfg.DATASET, "TRANSITION_FRAME_WINDOW", 8),
    }
    try:
        cfg.DATASET.NEG_SAME_SEQ_PROB = 1.0
        cfg.DATASET.NEG_HARD_PROB = 1.0
        cfg.DATASET.NEG_TRANSITION_PROB = 0.0
        cfg.DATASET.NEG_BACKGROUND_PROB = 0.0
        cfg.DATASET.TRANSITION_TEMPLATE_PROB = 1.0
        cfg.DATASET.TRANSITION_FRAME_WINDOW = 1

        dataset = SubDataset("ANTI", tmp_path / "rgb" / "crop511", train_json, frame_range=30, num_use=-1, start_idx=0)
        _, template = dataset.get_random_target(0, return_video=True, prefer_transition=True)
        hard_negative = dataset.get_negative_search(0, preferred_video="train_seq001")

        transition_template_frames = {"000000", "000001", "000004", "000005", "000010", "000011", "000014", "000015"}
        assert template[0].name.split(".")[0] in transition_template_frames
        assert hard_negative[0].name.endswith("__hardneg__.x.jpg")

        cfg.DATASET.NEG_HARD_PROB = 0.0
        cfg.DATASET.NEG_TRANSITION_PROB = 1.0
        transition_negative = dataset.get_negative_search(0, preferred_video="train_seq001")
        assert transition_negative[0].name.endswith("__bg_transition__.x.jpg")
    finally:
        for key, value in original_values.items():
            setattr(cfg.DATASET, key, value)


def test_nanotrack_dataset_can_bias_positive_pairs_toward_fast_motion(tmp_path):
    from nanotrack.core.config import cfg
    from nanotrack.datasets.dataset import SubDataset

    crop_root = tmp_path / "rgb" / "crop511" / "train_seq001"
    crop_root.mkdir(parents=True)
    frame = np.full((255, 255, 3), 127, dtype=np.uint8)
    for frame_index in range(20):
        cv2.imwrite(str(crop_root / f"{frame_index:06d}.00.x.jpg"), frame)

    train_json = tmp_path / "rgb" / "train.json"
    train_json.write_text(
        json.dumps(
            {
                "train_seq001": {
                    "00": {f"{frame_index:06d}": [100, 100, 40, 32] for frame_index in range(20)}
                }
            }
        ),
        encoding="utf-8",
    )

    original_values = {
        "FAST_MOTION_PROB": getattr(cfg.DATASET, "FAST_MOTION_PROB", 0.0),
        "FAST_MOTION_MIN_GAP": getattr(cfg.DATASET, "FAST_MOTION_MIN_GAP", 12),
        "TRANSITION_TEMPLATE_PROB": getattr(cfg.DATASET, "TRANSITION_TEMPLATE_PROB", 0.5),
    }
    try:
        cfg.DATASET.FAST_MOTION_PROB = 1.0
        cfg.DATASET.FAST_MOTION_MIN_GAP = 8
        cfg.DATASET.TRANSITION_TEMPLATE_PROB = 0.0

        dataset = SubDataset("ANTI", tmp_path / "rgb" / "crop511", train_json, frame_range=19, num_use=-1, start_idx=0)
        template, search = dataset.get_positive_pair(0)

        template_frame = int(template[0].name.split(".")[0])
        search_frame = int(search[0].name.split(".")[0])
        assert abs(search_frame - template_frame) >= 8
    finally:
        for key, value in original_values.items():
            setattr(cfg.DATASET, key, value)


def test_nanotrack_val_eval_and_checkpoint_sweep_dry_run(tmp_path):
    source_root = tmp_path / "Anti-UAV300" / "train"
    create_mini_anti_uav_sequence(source_root, "seq001", "rgb", [[10, 12, 8, 6], [11, 12, 8, 6], [], [12, 13, 8, 6]])
    create_mini_anti_uav_sequence(source_root, "seq002", "rgb", [[14, 10, 7, 6], [], [15, 11, 7, 6], []])

    output_root = tmp_path / "nanotrack_export"
    converter = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "convert_anti_uav300_nanotrack.py"
    val_eval = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "nanotrack_val_eval.py"
    sweep = Path(__file__).resolve().parents[1] / "scripts" / "anti_uav" / "nanotrack_checkpoint_sweep.py"
    fake_cfg = tmp_path / "config.yaml"
    fake_cfg.write_text("META_ARC: \"nanotrack\"\n", encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "epoch_001.pth").write_bytes(b"fake")
    (snapshot_dir / "epoch_020.pth").write_bytes(b"fake")
    (snapshot_dir / "best.pth").write_bytes(b"fake")

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
        ],
        check=True,
    )

    val_manifest = tmp_path / "val_eval_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(val_eval),
            "--source-root",
            str(tmp_path / "Anti-UAV300"),
            "--converted-root",
            str(output_root),
            "--modality",
            "rgb",
            "--split",
            "train",
            "--dry-run",
            "--output-json",
            str(val_manifest),
        ],
        check=True,
    )
    manifest = json.loads(val_manifest.read_text(encoding="utf-8"))
    assert manifest["sequence_count"] >= 1
    assert manifest["sequence_names"]

    sweep_manifest = tmp_path / "checkpoint_sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(sweep),
            "--snapshot-dir",
            str(snapshot_dir),
            "--config",
            str(fake_cfg),
            "--source-root",
            str(tmp_path / "Anti-UAV300"),
            "--converted-root",
            str(output_root),
            "--modality",
            "rgb",
            "--split",
            "train",
            "--min-checkpoint-epoch",
            "10",
            "--include-best",
            "--dry-run",
            "--output-json",
            str(sweep_manifest),
        ],
        check=True,
    )
    sweep_payload = json.loads(sweep_manifest.read_text(encoding="utf-8"))
    assert not any(path.endswith("epoch_001.pth") for path in sweep_payload["checkpoints"])
    assert any(path.endswith("epoch_020.pth") for path in sweep_payload["checkpoints"])
    assert not any(path.endswith("best.pth") for path in sweep_payload["checkpoints"])


def test_nanotrack_val_eval_composite_penalizes_absent_false_positives_more_heavily():
    from scripts.anti_uav.nanotrack_val_eval import compute_composite

    low_absent_fp = compute_composite(0.80, 0.70, 0.05)
    high_absent_fp = compute_composite(0.80, 0.70, 0.35)

    assert low_absent_fp > high_absent_fp
    assert round(low_absent_fp - high_absent_fp, 6) == round((0.35 - 0.05) * 0.40, 6)


def test_nanotrack_checkpoint_sweep_prefers_hard_subset_within_overall_tolerance():
    from scripts.anti_uav.nanotrack_checkpoint_sweep import select_best_checkpoint

    results = [
        {
            "checkpoint": "epoch_005.pth",
            "aggregate": {"composite": 0.60, "precision": 0.60},
            "hard_aggregate": {"sequence_count": 1, "composite": 0.30, "precision": 0.30},
        },
        {
            "checkpoint": "epoch_025.pth",
            "aggregate": {"composite": 0.585, "precision": 0.585},
            "hard_aggregate": {"sequence_count": 1, "composite": 0.52, "precision": 0.52},
        },
        {
            "checkpoint": "epoch_040.pth",
            "aggregate": {"composite": 0.54, "precision": 0.54},
            "hard_aggregate": {"sequence_count": 1, "composite": 0.90, "precision": 0.90},
        },
    ]

    best, selection = select_best_checkpoint(
        results,
        metric="composite",
        hard_metric="composite",
        hard_overall_tolerance=0.02,
        hard_min_sequences=1,
    )

    assert best["checkpoint"] == "epoch_025.pth"
    assert selection["strategy"] == "overall_then_hard_subset"
    assert selection["shortlist_count"] == 2


def test_nanotrack_checkpoint_sweep_can_filter_early_epochs():
    from scripts.anti_uav.nanotrack_checkpoint_sweep import checkpoint_epoch

    assert checkpoint_epoch(Path("epoch_005.pth")) == 5
    assert checkpoint_epoch(Path("epoch_025.pth")) == 25
    assert checkpoint_epoch(Path("best.pth")) is None


def test_nanotrack_checkpoint_sweep_can_pick_motion_based_hard_subset(tmp_path):
    from scripts.anti_uav.nanotrack_checkpoint_sweep import resolve_hard_sequence_names

    root = tmp_path / "Anti-UAV300" / "train"
    slow = create_mini_anti_uav_sequence(
        root,
        "slow_seq",
        "rgb",
        [[10, 10, 8, 6], [11, 10, 8, 6], [12, 10, 8, 6], [13, 10, 8, 6], [14, 10, 8, 6]],
    )
    fast = create_mini_anti_uav_sequence(
        root,
        "fast_seq",
        "rgb",
        [[10, 10, 8, 6], [30, 10, 8, 6], [52, 10, 8, 6], [76, 10, 8, 6], [100, 10, 8, 6]],
    )
    medium = create_mini_anti_uav_sequence(
        root,
        "medium_seq",
        "rgb",
        [[10, 10, 8, 6], [16, 10, 8, 6], [23, 10, 8, 6], [31, 10, 8, 6], [40, 10, 8, 6]],
    )

    entries = [
        {"name": "slow_seq", "label": str(slow / "rgb_label.json")},
        {"name": "fast_seq", "label": str(fast / "rgb_label.json")},
        {"name": "medium_seq", "label": str(medium / "rgb_label.json")},
    ]
    args = type(
        "Args",
        (),
        {
            "hard_sequence_mode": "motion",
            "hard_sequence_patterns": [],
            "hard_motion_top_k": 1,
            "hard_motion_quantile": 0.9,
            "hard_motion_min_present": 2,
        },
    )()

    hard_names, hard_details = resolve_hard_sequence_names(entries, args)

    assert hard_names == ["fast_seq"]
    assert hard_details["fast_seq"]["motion_score"] > hard_details["medium_seq"]["motion_score"] > hard_details["slow_seq"]["motion_score"]
    assert hard_details["fast_seq"]["selected_by"] == "motion"


def test_nanotrack_checkpoint_sweep_can_union_pattern_and_motion_hard_subsets(tmp_path):
    from scripts.anti_uav.nanotrack_checkpoint_sweep import resolve_hard_sequence_names

    root = tmp_path / "Anti-UAV300" / "train"
    steady = create_mini_anti_uav_sequence(
        root,
        "steady_seq",
        "rgb",
        [[10, 10, 8, 6], [11, 10, 8, 6], [12, 10, 8, 6], [13, 10, 8, 6]],
    )
    fast = create_mini_anti_uav_sequence(
        root,
        "fast_seq",
        "rgb",
        [[10, 10, 8, 6], [40, 10, 8, 6], [70, 10, 8, 6], [100, 10, 8, 6]],
    )

    entries = [
        {"name": "steady_seq", "label": str(steady / "rgb_label.json")},
        {"name": "fast_seq", "label": str(fast / "rgb_label.json")},
    ]
    args = type(
        "Args",
        (),
        {
            "hard_sequence_mode": "union",
            "hard_sequence_patterns": ["steady"],
            "hard_motion_top_k": 1,
            "hard_motion_quantile": 0.9,
            "hard_motion_min_present": 2,
        },
    )()

    hard_names, hard_details = resolve_hard_sequence_names(entries, args)

    assert hard_names == ["fast_seq", "steady_seq"]
    assert hard_details["steady_seq"]["selected_by"] == "patterns"
    assert hard_details["fast_seq"]["selected_by"] == "motion"


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
            "--neg-transition-prob",
            "0.25",
            "--neg-hard-prob",
            "0.25",
            "--transition-template-prob",
            "0.5",
            "--transition-frame-window",
            "8",
        ],
        check=True,
    )
    config_text = config_path.read_text(encoding="utf-8")
    assert "NEG_TRANSITION_PROB: 0.25" in config_text
    assert "NEG_HARD_PROB: 0.25" in config_text
    assert "TRANSITION_TEMPLATE_PROB: 0.5" in config_text
    assert "TRANSITION_FRAME_WINDOW: 8" in config_text

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
            "--neg-transition-prob",
            "0.25",
            "--neg-hard-prob",
            "0.25",
            "--transition-template-prob",
            "0.5",
            "--transition-frame-window",
            "8",
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
