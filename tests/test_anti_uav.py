# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np

from ultralytics import solutions


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

    system = solutions.AntiUAVSystem(detector, tracker=tracker, detect_interval=5, pending_frames=1, manual_confirmation=True)
    state = system.step(frame)
    event = system.confirm_current_target(True, note="unit_test_confirm")
    events = system.drain_alerts()

    assert state.confirmation_state == "pending"
    assert event is not None
    assert event.event_type == "alert_raised"
    assert events[0].event_type == "alert_raised"


def test_tracker_registry_contains_expected_defaults():
    names = solutions.available_trackers()
    assert "template_match" in names
    assert "opencv" in names
    assert isinstance(solutions.build_tracker("template_match"), solutions.TemplateMatchTracker)


def test_iter_tiles_covers_full_frame():
    tiles = solutions.iter_tiles((100, 120), tile_size=(64, 64), overlap=0.25)
    assert tiles
    assert tiles[0] == (0.0, 0.0, 64.0, 64.0)
    assert any(tile[2] == 120.0 for tile in tiles)
    assert any(tile[3] == 100.0 for tile in tiles)
