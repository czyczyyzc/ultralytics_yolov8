import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.anti_uav.efficient_gmc import EfficientGMC
from scripts.anti_uav.motion_native_runtime import DEFAULTS, NativeMotion
from scripts.anti_uav.render_cached_tracker_result_video import panel, validate_observations

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = np.eye(2, 3)


@pytest.fixture(scope="session")
def library(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if not compiler:
        pytest.skip("A C++17 compiler is required")
    output = tmp_path_factory.mktemp("motion_native") / "libmotion_tracker.so"
    src = ROOT / "scripts/anti_uav/dist_native"
    subprocess.run([compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-shared", "-fPIC",
        str(src / "motion_tracker.cpp"), str(src / "third_party/lap/lapjv.cpp"), "-o", str(output)], check=True)
    return output


def boxes(x, y=100, size=4, score=.9):
    return [[x, y, x+size, y+size, score]]


def fast_tracker(library, fps=30.):
    return NativeMotion(library, fps, dict(DEFAULTS, nominal_fps=fps,
        confirmation_hits=2, confirmation_window=3))


@pytest.mark.parametrize("fps,step", [(30., 12.), (100., 8.)])
def test_small_nonoverlap_motion_and_maneuver_keep_id(library, fps, step):
    tracker = fast_tracker(library, fps)
    try:
        for i in range(30):
            jump = 40 if fps==30 else 15
            x = 100+i*step+(jump if i>=12 else 0)
            detections = boxes(x)
            output = tracker.update(detections, IDENTITY, 1., i/fps)
            if i==0:
                assert not output
            else:
                assert len(output)==1 and output[0]["id"]==1, (i, output)
                assert output[0]["box"]==detections[0][:4]
        assert tracker.stats()["zero_iou_matches"]>0
        assert tracker.stats()["allocated_ids"]==1
    finally:
        tracker.close()


def test_short_gap_has_no_predicted_output_then_recovers_id(library):
    tracker = fast_tracker(library)
    try:
        assert not tracker.update(boxes(100), IDENTITY, 0., 0)
        assert tracker.update(boxes(110), IDENTITY, 0., 1/30)[0]["id"]==1
        assert not tracker.update([], IDENTITY, 0., 2/30)
        assert tracker.update(boxes(130), IDENTITY, 0., 3/30)[0]["id"]==1
    finally:
        tracker.close()


def test_single_frame_false_positive_is_never_confirmed(library):
    tracker = NativeMotion(library)
    try:
        assert not tracker.update(boxes(100), IDENTITY, 0., 0)
        for i in range(1, 8):
            assert not tracker.update([], IDENTITY, 0., i/30)
        assert tracker.stats()["confirmations"]==0
    finally:
        tracker.close()


def test_expiry_does_not_force_old_identity(library):
    tracker = fast_tracker(library)
    try:
        tracker.update(boxes(100), IDENTITY, 0., 0)
        assert tracker.update(boxes(100), IDENTITY, 0., .03)[0]["id"]==1
        assert not tracker.update(boxes(100), IDENTITY, 0., 2)
        assert tracker.update(boxes(100), IDENTITY, 0., 2.03)[0]["id"]==2
    finally:
        tracker.close()


def test_camera_warp_compensates_pending_and_confirmed_tracks(library):
    tracker = fast_tracker(library)
    warp = np.array([[1., 0., 8.], [0., 1., -3.]])
    try:
        for i in range(15):
            output = tracker.update(boxes(100+8*i, 100-3*i), warp if i else IDENTITY, 1., i/30)
            if i:
                assert len(output)==1 and output[0]["id"]==1, (i, output)
    finally:
        tracker.close()


def test_ambiguous_observation_is_not_forced_or_reborn(library):
    tracker = fast_tracker(library)
    try:
        pair = boxes(100)+boxes(108)
        tracker.update(pair, IDENTITY, 0., 0)
        assert {t["id"] for t in tracker.update(pair, IDENTITY, 0., .03)}=={1, 2}
        assert not tracker.update(boxes(104), IDENTITY, 0., .06)
        assert tracker.stats()["allocated_ids"]==2
        assert tracker.stats()["ambiguous_columns"]>0
    finally:
        tracker.close()


def test_far_distractor_cannot_take_existing_identity(library):
    tracker = fast_tracker(library)
    try:
        tracker.update(boxes(100), IDENTITY, 0., 0)
        assert tracker.update(boxes(110), IDENTITY, 0., .03)[0]["id"]==1
        assert not tracker.update(boxes(1000), IDENTITY, 0., .06)
        output = tracker.update(boxes(1000), IDENTITY, 0., .09)
        assert len(output)==1 and output[0]["id"]==2
    finally:
        tracker.close()


def test_maneuver_beyond_configured_search_gate_is_not_forced(library):
    tracker = NativeMotion(library, 100.)
    try:
        for i in range(12):
            tracker.update(boxes(100+8*i), IDENTITY, 1., i/100)
        # Innovation 40px exceeds the default 30px search budget at dt=.01s.
        assert not tracker.update(boxes(100+12*8+40), IDENTITY, 1., .12)
    finally:
        tracker.close()


def test_all_absolute_box_sizes_are_supported(library):
    tracker = fast_tracker(library)
    large = [[10, 10, 1810, 1010, .9]]
    try:
        assert not tracker.update(large, IDENTITY, 0., 0)
        output = tracker.update(large, IDENTITY, 0., .03)
        assert len(output)==1 and output[0]["box"]==large[0][:4]
    finally:
        tracker.close()


def test_large_target_localization_jitter_does_not_break_identity(library):
    tracker = fast_tracker(library, 100.)
    try:
        for i in range(40):
            x=600+2*i+(25 if i%2 else -25)
            output=tracker.update([[x, 900, x+320, 1050, .8]], IDENTITY, 1., i/100)
            if i:
                assert len(output)==1 and output[0]["id"]==1
    finally:
        tracker.close()


def test_invalid_timestamps_scores_and_quality_fail(library):
    tracker = NativeMotion(library)
    try:
        tracker.update(boxes(100), IDENTITY, 0., 0)
        with pytest.raises(RuntimeError, match="timestamps"):
            tracker.update(boxes(100), IDENTITY, 0., 0)
        with pytest.raises(RuntimeError, match="quality"):
            tracker.update(boxes(100), IDENTITY, float("nan"), .03)
        with pytest.raises(RuntimeError, match="box/score"):
            tracker.update(boxes(100, score=1.1), IDENTITY, 0., .03)
        tracker.update(boxes(100), IDENTITY, 0., .03)
    finally:
        tracker.close()
    with pytest.raises(RuntimeError, match="configuration"):
        NativeMotion(library, config=dict(DEFAULTS, confirmation_hits=0))


def test_gmc_quality_distinguishes_stationary_scene_and_failed_estimate():
    gmc = EfficientGMC(320, 128, 5, True)
    gray = np.random.default_rng(84).integers(0, 256, (180, 320), dtype=np.uint8)
    gmc.apply(gray)
    assert gmc.last_quality==0
    stationary = gmc.apply(gray)
    np.testing.assert_allclose(stationary, IDENTITY, atol=1e-5)
    assert gmc.last_quality>0 and gmc.last_meta["estimated"]
    blank = np.zeros_like(gray)
    gmc.apply(blank)
    failed = gmc.apply(blank)
    np.testing.assert_array_equal(failed, IDENTITY)
    assert gmc.last_quality==0 and not gmc.last_meta["estimated"]


def test_weak_observations_need_cumulative_confidence_evidence(library):
    tracker = NativeMotion(library)
    try:
        assert not tracker.update(boxes(100, score=.11), IDENTITY, 0., 0)
        assert not tracker.update(boxes(100, score=.04), IDENTITY, 0., .03)
        assert not tracker.update(boxes(100, score=.04), IDENTITY, 0., .06)
        assert tracker.update(boxes(100, score=.4), IDENTITY, 0., .09)[0]["id"]==1
    finally:
        tracker.close()


def test_float32_threshold_boundaries_are_consistent(library):
    tracker = NativeMotion(library, config=dict(DEFAULTS, birth=.03, confirmation_hits=2, confirmation_window=3))
    try:
        assert not tracker.update(boxes(100, score=.03), IDENTITY, 1., 0)
        assert tracker.update(boxes(100, score=.03), IDENTITY, 1., .03)[0]["id"]==1
    finally:
        tracker.close()
    with pytest.raises(RuntimeError, match="closed"):
        tracker.stats()


def test_balanced_default_requires_three_observations_in_four_frames(library):
    tracker = NativeMotion(library)
    try:
        assert not tracker.update(boxes(100), IDENTITY, 0., 0)
        assert not tracker.update([], IDENTITY, 0., .03)
        assert not tracker.update(boxes(100), IDENTITY, 0., .06)
        assert tracker.update(boxes(100), IDENTITY, 0., .09)[0]["id"]==1
    finally:
        tracker.close()
    with pytest.raises(ValueError, match="source FPS"):
        NativeMotion(library, 100., dict(DEFAULTS))


def test_fast_camera_motion_then_gmc_loss_preserves_identity(library):
    tracker = NativeMotion(library, 100.)
    warp = np.array([[1., 0., 110.], [0., 1., 60.]])
    try:
        for i in range(8):
            actual_warp=IDENTITY if i==0 or i in (4,5) else warp
            quality=0. if i==0 or i in (4,5) else .9
            output=tracker.update(boxes(100+110*i, 100+60*i, size=8), actual_warp, quality, i/100)
            if i>=2:
                assert len(output)==1 and output[0]["id"]==1, (i, output)
        assert tracker.stats()["allocated_ids"]==1
    finally:
        tracker.close()


def test_every_detection_survives_without_inventing_an_identity(library):
    tracker = NativeMotion(library)
    try:
        detections = boxes(100)+boxes(900, score=.04)+boxes(1500, score=.005)
        for i in range(6):
            confirmed = tracker.update(detections, IDENTITY, 1., i/30)
            observed = tracker.observations()
            validate_observations(detections, observed, confirmed, i)
            assert len(observed)==3
            assert observed[1]["id"] is None and observed[1]["status"]=="unassigned"
            assert observed[2]["id"] is None and observed[2]["status"]=="below_low"
            assert observed[0]["id"]==(1 if i>=2 else None)
            assert observed[0]["status"]==("confirmed" if i>=2 else "pending")
        assert not tracker.update([], IDENTITY, 1., .2)
        assert tracker.observations()==[]
    finally:
        tracker.close()


def test_ambiguity_preserves_box_but_does_not_fake_id(library):
    tracker = fast_tracker(library)
    try:
        pair = boxes(100)+boxes(108)
        tracker.update(pair, IDENTITY, 0., 0)
        tracker.update(pair, IDENTITY, 0., .03)
        detections = boxes(104)
        confirmed = tracker.update(detections, IDENTITY, 0., .06)
        observed = tracker.observations()
        validate_observations(detections, observed, confirmed, 2)
        assert not confirmed
        assert observed[0]["id"] is None and observed[0]["status"]=="ambiguous"
        assert tracker.stats()["allocated_ids"]==2
    finally:
        tracker.close()


def test_observation_validation_rejects_missing_boxes_and_fake_ids():
    detection = boxes(100)[0]
    valid = dict(id=None, detection_index=0, status="pending", confirmed=False,
        predicted=False, box=detection[:4], score=detection[4])
    validate_observations([detection], [valid], [], 0)
    with pytest.raises(ValueError, match="Missing"):
        validate_observations([detection], [], [], 0)
    for change in (dict(id=1), dict(box=[100, 100, 105, 105]), dict(predicted=True),
                   dict(detection_index=1), dict(status="confirmed", confirmed=True, id=1)):
        with pytest.raises(ValueError):
            validate_observations([detection], [dict(valid, **change)], [], 0)


def test_observation_renderer_keeps_clean_source_and_pending_box():
    frame = np.full((1080, 1920, 3), 120, np.uint8)
    original = frame.copy()
    detected = [dict(id=None, box=[100, 100, 110, 110], score=.03, status="pending")]
    result = panel(frame, detected, 0, 100, 30., "Observed detections; pending ID")
    assert result.shape==(784, 1600, 3)
    np.testing.assert_array_equal(frame, original)


def test_observation_capacity_error_and_closed_handle(library):
    tracker = NativeMotion(library)
    try:
        tracker.update(boxes(100)+boxes(900), IDENTITY, 0., 0)
        output = np.empty((1, 3), np.int32)
        assert tracker.lib.motion_observations(tracker.handle, output.ctypes.data, 1)==-1
        assert b"capacity" in tracker.lib.motion_error()
        assert len(tracker.observations())==2
    finally:
        tracker.close()
    with pytest.raises(RuntimeError, match="closed"):
        tracker.update(boxes(100), IDENTITY, 0., .03)
    with pytest.raises(RuntimeError, match="closed"):
        tracker.observations()


def test_large_target_nested_duplicate_does_not_replace_confirmed_identity(library):
    tracker = NativeMotion(library, 100.)
    full = [[720, 1000, 1028, 1080, .8]]
    growing = [[664, 984, 1024, 1080, .6], [792, 1022, 1011, 1080, .2]]
    next_frame = [[691, 990, 973, 1080, .7]]
    try:
        for i in range(5):
            tracker.update(full, IDENTITY, 1., i/100)
        matched = tracker.update(growing, IDENTITY, 1., .05)
        assert any(t["id"]==1 and t["detection_index"]==0 for t in matched)
        validate_observations(growing, tracker.observations(), matched, 5)
        assert tracker.update(next_frame, IDENTITY, 1., .06)[0]["id"]==1
    finally:
        tracker.close()


def test_all_observation_coverage_is_not_confirmed_tracking_accuracy():
    from scripts.anti_uav.audit_motion_observations import coverage, labelled
    detections = boxes(100)
    rows = [dict(boxes_xyxy_score=detections, displayed_tracks=[], observations=[
        dict(id=None, detection_index=0, status="pending", confirmed=False,
            predicted=False, box=detections[0][:4], score=detections[0][4])])]
    metrics = coverage(rows)
    assert metrics["missing_measured_boxes"]==0
    assert metrics["confirmed_track_boxes"]==0
    measured = labelled(rows, {0: [detections[0][:4]]}, {0})
    assert measured["detector_tp_without_confirmed_track_count"]==1
    assert measured["confirmed_only_metrics"]["recall"]==0


def test_immediate_confirmation_can_assign_low_score_birth_an_id(library):
    config = dict(DEFAULTS, birth=.03, confirmation_hits=1, confirmation_window=1)
    tracker = NativeMotion(library, config=config)
    try:
        detections = boxes(100, score=.03)+boxes(900, score=.04)
        output = tracker.update(detections, IDENTITY, 1., 0.)
        assert {t["id"] for t in output}=={1, 2}
        assert tracker.stats()["confirmations"]==2
        validate_observations(detections, tracker.observations(), output, 0)
        assert tracker.update(boxes(110, score=.03), IDENTITY, 1., 1/30)[0]["id"]==1
    finally:
        tracker.close()


def test_disabled_gmc_models_image_motion_without_camera_variance(library):
    from scripts.anti_uav.track_cached_native_dist_video import motion_config_for_replay
    config = motion_config_for_replay(30., None, "disabled")
    assert config["unknown_gmc_speed_px_s"]==0.
    assert motion_config_for_replay(30., None, "estimate")["unknown_gmc_speed_px_s"]==1500.
    tracker = NativeMotion(library, config=config)
    try:
        for i in range(15):
            output = tracker.update(boxes(100+10*i), IDENTITY, 0., i/30)
            if i>=2:
                assert len(output)==1 and output[0]["id"]==1
    finally:
        tracker.close()


def test_immediate_id_does_not_make_a_single_frame_false_detection_reliable(library):
    config = dict(DEFAULTS, birth=.03, confirmation_hits=1, confirmation_window=1)
    tracker = NativeMotion(library, config=config)
    try:
        assert tracker.update(boxes(100, score=.04), IDENTITY, 1., 0)[0]["id"]==1
        assert tracker.stats()["confirmations"]==1
        assert not tracker.update([], IDENTITY, 1., 1/30)
    finally:
        tracker.close()


def test_top_observation_diagnostic_detects_id_change_across_pending_gap():
    from scripts.anti_uav.compare_motion_gmc_ablation import top_observation_ids
    rows = [dict(boxes_xyxy_score=boxes(100), observations=[dict(detection_index=0, id=identity)])
            for identity in (1, None, 2)]
    result = top_observation_ids(rows, range(3))
    assert result["id_changes_between_identified_observations"]==1
    assert result["adjacent_frame_id_changes"]==0
    assert result["top_detection_without_id"]==1


def test_top_observation_diagnostic_rejects_unsorted_detection_scores():
    from scripts.anti_uav.compare_motion_gmc_ablation import top_observation_ids
    rows = [dict(boxes_xyxy_score=boxes(100,score=.1)+boxes(900,score=.9),
                 observations=[dict(detection_index=0,id=1),dict(detection_index=1,id=2)])]
    with pytest.raises(ValueError,match="score-sorted"):
        top_observation_ids(rows, [0])
