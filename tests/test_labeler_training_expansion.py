import cv2
import numpy as np
import pytest

from scripts.anti_uav.prepare_labeler_training_expansion import extract, excluded_calibration_source, frame_plan, sha256


def row(index, status="confirmed", box=(0, 0, 100, 100)):
    return dict(frame_index=index, status=status,
                **dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), box)))


def test_pending_absent_and_unknown_are_not_negatives():
    plan, excluded = frame_plan([row(0), row(1, "absent"), row(2, "uncertain")], 4, 100, 100, False)
    assert plan == {0: [[.5, .5, 1., 1.]]}
    assert excluded == dict(unlabelled=1, unreviewed_absent=1, uncertain=1)


def test_reviewed_negative_only_video_is_allowed():
    plan, excluded = frame_plan([row(0, "absent")], 1, 100, 100, True)
    assert plan == {0: []}
    assert excluded == dict(unlabelled=0)


def test_uncertain_object_excludes_whole_frame():
    plan, excluded = frame_plan([row(0), row(0, "uncertain")], 1, 100, 100, True)
    assert not plan
    assert excluded["uncertain"] == 1


def test_no_tiny_or_large_area_filter():
    plan, _ = frame_plan([row(0, box=(1, 1, 5, 5)), row(1, box=(0, 0, 95, 95))], 2, 100, 100, False)
    assert set(plan) == {0, 1}
    assert plan[0][0][2:] == [.04, .04]
    assert plan[1][0][2:] == [.95, .95]


def test_partially_out_of_frame_boxes_clipped_but_invalid_not_negatives():
    plan, excluded = frame_plan([row(0, box=(90, 0, 120, 100)), row(1, box=(150, 0, 160, 100))], 2, 100, 100, True)
    assert plan == {0: [[.95, .5, .1, 1.]]}
    assert excluded["invalid_bbox"] == 1


def test_calibration_does_not_confuse_holdout_named_root_with_holdout_images():
    assert not excluded_calibration_source("/data/strict_holdout_Video00004/images/Video00003/001.jpg", "Video00004")
    assert excluded_calibration_source("/data/images/Video00004/001.jpg", "Video00004")
    assert excluded_calibration_source("/data/images/video00009/001.jpg", "Video00009")
    assert excluded_calibration_source("/data/zoom_val/001.jpg", "zoom_val")


def test_legacy_submitted_positive_preserves_the_exact_displayed_frame(tmp_path):
    source = tmp_path / "video.ts"
    source.write_bytes(b"not used as a decoder for legacy frame indexes")
    asset = tmp_path / "display.jpg"
    assert cv2.imwrite(str(asset), np.full((100, 100, 3), 100, np.uint8))
    item = dict(video=str(source), original_path=source.name, source_sha256=sha256(source), task_id="pending",
        status="submitted", revision=1, frame_count=2, frame_height=100, frame_width=100,
        annotations=[row(0, box=(10, 20, 30, 40)), row(1, "absent")])
    binding = (dict(decode_mode="legacy_display"), [(asset, sha256(asset))]*2)
    record, positives, negatives, cached = extract(item, tmp_path/"out", binding)
    assert len(positives) == len(cached) == 1 and negatives == []
    assert record["frame_alignment"]["display_jpegs_verified"] == 1
    assert record["frame_alignment"]["gt_matches_approval"] is False
    assert open(positives[0], "rb").read() == asset.read_bytes()
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Displayed JPEG changed"):
        extract(item, tmp_path/"changed", binding)
