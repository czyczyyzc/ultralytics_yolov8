from copy import deepcopy

import pytest

from scripts.anti_uav.extend_online_gray_replacement import compatible_video, incremental_snapshot


def test_incremental_snapshot_keeps_only_new_audited_videos():
    def row(h):
        return dict(sha256=h, positive_frames=4, negative_frames=3,
                    video_and_annotation_hashes_valid=True, coco_yolo_geometry_valid=True)
    audit = dict(heldout_sha256=["test", "val"], new_rows=[row("cached"), row("new")])
    source = dict(train_video_hashes=["old", "cached"], test_sha256="test", validation_sha256="val")
    result = incremental_snapshot(audit, source)
    assert result["new_videos"] == 1
    assert result["expanded_training_videos"] == 3
    assert result["new_positive_frames"] == 4
    assert len(audit["new_rows"]) == 2
    audit["new_rows"][1]["sha256"] = "test"
    with pytest.raises(ValueError, match="held-out"):
        incremental_snapshot(audit, source)


def test_cached_geometry_and_existing_mapping_must_match():
    before = dict(record=dict(sha256="video"), annotation_sha256="annotation",
                  positive=[dict(frame=1, box=[1, 2, 3, 4], width=100, height=50, image="a", label="b"),
                            dict(frame=2, box=[2, 3, 4, 5], width=100, height=50)])
    after = deepcopy(before)
    compatible_video(before, after)
    after["positive"][0]["image"] = "other"
    with pytest.raises(ValueError, match="mapping"):
        compatible_video(before, after)
    after = deepcopy(before)
    after["positive"][1]["box"][0] = 9
    with pytest.raises(ValueError, match="geometry"):
        compatible_video(before, after)
    after = deepcopy(before)
    after["annotation_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        compatible_video(before, after)


def test_incomplete_audit_is_rejected():
    audit = dict(heldout_sha256=["test", "val"], new_rows=[dict(sha256="new", positive_frames=2)])
    source = dict(train_video_hashes=["old"], test_sha256="test", validation_sha256="val")
    with pytest.raises(ValueError, match="fully audited"):
        incremental_snapshot(audit, source)
