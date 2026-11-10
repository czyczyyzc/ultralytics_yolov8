import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from scripts.anti_uav.evaluate_fixedwing_replay import frame_ground_truth, pooled, score_cache, snapshot
from scripts.anti_uav.render_pt_detector_video import inference_frames


def annotation(status="confirmed", box=(1, 1, 11, 11), frame=0):
    return dict(frame_index=frame, status=status,
                **dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), box)))


def item():
    return dict(frame_count=3, frame_width=100, frame_height=100, is_published=1,
                task_status="approved", revision=4, approved_revision=4,
                annotations=[annotation(), annotation("uncertain", frame=1), annotation("absent", frame=2)])


def cache(tmp_path, indices=(0, 1, 2)):
    path = tmp_path / "predictions.jsonl"
    rows = {0: [[1, 1, 11, 11, .9], [30, 30, 40, 40, .002]], 1: [[50, 50, 60, 60, .99]], 2: []}
    path.write_text("".join(json.dumps(dict(frame_index=i, boxes_xyxy_score=rows[i])) + "\n" for i in indices))
    return path


def test_gray3_inference_preserves_original_visualization_pixels():
    image = np.full((8, 16, 3), [20, 80, 220], np.uint8)
    original = image.copy()
    frames, changed = inference_frames([image], "gray3")
    assert changed == 1
    assert np.array_equal(image, original)
    assert np.array_equal(frames[0][..., 0], frames[0][..., 1])
    assert np.array_equal(frames[0][..., 1], frames[0][..., 2])
    passthrough = [image]
    assert inference_frames(passthrough, "source")[0] is passthrough
    assert inference_frames(frames, "gray3")[1] == 0


def test_missing_and_uncertain_annotations_are_not_background():
    assert frame_ground_truth([], 100, 100) == (None, "unlabelled")
    assert frame_ground_truth([annotation("uncertain")], 100, 100) == (None, "uncertain")
    assert frame_ground_truth([annotation("absent")], 100, 100) == ([], "absent")


def test_small_and_full_image_boxes_are_both_retained():
    for box in ((1, 1, 5, 5), (0, 0, 100, 100)):
        assert frame_ground_truth([annotation(box=box)], 100, 100) == ([list(box)], "positive")
    assert frame_ground_truth([annotation(box=(None, 1, 3, 4))], 100, 100) == (None, "invalid")


def test_low_floor_is_for_ap_not_display_confidence_and_unknown_frames_excluded(tmp_path):
    result, arrays = score_cache(item(), cache(tmp_path), calculate_ap=True)
    metrics = result["metrics"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["frames"], metrics["gt_boxes"]) == (1, 0, 0, 2, 1)
    assert result["ignored_frames"]["uncertain"] == 1
    assert metrics["recall"] == metrics["precision"] == 1
    assert metrics["map50"] > .99 and metrics["map50_95"] > .99
    assert len(arrays["conf"]) == 2
    assert float(arrays["conf"].min()) < .03
    aggregate = pooled([result, result], [arrays, arrays])
    assert aggregate["gt_boxes"] == 2 and aggregate["tp"] == 2
    assert aggregate["map50"] > .99


@pytest.mark.parametrize("updates", [dict(task_status="submitted"), dict(approved_revision=3), dict(is_published=0)])
def test_only_current_reviewed_revision_has_formal_metrics(tmp_path, updates):
    entry = dict(item(), **updates)
    result, arrays = score_cache(entry, cache(tmp_path), calculate_ap=True)
    assert result["metrics"]["gt_boxes"] == 0
    assert result["metrics"]["recall"] is None
    assert result["metrics"]["map50"] is None
    assert result["ignored_frames"]["unreviewed"] == 3
    assert len(arrays["target_cls"]) == 0


@pytest.mark.parametrize("indices", [(0, 2, 1), (0, 1)])
def test_predictions_must_cover_entire_source_in_order(tmp_path, indices):
    with pytest.raises(ValueError, match="sequential|coverage"):
        score_cache(item(), cache(tmp_path, indices), calculate_ap=False)


def test_snapshot_maps_current_task_by_source_hash_not_deleted_old_id(tmp_path):
    database = tmp_path / "labeler.sqlite"
    with sqlite3.connect(database) as con:
        con.executescript("""
            CREATE TABLE videos(id TEXT,source_sha256 TEXT);
            CREATE TABLE tasks(id TEXT,status TEXT,revision INT,approved_revision INT,is_published INT,updated_at INT,video_id TEXT);
            CREATE TABLE annotations(task_id TEXT,frame_index INT,object_id INT,status TEXT,
                bbox_x1 REAL,bbox_y1 REAL,bbox_x2 REAL,bbox_y2 REAL);
            INSERT INTO videos VALUES('v','same-video');
            INSERT INTO tasks VALUES('current','approved',9,9,1,100,'v');
            INSERT INTO annotations VALUES('current',0,0,'confirmed',1,1,5,5);
        """)
    cohort = [dict(task_id="deleted-old-id", source_sha256="same-video"), dict(source_sha256="missing-video")]
    snapshot(cohort, database, {"same-video"})
    assert cohort[0]["task_id"] == "current"
    assert cohort[0]["revision"] == 9
    assert cohort[0]["in_training_source"] is True
    assert len(cohort[0]["annotations"]) == 1
    assert cohort[1]["task_status"] == "no_current_task"
    assert cohort[1]["annotations"] == []
