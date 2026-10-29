import json

import pytest

from scripts.anti_uav.evaluate_imported_detector_videos import score_predictions


def annotation(index, status="confirmed"):
    return dict(frame_index=index, status=status, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10)


def test_unknown_and_uncertain_are_not_negative_frames(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("\n".join(json.dumps(dict(frame_index=i, boxes_xyxy_score=[[0, 0, 10, 10, .2]]))
                                     for i in range(4)) + "\n")
    item = dict(frame_count=4, task_status="submitted", revision=9,
                annotations=[annotation(0), annotation(1, "absent"), annotation(2, "uncertain")])
    result = score_predictions(item, predictions)
    assert result["provisional_labels"]
    assert result["unlabelled_frames"] == result["uncertain_frames"] == 1
    assert result["metrics"]["0.03"] == dict(tp=1, fp=1, fn=0, frames=2, gt_boxes=1, recall=1, precision=.5)
    assert result["metrics"]["0.25"]["fn"] == 1
    assert result["frames_with_predictions"]["0.03"] == 4


def test_incomplete_task_has_no_accuracy_metrics(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(dict(frame_index=0, boxes_xyxy_score=[])) + "\n")
    item = dict(frame_count=1, task_status="in_progress", revision=1, annotations=[annotation(0)])
    assert score_predictions(item, predictions)["metrics"] is None


@pytest.mark.parametrize("index,count", [(1, 1), (0, 2)])
def test_missing_or_out_of_order_predictions_rejected(tmp_path, index, count):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(dict(frame_index=index, boxes_xyxy_score=[])) + "\n")
    item = dict(frame_count=count, task_status="submitted", revision=1, annotations=[])
    with pytest.raises(ValueError):
        score_predictions(item, predictions)
