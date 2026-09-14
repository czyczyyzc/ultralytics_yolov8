import numpy as np

from scripts.anti_uav.audit_gray_training_regression import bucket, compare_labels, summarize


def label(box):
    return dict(shape=(1080, 1920), bboxes=np.asarray(box, dtype=float).reshape(-1, 4),
                normalized=True, bbox_format="xywh")


def test_weighted_exposure_and_pixel_scale():
    positive = label([[.5, .5, 8/1920, 4/1080]])
    result = summarize([("p", positive, 5, "g"), ("n", label([]), 2, "g")])
    assert result["positive_exposures"] == 5
    assert result["unique_positive"] == 1
    assert result["negative_exposures"] == 2
    assert result["positive_boxes_by_long_edge"] == {"le4": 5}
    assert result["long_edge_at_960x544_weighted"]["p50"] == 4


def test_exclusive_bins():
    assert [bucket(x) for x in (4, 4.1, 6, 7, 33)] == ["le4", "le6", "le6", "le8", "gt32"]


def test_label_mismatch_detection():
    p = label([[.5, .5, .1, .1]])
    q = label([[.6, .5, .1, .1]])
    old = {("v", 0): ("a", p, 2), ("v", 1): ("b", p, 1)}
    new = {("v", 0): ("c", q, 1), ("v", 1): ("d", label([]), 1)}
    result = compare_labels(old, new)
    assert result["changed_boxes_over_0_1px"] == 1
    assert len(result["presence_mismatches"]) == 1
    assert result["common_unique_frames"] == 2
