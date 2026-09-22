import numpy as np
import pytest

from scripts.anti_uav.evaluate_p3_gray_pair import pool_native
from ultralytics.utils.metrics import ap_per_class


def entry(tp, fp, fn, frames, scores, correct, tiny_gt, tiny_tp):
    metrics = {}
    scales = {}
    for conf in (.01, .03, .05):
        q = f"native/c{conf:.2f}/"
        metrics.update({q + k: v for k, v in dict(TP=tp, FP=fp, FN=fn, FRAMES=frames).items()})
        scales[(conf, "long_4to8px")] = dict(gt=tiny_gt, tp=tiny_tp)
    return dict(metrics=metrics, scales=scales,
                arrays=dict(tp=np.tile(np.asarray(correct, dtype=bool)[:, None], (1, 10)),
                            conf=np.asarray(scores), pred_cls=np.zeros(len(scores)), target_cls=np.zeros(tp+fn)))


def test_micro_counts_are_not_mean_of_video_percentages():
    a = entry(1, 0, 0, 2, [.8], [True], 1, 1)
    b = entry(1, 2, 2, 8, [.9, .7, .6], [False, False, True], 3, 1)
    m, arrays = pool_native([a, b])
    assert m["native/c0.03/TP"] == 2
    assert m["native/c0.03/FRAMES"] == 10
    assert m["native/c0.03/P"] == .5
    assert m["native/c0.03/R"] == .5
    assert m["native/c0.03/long_4to8px/R"] == .5
    expected = ap_per_class(**arrays, plot=False)[5][:, 0].mean()
    assert m["native/mAP50"] == pytest.approx(expected)
    separate = [ap_per_class(**e["arrays"], plot=False)[5][:, 0].mean() for e in (a, b)]
    assert abs(expected - np.mean(separate)) > .01


def test_no_predictions_with_gt_stays_zero_ap_and_recall():
    e = entry(0, 0, 3, 5, [], [], 0, 0)
    m, arrays = pool_native([e, e])
    assert m["native/mAP50"] == 0
    assert m["native/c0.03/R"] == 0
    assert m["native/c0.03/FN"] == 6
    assert m["native/c0.03/long_4to8px/R"] is None
    assert arrays["tp"].shape == (0, 10)
