from collections import Counter

import numpy as np
import torch

from scripts.anti_uav.calibrate_addon_branches import (
    add_counts, branch_nms, gate_branch_scores, matches_at_half, metrics, select_recall_safe,
)


def test_branch_threshold_is_applied_before_nms():
    raw = torch.tensor([[0, 0, 10, 10, .04, 0], [0, 0, 10, 10, .05, 1]], dtype=torch.float32)
    assert branch_nms(raw, .03, .03)[0, 5] == 1
    assert branch_nms(raw, .03, .06)[0, 5] == 0
    assert len(branch_nms(raw, .06, .06)) == 0


def test_empty_boxes_and_whole_frame_boxes_are_supported():
    assert branch_nms(torch.empty(0, 6), .03, .03).shape == (0, 6)
    raw = torch.tensor([[0, 0, 960, 544, .8, 0], [10, 10, 12, 12, .7, 1]])
    assert len(branch_nms(raw, .03, .03)) == 2


def test_multi_gt_matching_is_one_to_one():
    gt = torch.tensor([[0., 0, 10, 10], [20, 20, 30, 30]])
    boxes = torch.tensor([[0., 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]])
    pairs = matches_at_half(gt, boxes)
    assert len(pairs) == 2 and len(set(pairs[:, 1])) == 2
    assert matches_at_half(torch.empty(0, 4), boxes).shape == (0, 2)


def test_counts_include_negative_frames_and_tiny_targets():
    c = Counter()
    gt = torch.tensor([[0., 0, 6, 6]])
    det = torch.tensor([[0., 0, 6, 6, .5, 1], [20, 20, 24, 24, .1, 0]])
    add_counts(c, gt, det, {"long_4to8px": [True]})
    add_counts(c, torch.empty(0, 4), det[:1], {"long_4to8px": []})
    m = metrics(c)
    assert (m["tp"], m["fp"], m["fn"], m["frames"]) == (1, 2, 0, 2)
    assert m["tp_p2"] == 1 and m["fp_p2"] == 1 and m["fp_legacy"] == 1
    assert m["tiny_recall"] == 1 and m["negative_fp_frames"] == 1


def test_selection_preserves_tiny_and_large_recall_not_only_total():
    base = dict(tp=10, fp=20, long_4to8px_gt=6, long_4to8px_tp=4,
                area_ge80pct_gt=2, area_ge80pct_tp=2, legacy_threshold=.03, p2_threshold=.03)
    unsafe_tiny = dict(base, tp=11, fp=1, long_4to8px_tp=3, p2_threshold=.1)
    unsafe_large = dict(base, tp=11, fp=2, area_ge80pct_tp=1, p2_threshold=.1)
    safe = dict(base, fp=15, p2_threshold=.05)
    assert select_recall_safe([base, unsafe_tiny, unsafe_large, safe], base) == safe


def test_matching_matches_repository_validator():
    from types import SimpleNamespace
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    gt = torch.tensor([[0., 0, 8, 8], [2, 2, 10, 10], [20, 20, 30, 30]])
    det = torch.tensor([[1., 1, 9, 9], [0, 0, 8, 8], [20, 20, 30, 30]])
    reference = BaseValidator.match_predictions(SimpleNamespace(iouv=torch.tensor([.5])),
        torch.zeros(len(det)), torch.zeros(len(gt)), box_iou(gt, det))
    expected = np.flatnonzero(reference[:, 0].numpy())
    assert sorted(matches_at_half(gt, det)[:, 1]) == expected.tolist()


def test_live_gate_preserves_coordinates_and_accepted_scores():
    pred = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5)
    pred[:, 4] = torch.tensor([.02, .04, .04, .08, .9])
    before = pred.clone()
    result = gate_branch_scores(pred, 2, .075, .03)
    torch.testing.assert_close(pred, before)
    torch.testing.assert_close(result[:, :4], pred[:, :4])
    torch.testing.assert_close(result[:, 4], torch.tensor([[0., .04, 0., .08, .9]]))


def test_live_gate_and_cached_nms_agree():
    from ultralytics.utils import ops
    pred = torch.tensor([[[5., 5, 30, 50, 60, 70, 80], [5, 5, 30, 50, 60, 70, 80],
                          [10, 10, 4, 4, 4, 4, 4], [10, 10, 4, 4, 4, 4, 4],
                          [.05, .04, .5, .01, .02, .04, .8]]])
    raw = torch.cat((ops.xywh2xyxy(pred[0, :4].T), pred[0, 4, :, None],
                     torch.tensor([[1.], [0.], [0.], [0.], [0.], [0.], [0.]])), dim=1)
    cached = branch_nms(raw, .03, .06)
    live = ops.non_max_suppression(gate_branch_scores(pred, 1, .03, .06), .001, .45, max_det=100)[0]
    torch.testing.assert_close(cached[:, :5], live[:, :5])
