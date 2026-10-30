from copy import deepcopy

import numpy as np
import pytest

from scripts.anti_uav.run_fullreplay_distilled_finetune import (
    LOSS_OPTIONS, choose_eligible, retainable, verify_full_replay,
)


def row(name, size=.01):
    return dict(im_file=name, shape=(1080, 1920), cls=np.zeros((1, 1)),
                bboxes=np.array([[.5, .5, size, size]], dtype=np.float32))


def test_all_old_multiplicities_and_full_frame_boxes_preserved():
    old = [row("tiny"), row("tiny"), row("full_frame", .95)]
    report = verify_full_replay(old, deepcopy(old)+[row("new_video")])
    assert report["legacy_candidate_slots"] == 3
    assert report["legacy_positive_slots"] == 3
    assert report["complete_legacy_exposure_preserved"]
    with pytest.raises(ValueError, match="multiplicity"):
        verify_full_replay(old, [old[0], old[2], row("new_video")])
    wrong = deepcopy(old)
    wrong[-1]["bboxes"][0, 0] = .1
    with pytest.raises(ValueError, match="label"):
        verify_full_replay(old, wrong)


def test_retention_is_hard_constraint_not_shared_penalty():
    baseline = {"native/c0.03/R": .64, "native/c0.03/long_4to8px/R": .64, "fitness": .7}
    bad = dict(baseline, **{"native/c0.03/R": .50, "fitness": .95})
    good = dict(baseline, **{"native/c0.03/R": .63, "fitness": .71})
    assert not retainable(bad, baseline)
    assert choose_eligible({"bad": bad, "good": good}, baseline) == "good"
    assert choose_eligible({"bad": bad}, baseline) is None
    assert not retainable(dict(good, fitness=float("nan")), baseline)


def test_original_distillation_and_localization_restored():
    assert LOSS_OPTIONS["auxiliary_weight"] == .25
    assert LOSS_OPTIONS["p3_cls_weight"] == .10
    assert LOSS_OPTIONS["p3_dfl_weight"] == .05
    assert LOSS_OPTIONS["p2_cls_weight"] == .25
    assert LOSS_OPTIONS["p2_box_weight"] == .10
    assert LOSS_OPTIONS["feature_weight"] == .10
    assert LOSS_OPTIONS["localization_weight"] == .12
    assert LOSS_OPTIONS["localization_min_px"] == 4
    assert LOSS_OPTIONS["localization_max_px"] == 960
