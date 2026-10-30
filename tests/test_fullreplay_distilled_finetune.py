from copy import deepcopy
import json

import numpy as np
import pytest
import yaml

from scripts.anti_uav.prepare_fixedwing_finetune import write_split
from scripts.anti_uav.run_fullreplay_distilled_finetune import (
    LOSS_OPTIONS, choose_eligible, prepare_data, retainable, verify_full_replay,
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


def test_legacy_yaml_uses_original_protocol_pins_and_no_subsampling(tmp_path):
    old, full = tmp_path/"old", tmp_path/"full"
    old.mkdir(); full.mkdir()
    legacy = [row("/images/a/0.jpg")]*2
    added = row("/images/new0/0.jpg")
    validation = [row("/images/Video00009/0.jpg")]
    old_train = old/"train.txt"
    old_pins = {str(old_train): write_split(old_train, legacy, "1.0.3", "old")}
    (old/"train_online_gray_monitor.yaml").write_text(yaml.safe_dump(dict(train=str(old_train))))
    train, val = full/"train.txt", full/"val.txt"
    pins = {str(train): write_split(train, legacy+[added], "1.0.3", "frozen"),
            str(val): write_split(val, validation, "1.0.3", "frozen")}
    config = dict(train=str(train), val=str(val), trusted_label_caches=pins,
        tiny_sampling=dict(policy="balanced_v2"), online_replacement=dict(cache="/cache"))
    (full/"train_online_gray_monitor.yaml").write_text(yaml.safe_dump(config))
    manifest = dict(train_video_hashes=[str(i) for i in range(73)],
        latest_appended_videos=[dict(image_directory=f"/images/new{i}") for i in range(21)],
        test_sha256="test4", validation_sha256="val9", added_positive_samples=1,
        added_negative_samples=0, epoch_positive=3, epoch_negative=0,
        sampler_audit=dict(policy="balanced_v2"))
    (full/"manifest.json").write_text(json.dumps(manifest))
    (full/"annotation_snapshot.json").write_text("{}")
    teacher, student, audit = prepare_data(full, old, tmp_path/"output", old_pins)
    assert audit["legacy_positive_slots"] == 2
    assert yaml.safe_load(student.read_text())["train"] == str(train)
    assert yaml.safe_load(teacher.read_text())["online_replacement"]["epoch_offset"] == 42
    assert yaml.safe_load(student.read_text())["online_replacement"]["epoch_offset"] == 57
    assert "trusted_label_caches" not in yaml.safe_load((old/"train_online_gray_monitor.yaml").read_text())
