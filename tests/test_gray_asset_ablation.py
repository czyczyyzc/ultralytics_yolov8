from copy import deepcopy

import pytest

from scripts.anti_uav.prepare_gray_asset_ablation import assert_asset_only_configs, select_records
from scripts.anti_uav.online_gray_replacement import select_variant


def test_screened_assets_only_and_preserve_original_enabled_ids():
    original = dict(records=[dict(id="1"), dict(id="24")])
    candidates = dict(records=[dict(id="50", status="visually_screened_compositing_candidate", cutout="50.png")])
    result = select_records(original, candidates, ["1"])
    assert [r["id"] for _, r in result] == ["1", "50"]
    candidates["records"][0]["status"] = "rejected"
    with pytest.raises(ValueError, match="screened"):
        select_records(original, candidates, ["1"])


def test_duplicate_assets_rejected():
    original = dict(records=[dict(id="1")])
    candidates = dict(records=[dict(id="1", status="visually_screened_compositing_candidate", cutout="1.png")])
    with pytest.raises(ValueError, match="Overlapping"):
        select_records(original, candidates, ["1"])


def test_replacement_decisions_do_not_depend_on_catalog_size():
    for frame in range(10):
        for epoch in range(30):
            for occurrence in range(3):
                old = select_variant(20260918, str(frame), epoch, occurrence, 53)
                new = select_variant(20260918, str(frame), epoch, occurrence, 328)
                assert (old is None) == (new is None)


def test_asset_only_config_guard(tmp_path):
    for name in ("train", "val", "neg"):
        (tmp_path/name).write_text(name)
    a = dict(path="old", train=str(tmp_path/"train"), val=str(tmp_path/"val"),
             label_sampling=dict(negative_pool=str(tmp_path/"neg"), negatives_per_epoch=12),
             online_replacement=dict(cache="old", replacement_probability=.5, seed=18))
    b = deepcopy(a)
    b["online_replacement"]["cache"] = "new"
    assert_asset_only_configs(a, b)
    assert a["online_replacement"]["cache"] == "old"
    b["online_replacement"]["replacement_probability"] = .8
    with pytest.raises(ValueError, match="policy"):
        assert_asset_only_configs(a, b)
    b = deepcopy(a)
    b["label_sampling"]["negatives_per_epoch"] = 13
    with pytest.raises(ValueError, match="policy"):
        assert_asset_only_configs(a, b)
