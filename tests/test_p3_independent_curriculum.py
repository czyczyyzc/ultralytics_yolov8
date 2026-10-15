import json

import numpy as np
import pytest
import yaml

from scripts.anti_uav.run_p3_independent_curriculum_training import audit_label_caches, stage_yaml
from ultralytics.data.dataset import DATASET_CACHE_VERSION, load_audited_dataset_cache


def test_stage_yaml_rebalances_appended_scale_slots(tmp_path):
    donors = tmp_path / "donors.json"
    donors.write_text(json.dumps({"donors": [{"im_file": str(index)} for index in range(7)]}))
    source = {
        "train": "train.txt",
        "val": "val.txt",
        "names": {0: "drone"},
        "label_sampling": {"anchor_slots": 100},
        "online_replacement": {"cache": "cache", "epoch_offset": 3},
        "online_scale": {"donor_manifest": str(donors), "views_per_donor": 5},
    }

    native_path = stage_yaml(source, tmp_path / "native.yaml", 10, scale_views=0)
    native = yaml.safe_load(native_path.read_text())
    assert "online_scale" not in native
    assert native["label_sampling"]["anchor_slots"] == 65
    assert native["online_replacement"]["epoch_offset"] == 13

    balanced_path = stage_yaml(
        source, tmp_path / "balanced.yaml", 20, scale_views=2, keep_replacement=False
    )
    balanced = yaml.safe_load(balanced_path.read_text())
    assert balanced["online_scale"]["views_per_donor"] == 2
    assert balanced["label_sampling"]["anchor_slots"] == 79
    assert "online_replacement" not in balanced


def test_audited_cache_avoids_per_member_metadata_scan(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("/dataset/frame_1.jpg\n")
    val.write_text("/dataset/frame_2.jpg\n")
    for source, identity in ((train, "train-id"), (val, "val-id")):
        cache = {
            "labels": [],
            "hash": identity,
            "results": (1, 0, 0, 0, 1),
            "msgs": [],
            "version": DATASET_CACHE_VERSION,
        }
        with source.with_suffix(".cache").open("wb") as stream:
            np.save(stream, cache)

    records = audit_label_caches({"train": str(train), "val": str(val)})
    loaded = load_audited_dataset_cache(
        train.with_suffix(".cache"), train, {"trusted_label_caches": records}
    )
    assert loaded["hash"] == "train-id"
    assert records[str(train.resolve())]["entries"] == 1

    train.write_text("/dataset/changed.jpg\n")
    with pytest.raises(ValueError, match="dataset list changed"):
        load_audited_dataset_cache(
            train.with_suffix(".cache"), train, {"trusted_label_caches": records}
        )
