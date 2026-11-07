import json

import numpy as np
import pytest
import yaml

from scripts.anti_uav.run_p3_independent_curriculum_training import audit_label_caches, stage_yaml, common_train
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

    native_path = stage_yaml(source, tmp_path / "native.yaml", 10, replacement_probability=.5)
    native = yaml.safe_load(native_path.read_text())
    assert "online_scale" not in native
    assert native["label_sampling"]["anchor_slots"] == 65
    assert native["online_replacement"]["epoch_offset"] == 13

    balanced_path = stage_yaml(
        source, tmp_path / "balanced.yaml", 20, replacement_probability=.25
    )
    balanced = yaml.safe_load(balanced_path.read_text())
    assert "online_scale" not in balanced
    assert balanced["label_sampling"]["anchor_slots"] == 65
    assert balanced["online_replacement"]["replacement_probability"] == .25
    assert balanced["online_replacement"]["epoch_offset"] == 23


@pytest.mark.parametrize("color,expected", [("gray3", (0.,0.,.3)), ("source", (.015,.4,.3))])
def test_training_color_mode_has_explicit_photometric_parameters(tmp_path, color, expected):
    from types import SimpleNamespace
    path=tmp_path/'data.yaml';path.write_text(yaml.safe_dump(dict(input_color=color)))
    args=SimpleNamespace(batch=64,device='cpu',workers=16)
    config=common_train(path,args,30)
    assert (config['hsv_h'],config['hsv_s'],config['hsv_v'])==expected
    assert config['imgsz']==[544,960]


def test_gray_mode_and_fixed_sampler_are_preserved_across_stages(tmp_path):
    source=dict(input_color='gray3',tiny_sampling=dict(policy='fixed_domain_v1'),
        online_replacement=dict(cache='unchanged',replacement_probability=.5))
    for stage,offset in [('p3',0),('teacher',30),('student',42)]:
        path=stage_yaml(source,tmp_path/(stage+'.yaml'),offset,.5)
        config=yaml.safe_load(path.read_text())
        assert config['input_color']=='gray3'
        assert config['tiny_sampling']['policy']=='fixed_domain_v1'
        assert config['online_replacement']['epoch_offset']==offset


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


def test_curriculum_does_not_reapprove_changed_pinned_cache(tmp_path):
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    for listing in (train, val):
        listing.write_text("/dataset/image.jpg\n")
        with listing.with_suffix(".cache").open("wb") as stream:
            np.save(stream, dict(labels=[], hash="original", results=(1, 0, 0, 0, 1),
                                 msgs=[], version=DATASET_CACHE_VERSION))
    source = dict(train=str(train), val=str(val))
    source["trusted_label_caches"] = audit_label_caches(source)
    with train.with_suffix(".cache").open("wb") as stream:
        np.save(stream, dict(labels=[], hash="changed", results=(1, 0, 0, 0, 1),
                             msgs=[], version=DATASET_CACHE_VERSION))
    with pytest.raises(ValueError, match="label cache changed"):
        audit_label_caches(source)


def test_curriculum_rejects_missing_pin_in_existing_pin_map(tmp_path):
    listing = tmp_path / "train.txt"
    listing.write_text("/dataset/image.jpg\n")
    listing.with_suffix(".cache").write_bytes(b"must not be unpickled")
    with pytest.raises(ValueError, match="Missing audited cache pin"):
        audit_label_caches(dict(train=str(listing), val=str(listing), trusted_label_caches={"wrong-key": {}}))
