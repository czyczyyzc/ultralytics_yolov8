import json

import numpy as np
import pytest
import torch
import yaml

from scripts.anti_uav.run_p3_independent_curriculum_training import (
    audit_label_caches, stage_yaml, common_train, stage_resume_state, add_status_callback,
)
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


def saved_stage(root, completed=2, target=30, stripped=False):
    stage = root / "p3"
    (stage / "weights").mkdir(parents=True)
    (stage / "results.csv").write_text("epoch,loss\n" + "".join(f"{i},1\n" for i in range(1, completed + 1)))
    checkpoint = dict(epoch=-1 if stripped else completed - 1, train_args=dict(epochs=target),
                      optimizer=None if stripped else dict(state={1: dict(step=3)}),
                      ema=torch.nn.Linear(1, 1))
    for name in ("last", "best"):
        torch.save(checkpoint, stage / f"weights/{name}.pt")
    return stage


def test_partial_best_does_not_skip_foundation(tmp_path):
    stage = saved_stage(tmp_path)
    state = stage_resume_state(stage, 30, True)
    assert not state["complete"]
    assert state["completed_epochs"] == 2
    assert state["checkpoint"] == str(stage / "weights/last.pt")
    assert len(state["checkpoint_sha256"]) == 64


@pytest.mark.parametrize("stripped", [False, True])
def test_only_full_epoch_budget_marks_a_stage_complete(tmp_path, stripped):
    stage = saved_stage(tmp_path, completed=30, stripped=stripped)
    state = stage_resume_state(stage, 30, True)
    assert state["complete"] and state["completed_epochs"] == 30
    assert state["checkpoint"] is None


def test_stripped_partial_checkpoint_cannot_silently_restart_optimizer(tmp_path):
    stage = saved_stage(tmp_path, stripped=True)
    with pytest.raises(ValueError, match="optimizer/EMA"):
        stage_resume_state(stage, 30, True)


def test_resume_rejects_missing_last_and_inconsistent_history(tmp_path):
    stage = saved_stage(tmp_path)
    (stage / "results.csv").write_text("epoch,loss\n1,1\n")
    with pytest.raises(ValueError, match="history differ"):
        stage_resume_state(stage, 30, True)
    (stage / "weights/last.pt").unlink()
    with pytest.raises(ValueError, match="Missing last"):
        stage_resume_state(stage, 30, True)


def test_new_teacher_is_fresh_while_partial_foundation_resumes(tmp_path):
    assert stage_resume_state(tmp_path / "new", 12, True) == dict(
        complete=False, completed_epochs=0, checkpoint=None)
    stage = saved_stage(tmp_path)
    with pytest.raises(ValueError, match="epoch budget"):
        stage_resume_state(stage, 12, True)


def test_resumed_training_stays_in_original_directory_and_keeps_recipe(tmp_path):
    from types import SimpleNamespace
    data = tmp_path / "data.yaml"
    data.write_text("input_color: gray3\n")
    before = common_train(data, SimpleNamespace(batch=64, device="6", workers=16), 30)
    after = common_train(data, SimpleNamespace(batch=64, device="2", workers=16, resume=True), 30)
    assert after["device"] == "2" and after["exist_ok"] is True
    assert {k: v for k, v in before.items() if k not in ("device", "exist_ok")} == {
        k: v for k, v in after.items() if k not in ("device", "exist_ok")}


def test_runtime_resume_audit_records_restored_optimizer_and_device(tmp_path):
    from types import SimpleNamespace
    (tmp_path / "audit").mkdir()
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda event, fn: callbacks.update({event: fn}))
    add_status_callback(model, tmp_path, "train_native_p3_foundation")
    trainer = SimpleNamespace(resume=True, args=SimpleNamespace(device="2", resume="last.pt"),
                              start_epoch=2, batch_size=64, optimizer=SimpleNamespace(state={1: {}}),
                              ema=SimpleNamespace(updates=7721))
    callbacks["on_train_start"](trainer)
    result = json.loads((tmp_path / "audit/active_resume_train_native_p3_foundation.json").read_text())
    assert result["device"] == "2" and result["start_epoch_zero_based"] == 2
    assert result["optimizer_state_entries"] == 1 and result["ema_updates"] == 7721
