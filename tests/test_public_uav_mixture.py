from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.anti_uav.run_public_uav_research import (
    bind_control, check_groups, completed_manifest, positive_candidates, read_positive_label, sampling_extension,
)


def test_subpixel_boxes_remain_valid(tmp_path):
    label = tmp_path / "tiny.txt"
    label.write_text("0 0.5 0.5 0.0001 0.0002\n")
    assert read_positive_label(label)[0][3] == .0001


@pytest.mark.parametrize("text", ["", "1 .5 .5 .1 .1", "0 nan .5 .1 .1", "0 .5 .5 0 .1", "0 .98 .5 .2 .1"])
def test_unusable_labels_cannot_become_background(tmp_path, text):
    path = tmp_path / "bad.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_positive_label(path)


def test_public_group_must_not_cross_train_test():
    with pytest.raises(ValueError, match="crosses splits"):
        check_groups([dict(dataset="h", group="video1", split=s) for s in ["train", "test"]])


def test_unknown_split_is_not_training():
    with pytest.raises(ValueError, match="Unknown public split"):
        check_groups([dict(dataset="h", group="v", split="unknown")])


def test_distinct_datasets_may_share_group_names():
    check_groups([dict(dataset="a", group="v", split="train"), dict(dataset="b", group="v", split="test")])


def test_repeated_native_anchors_preserved_and_negative_fraction_retained():
    config = dict(label_sampling=dict(negative_pool_count=100, anchor_slots=90,
                  negatives_per_epoch=10, target_negative_fraction=.15))
    old = deepcopy(config)
    new, exposure = sampling_extension(config, 190, 85)
    assert config == old
    assert new["label_sampling"]["anchor_slots"] == 175
    assert exposure["native_positive"] == 85
    assert exposure["epoch_positive"] == 170
    assert exposure["epoch_negative"] == 30
    assert exposure["epoch_slots"] == 200


def test_changed_native_schedule_fails_closed():
    with pytest.raises(ValueError, match="Baseline schedule"):
        sampling_extension(dict(label_sampling=dict(negative_pool_count=2, anchor_slots=4)), 10, 3)


def test_insufficient_reviewed_negatives_fails_closed():
    with pytest.raises(ValueError, match="reviewed pool"):
        sampling_extension(dict(label_sampling=dict(negative_pool_count=5, anchor_slots=90,
            negatives_per_epoch=5, target_negative_fraction=.15)), 95, 1000)


def test_incomplete_or_modified_public_exports_rejected(tmp_path):
    import json
    from scripts.anti_uav.run_public_uav_research import sha256
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(dict(complete=False, manifest_sha256=sha256(manifest))))
    with pytest.raises(ValueError, match="incomplete"):
        completed_manifest(manifest)
    summary.write_text(json.dumps(dict(complete=True, manifest_sha256=sha256(manifest))))
    assert completed_manifest(manifest)["complete"]
    manifest.write_text("{}\n{}\n")
    with pytest.raises(ValueError, match="manifest changed"):
        completed_manifest(manifest)


def test_other_native69_data_cannot_be_called_the_control(tmp_path):
    import json
    from types import SimpleNamespace
    (tmp_path / "protocol.json").write_text(json.dumps(dict(data=str(tmp_path / "control.yaml"))))
    with pytest.raises(ValueError, match="not the control"):
        bind_control(SimpleNamespace(baseline_run=tmp_path, baseline_yaml=tmp_path / "other.yaml"))


def test_empty_public_labels_are_excluded_not_negative():
    positive = dict(split="train", boxes=1, image="a")
    empty = dict(split="train", boxes=0, image="b")
    candidates, excluded = positive_candidates([positive, empty])
    assert candidates == [positive]
    assert excluded == [dict(image="b", reason="no_positive_gt_not_assumed_negative")]
    with pytest.raises(ValueError, match="box count"):
        positive_candidates([dict(split="train", image="unknown")])


def test_prepare_preserves_native_repeats_and_excludes_heldout_pixels(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import cv2
    import numpy as np
    import yaml
    from scripts.anti_uav import run_native_pool_comparison
    from scripts.anti_uav.run_public_uav_research import prepare, sha256

    base = tmp_path / "base"
    base.mkdir()
    def picture(name, value):
        path = tmp_path / (name + ".png")
        assert cv2.imwrite(str(path), np.full((8, 8, 3), value, np.uint8))
        return str(path)
    validation, test = picture("val", 11), picture("test", 22)
    added, duplicate = picture("public", 33), picture("duplicate_test", 22)
    public_val = picture("public_val", 44)
    native_paths = ["/native/repeated.png"] * 10 + [f"/native/neg{i}.png" for i in range(8)]
    train, val = base / "train.txt", base / "val.txt"
    train.write_text("\n".join(native_paths) + "\n")
    val.write_text(validation + "\n")
    labels = [dict(im_file=p, cls=np.empty((0, 1)) if "/neg" in p else np.ones((1, 1))) for p in native_paths]
    pins = {}
    for listing, rows in [(train, labels), (val, [dict(im_file=validation, cls=np.ones((1, 1)))])]:
        cache = dict(labels=rows, version="test", hash="old", results=(len(rows), 0, 0, 0, len(rows)))
        with listing.with_suffix(".cache").open("wb") as stream:
            np.save(stream, cache, allow_pickle=True)
        pins[str(listing)] = dict(source_sha256=sha256(listing), cache_sha256=sha256(listing.with_suffix(".cache")))
    data = base / "train.yaml"
    data.write_text(yaml.safe_dump(dict(train=str(train), val=str(val), trusted_label_caches=pins,
        label_sampling=dict(negative_pool_count=8, anchor_slots=10, negatives_per_epoch=2,
                            target_negative_fraction=1/6))))
    (base / "manifest.json").write_text(json.dumps(dict(train_video_hashes=list(range(69)),
        epoch_positive=10, epoch_negative=2)))
    test_list = tmp_path / "test.txt"
    test_list.write_text(test + "\n")
    test_yaml = tmp_path / "test.yaml"
    test_yaml.write_text(yaml.safe_dump(dict(val=str(test_list))))
    monkeypatch.setattr(run_native_pool_comparison, "HOLDOUT", test_yaml)
    label = tmp_path / "public.txt"
    label.write_text("0 .5 .5 .1 .1\n")
    records = [dict(dataset="dut", split=split, group=str(i), image=image, label=str(label), boxes=1)
               for i, (split, image) in enumerate([("train", added), ("train", duplicate), ("val", public_val)])]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(x) for x in records) + "\n")
    (tmp_path / "summary.json").write_text(json.dumps(dict(complete=True, manifest_sha256=sha256(manifest))))
    output = tmp_path / "mixed"
    prepare(SimpleNamespace(baseline_yaml=data, data_output=output, public_manifest=[manifest], workers=2))
    result = (output / "train_public.txt").read_text().splitlines()
    assert result == native_paths + [added]
    assert train.read_text().splitlines() == native_paths
    assert (output / "val_monitor.txt").read_bytes() == val.read_bytes()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["dataset_counts"] == {"dut": 1}
    assert summary["excluded"] == 1
