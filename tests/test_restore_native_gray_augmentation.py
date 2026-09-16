import json

import pytest
import yaml

from scripts.anti_uav.restore_native_gray_augmentation import native_schedule, restore


def test_keeps_repeats_order_negatives_and_border_sources():
    original = ["/images/gray/tiny.jpg", "/images/gray/tiny.jpg", "/images/gray/empty.jpg",
                "/images/gray/full_frame.jpg"]
    assert native_schedule(original + ["/images/zoom_train/extra.jpg"], ["/images/gray_val/one.jpg"]) == original


@pytest.mark.parametrize("path", ["/images/gray_val/one.jpg", "/images/zoom_val/one.jpg",
                                 "/images/Video00004/one.jpg"])
def test_rejects_holdout(path):
    with pytest.raises(ValueError):
        native_schedule([path], [])


def test_restores_config_without_touching_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "restored"
    source.mkdir()
    train = "/images/gray/tiny.jpg\n/images/gray/tiny.jpg\n/images/gray/empty.jpg\n"
    val = "/images/gray_val/test.jpg\n/images/zoom_val/stress.jpg\n"
    (source / "train.txt").write_text(train)
    (source / "val.txt").write_text(val)
    config = dict(train=str(source / "train.txt"), val=str(source / "val.txt"), names={0: "drone"},
                  online_scale=dict(views_per_donor=4))
    (source / "train_hardneg_gray_monitor.yaml").write_text(yaml.safe_dump(config))
    manifest = dict(append_only_entries=3, negative=1, validation_video=dict(sha256="0123456789abcdef"))
    (source / "manifest.json").write_text(json.dumps(manifest))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    audit = restore(source, output)
    assert audit["final_entries"] == 3 and audit["online_additional_slots"] == 0
    assert audit["training_started"] is False
    assert (output / "train_hardneg.txt").read_text() == train
    assert (output / "val_monitor.txt").read_text() == val
    assert "online_scale" not in yaml.safe_load((output / "train_hardneg_gray_monitor.yaml").read_text())
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(FileExistsError):
        restore(source, output)
