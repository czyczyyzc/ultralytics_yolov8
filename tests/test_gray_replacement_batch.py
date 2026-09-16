import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.anti_uav import build_gray_replacement_batch as batch


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_selection_balances_size_and_is_reproducible():
    rows = [dict(frame=i * 10 + k * 1000, size_bin=name)
            for k, name in enumerate(("le8", "8to16", "16to32", "gt32")) for i in range(10)]
    chosen = batch.select_frames(rows, 12, 42)
    assert chosen == batch.select_frames(rows, 12, 42)
    assert len({row["frame"] for row in chosen}) == 12
    assert batch.Counter(row["size_bin"] for row in chosen) == dict(le8=3, **{"8to16": 3, "16to32": 3, "gt32": 3})
    assert batch.size_bin([0, 0, 24, 16], 1920, 1080) == "le8"


def test_training_allowlist_excludes_heldout_hashes(tmp_path):
    old = tmp_path / "old"
    rows = [dict(sha256=d, video=d) for d in ("train", "val", "test", "unrelated")]
    write_json(old / "manifest.json", dict(approved_tasks=rows))
    write_json(tmp_path / "manifest.json", dict(approved_data=str(old / "data.yaml"),
        train_video_hashes=["train"], test_sha256="test", validation_sha256="val", appended_videos=[]))
    (tmp_path / "train_hardneg.txt").write_text("/images/train/000001.jpg\n" * 2)
    _, records, groups = batch.training_records(tmp_path)
    assert records == rows[:1]
    assert len(groups["/images/train"]) == 1
    meta = json.loads((tmp_path / "manifest.json").read_text())
    meta["train_video_hashes"].append("test")
    write_json(tmp_path / "manifest.json", meta)
    with pytest.raises(ValueError, match="overlap"):
        batch.training_records(tmp_path)


def test_candidate_selection_requires_approved_and_current_training_membership(tmp_path):
    coco = tmp_path / "coco/annotations.json"
    write_json(coco, dict(images=[dict(id=i+100, frame_index=i, width=320, height=240) for i in range(5)],
        annotations=[dict(image_id=i+100, bbox=[100, 100, 10, 10]) for i in range(5)]))
    approved = tmp_path / "manifest.json"
    write_json(approved, dict(video=dict(sha256="video"), files=[dict(path="coco/annotations.json", sha256=batch.sha256(coco))],
        frames=dict(indexBase=0, includedFrameIndices=list(range(5)),
                    excludedUncertainFrameIndices=[1], excludedUnreviewedFrameIndices=[2])))
    row = dict(task=str(tmp_path), manifest_sha256=batch.sha256(approved), sha256="video",
               image_directory="/images", label_directory="/labels")
    rows, counts = batch.candidate_frames(row, {"/images": {str(i): Path(f"/images/{i}.jpg") for i in range(4)}})
    assert [r["frame"] for r in rows] == [0, 3]
    assert counts["excluded_uncertain_or_unreviewed_frames"] == 2
    coco.write_text("{}")
    with pytest.raises(ValueError, match="COCO changed"):
        batch.candidate_frames(row, {})


def test_batch_publishes_success_only_and_preserves_inputs(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("manifest.json", "train_hardneg.txt", "val_monitor.txt", "train_hardneg_gray_monitor.yaml"):
        (dataset / name).write_text("immutable")
    source = tmp_path / "source.png"
    gray = np.full((240, 320), 100, np.uint8)
    Image.fromarray(gray).save(source)
    lab = tmp_path / "source.txt"
    box = [100, 100, 16, 16]
    lab.write_text(batch.normalize_box(box, 320, 240))
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    meta = dict(test_sha256="test", validation_sha256="val", validation_video=dict(video="heldout.mp4"))
    row = dict(video="train.mp4", sha256="123456789abcdef", task="/task")
    monkeypatch.setattr(batch, "training_records", lambda _: (meta, [row], {}))
    monkeypatch.setattr(batch, "load_assets", lambda *_: [(dict(id="1", model="test"), np.zeros((2, 2, 4), np.uint8))])
    sources = [dict(frame=i, image=str(source), label=str(lab), box=box, width=320, height=240, size_bin="gt32") for i in (0, 20)]
    monkeypatch.setattr(batch, "candidate_frames", lambda *_: (sources, {}))

    class FakeTemporal:
        def __init__(self, *_):
            self.cache = {}
        def close(self):
            pass

    monkeypatch.setattr(batch, "TemporalBackgrounds", FakeTemporal)

    def replace(gray, box, rgba, seed, **kwargs):
        if kwargs["frame_index"] == 20:
            raise batch.SkipSample("no_safe_temporal_donor:test")
        result, mask = gray.copy(), np.zeros_like(gray)
        result[100:116, 100:116] = 50
        mask[100:116, 100:116] = 255
        return result, mask, box, dict(new_box_xywh=box)

    monkeypatch.setattr(batch, "replace_target", replace)
    args = argparse.Namespace(dataset=dataset, catalog=catalog, output=tmp_path / "out", per_video=2,
                              variants=1, previews=0, seed=1, exclude_assets="24,25")
    batch.build(args)
    result = json.loads((args.output / "manifest.json").read_text())
    assert result["summary"]["accepted"] == 1
    assert len(result["skipped"]) == 1
    assert len(list((args.output / "images").iterdir())) == 1
    assert len((args.output / "train_synthetic.txt").read_text().splitlines()) == 1
    assert np.array_equal(np.asarray(Image.open(source)), gray)
    assert result["summary"]["protected_inputs_unchanged"]
    assert json.loads((args.output / "status.json").read_text())["stage"] == "complete"
    with pytest.raises(FileExistsError):
        batch.build(args)
