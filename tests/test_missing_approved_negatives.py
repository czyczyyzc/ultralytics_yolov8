"""Torch-free: pytest --noconftest --import-mode=importlib tests/test_missing_approved_negatives.py."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.anti_uav import prepare_missing_approved_negatives as supplement


def dump(path, value):
    path.write_text(json.dumps(value))


def annotation(index, status):
    return dict(frame_index=index, object_id="drone", status=status,
                bbox_x1=1., bbox_y1=2., bbox_x2=1.25, bbox_y2=2.5)


@pytest.fixture
def frozen(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video_path = tmp_path / "video.bin"
    video_path.write_bytes(b"immutable synthetic source video")
    digest = supplement.sha256(video_path)
    name = "clip_" + digest[:8] + ".mp4"
    video = dict(task_id="task-a", source_sha256=digest, source_relpath=name, name=name,
                 video=str(video_path), frame_width=32, frame_height=24, frame_count=6, fps=30.,
                 status="approved", revision=7, approved_revision=7,
                 annotations=[annotation(0, "absent"), annotation(1, "confirmed"),
                              annotation(2, "uncertain"), annotation(3, "absent")])
    group = source / "images/approved_gray" / ("clip_" + digest[:12])
    cache = dict(labels=[dict(im_file=str(group / "000000001.jpg"))] * 2)
    cache_path = source / "train_hardneg.cache"
    with cache_path.open("wb") as stream:
        np.save(stream, cache, allow_pickle=True)
    train = source / "train_hardneg.txt"
    train.write_text("\n".join(r["im_file"] for r in cache["labels"]) + "\n")
    snapshot = tmp_path / "snapshot.json"
    dump(snapshot, dict(videos=[video], heldout_sha256=["f" * 64], frame_index_base=0))
    audit = tmp_path / "audit.json"
    record = dict(task_id=video["task_id"], source_sha256=digest, cache_image_groups=[str(group)],
                  current_frames_not_in_cache=2, errors={})
    dump(audit, dict(complete=True, coverage_verified=True, all_current_approved_training_sources_covered=True,
                     source_dataset=str(source), source_train_sha256=supplement.sha256(train),
                     source_cache_sha256=supplement.sha256(cache_path), snapshot_sha256=supplement.sha256(snapshot),
                     approved_training_videos=1, videos=[record], heldout_sha256=["f" * 64],
                     current_frames_not_in_cache=2))
    missing = tmp_path / "auditmissing_frames.json"
    plan = supplement.load_plan(audit, snapshot)
    dump(missing, supplement.missing_contract(plan))
    return SimpleNamespace(source=source, video=video, video_path=video_path, snapshot=snapshot, audit=audit,
                           missing=missing, cache=cache_path, train=train, output=tmp_path / "out")


def rebind(frozen, snapshot=None, cache=None):
    audit = json.loads(frozen.audit.read_text())
    if snapshot is not None:
        dump(frozen.snapshot, snapshot)
        audit["snapshot_sha256"] = supplement.sha256(frozen.snapshot)
    if cache is not None:
        with frozen.cache.open("wb") as stream:
            np.save(stream, cache, allow_pickle=True)
        frozen.train.write_text("\n".join(r["im_file"] for r in cache["labels"]) + "\n")
        audit["source_cache_sha256"] = supplement.sha256(frozen.cache)
        audit["source_train_sha256"] = supplement.sha256(frozen.train)
        audit["videos"][0]["cache_image_groups"] = sorted({str(Path(r["im_file"]).parent) for r in cache["labels"]})
    dump(frozen.audit, audit)


def run(frozen, **kwargs):
    return supplement.prepare(frozen.audit, frozen.snapshot, frozen.missing, frozen.output, **kwargs)


@pytest.fixture
def plenty_disk(monkeypatch):
    monkeypatch.setattr(supplement.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * supplement.GIB))


def mock_capture(monkeypatch, frozen, mode="ok"):
    pixels = [np.full((24, 32, 3), (i, 30 + i, 200 - i), dtype=np.uint8) for i in range(6)]
    reads = []

    class Capture:
        def __init__(self, path):
            assert path == str(frozen.video_path.resolve())
            self.index = 0
            self.released = False

        def isOpened(self):
            return mode != "unopened"

        def read(self):
            index = self.index
            self.index += 1
            reads.append(index)
            if index >= (5 if mode == "truncated" else len(pixels)):
                return False, None
            image = pixels[index]
            if mode == "geometry" and index == 3:
                image = image[:12]
            if mode == "changed" and index == 3:
                frozen.video_path.write_bytes(b"changed during decoding")
            return True, image.copy()

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    return pixels, reads


def test_exact_complement_preserves_every_tiny_positive_and_excludes_unknowns(frozen):
    plan = supplement.load_plan(frozen.audit, frozen.snapshot)
    assert plan["missing_frames"] == 2
    assert plan["eligible_positive_frames"] == 1
    assert plan["jobs"][0]["frame_indices"] == [0, 3]
    contract = supplement.missing_contract(plan)
    assert contract["missing_positive_frames"] == 0
    assert contract["decode_source_count"] == 1
    assert all(r["boxes"] == [] and r["reviewed_status"] == "absent" for r in contract["frames"])
    assert {r["frame_index"] for r in contract["frames"]} == {0, 3}
    assert supplement.validate_missing_contract(frozen.missing, plan) == supplement.sha256(frozen.missing)


@pytest.mark.parametrize("style", ["sha12", "sha64", "sha8"])
def test_all_audited_source_group_styles(frozen, style):
    sha = frozen.video["source_sha256"]
    name = {"sha12": "clip_" + sha[:12], "sha64": sha, "sha8": "clip_" + sha[:8]}[style]
    parent = frozen.source / "images" / name
    rebind(frozen, cache=dict(labels=[dict(im_file=str(parent / "000000001.png"))]))
    assert supplement.load_plan(frozen.audit, frozen.snapshot)["jobs"][0]["frame_indices"] == [0, 3]


def test_missing_positive_rejected_not_silently_skipped(frozen):
    parent = Path(json.loads(frozen.audit.read_text())["videos"][0]["cache_image_groups"][0])
    rebind(frozen, cache=dict(labels=[dict(im_file=str(parent / "000000000.jpg"))]))
    with pytest.raises(ValueError, match="positive"):
        run(frozen, dry_run=True)
    assert not frozen.output.exists()


@pytest.mark.parametrize("change", ["uncertain", "unlabelled", "unapproved", "revision", "holdout"])
def test_unsafe_current_snapshot_never_decodes(frozen, change, monkeypatch):
    snapshot = json.loads(frozen.snapshot.read_text())
    video = snapshot["videos"][0]
    if change == "uncertain":
        video["annotations"].append(annotation(0, "uncertain"))
    elif change == "unlabelled":
        video["annotations"] = [r for r in video["annotations"] if r["frame_index"] != 0]
    elif change == "unapproved":
        video["status"] = "submitted"
    elif change == "revision":
        video["revision"] += 1
    else:
        snapshot["heldout_sha256"].append(video["source_sha256"])
    rebind(frozen, snapshot=snapshot)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: pytest.fail("Unsafe input started decoding"))
    with pytest.raises(ValueError):
        run(frozen)
    assert not frozen.output.exists()


@pytest.mark.parametrize("target", ["snapshot", "train", "cache", "audit"])
def test_changed_audited_input_fails_before_output(frozen, target):
    path = getattr(frozen, target)
    if target == "audit":
        audit = json.loads(path.read_text())
        audit["complete"] = False
        dump(path, audit)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        run(frozen, dry_run=True)
    assert not frozen.output.exists()


@pytest.mark.parametrize("change", ["drop", "duplicate", "positive", "revision", "index", "bool_index", "sha", "pin"])
def test_frozen_missing_contract_cannot_expand_or_shrink(frozen, change):
    value = json.loads(frozen.missing.read_text())
    first = value["frames"][0]
    if change == "drop":
        value["frames"].pop()
    elif change == "duplicate":
        value["frames"].append(dict(first))
    elif change == "positive":
        first["boxes"] = [[.1, .1, .1, .1]]
    elif change == "revision":
        first["revision"] += 1
    elif change == "index":
        first["frame_index"] = 1
    elif change == "bool_index":
        first["frame_index"] = False
    elif change == "sha":
        first["source_sha256"] = "e" * 64
    else:
        value["source_snapshot_sha256"] = "e" * 64
    dump(frozen.missing, value)
    with pytest.raises(ValueError, match="Frozen"):
        run(frozen, dry_run=True)
    assert not frozen.output.exists()


def test_dry_run_has_no_output_and_does_not_hash_or_decode_video(frozen, plenty_disk, monkeypatch):
    monkeypatch.setattr(supplement, "check_video", lambda _: pytest.fail("Dry run hashed a raw video"))
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: pytest.fail("Dry run decoded a video"))
    result = run(frozen, dry_run=True, workers=8)
    assert result["preflight_complete"] and not result["complete"]
    assert result["negative_frames"] == 2 and result["positive_frames"] == 0
    assert result["reserved_disk_bytes"] == 40 * supplement.GIB
    assert result["worst_case_png_bytes"] >= 2 * 32 * 24 * 3
    assert not frozen.output.exists()


def test_reserve_plus_worst_case_bound_is_required(frozen, monkeypatch):
    monkeypatch.setattr(supplement.shutil, "disk_usage", lambda _: SimpleNamespace(free=40 * supplement.GIB))
    with pytest.raises(OSError, match="worst-case"):
        run(frozen, dry_run=True)
    assert not frozen.output.exists()


@pytest.mark.parametrize("kwargs", [dict(workers=0), dict(workers=17), dict(reserve_gib=39)])
def test_unsafe_resource_arguments_rejected(frozen, kwargs):
    with pytest.raises(ValueError, match="workers"):
        run(frozen, **kwargs)


def test_existing_output_and_source_child_rejected(frozen, plenty_disk):
    frozen.output.mkdir()
    sentinel = frozen.output / "sentinel"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="fresh"):
        run(frozen)
    assert sentinel.read_text() == "preserve"
    frozen.output = frozen.source / "new-supplement"
    with pytest.raises(ValueError, match="frozen source"):
        run(frozen)
    assert not frozen.output.exists()


def test_source_raw_sha_mismatch_prevents_any_output_or_decode(frozen, plenty_disk, monkeypatch):
    frozen.video_path.write_bytes(b"not the approved video")
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: pytest.fail("Unverified source decoded"))
    with pytest.raises(ValueError, match="Raw source video SHA256"):
        run(frozen)
    assert not frozen.output.exists()


def test_sequential_lossless_native_rgb_export_and_completion_contract(frozen, plenty_disk, monkeypatch):
    pixels, reads = mock_capture(monkeypatch, frozen)
    before = {p: p.read_bytes() for p in (frozen.train, frozen.cache, frozen.snapshot, frozen.audit, frozen.missing)}
    result = run(frozen, workers=8)
    assert result["complete"] and result["raw_video_hashes_verified"]
    manifest = frozen.output / "manifest.jsonl"
    assert result["manifest_sha256"] == supplement.sha256(manifest)
    assert result["source_snapshot_sha256"] == supplement.sha256(frozen.snapshot)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [r["frame_index"] for r in rows] == [0, 3]
    assert reads == list(range(7))
    for row in rows:
        assert row["dataset"] == "approved_negative_supplement" and row["split"] == "train"
        assert row["status"] == "approved" and row["reviewed_status"] == "absent" and row["boxes"] == 0
        assert row["source_sha256"] == row["raw_video_sha256"] == row["group"] == frozen.video["source_sha256"]
        assert row["revision"] == row["approved_revision"] == 7
        assert row["snapshot_sha256"] == result["source_snapshot_sha256"]
        image, label = Path(row["image"]), Path(row["label"])
        assert image.is_absolute() and label.is_absolute()
        assert image.name == f"{row['frame_index']:09d}.png"
        assert label.read_bytes() == b""
        assert np.array_equal(cv2.imread(str(image)), pixels[row["frame_index"]])
    assert all(p.read_bytes() == content for p, content in before.items())
    assert json.loads((frozen.output / "summary.json").read_text())["complete"] is True


@pytest.mark.parametrize("mode", ["truncated", "geometry", "unopened", "changed"])
def test_decoder_or_source_failure_never_marks_complete(frozen, plenty_disk, monkeypatch, mode):
    mock_capture(monkeypatch, frozen, mode)
    with pytest.raises(ValueError):
        run(frozen)
    assert json.loads((frozen.output / "summary.json").read_text())["complete"] is False
    assert not (frozen.output / "manifest.jsonl").exists()


def test_png_write_failure_never_marks_complete(frozen, plenty_disk, monkeypatch):
    mock_capture(monkeypatch, frozen)
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="Cannot write PNG"):
        run(frozen)
    assert json.loads((frozen.output / "summary.json").read_text())["complete"] is False
