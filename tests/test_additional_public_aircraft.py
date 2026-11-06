"""CPU-only: python3 -m pytest --noconftest --import-mode=importlib -o addopts='' this_file."""

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.anti_uav import prepare_additional_public_aircraft as ingest


def annotation(category=3, bbox=(1, 2, 0.25, 0.5), **flags):
    return dict(category_id=category, bbox=list(bbox), iscrowd=0, **flags)


def output_dirs(tmp_path):
    output = tmp_path / "output"
    for split in ingest.SPLITS:
        for kind in ("images", "labels"):
            (output / kind / split).mkdir(parents=True)
    return output


@pytest.fixture
def ample_disk(monkeypatch):
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * ingest.GIB))


def test_all_aircraft_and_subpixel_boxes():
    boxes, names = ingest.aod_boxes([annotation(i) for i in (1, 3, 4)], ingest.AOD_CATEGORIES, 20, 16)
    assert names == ["airplane", "drone", "helicopter"]
    assert boxes == [(1, 2, 1.25, 2.5)] * 3
    assert list(map(float, ingest.yolo_line(boxes[0], 20, 16).split()[1:])) == pytest.approx(
        [1.125 / 20, 2.25 / 16, 0.25 / 20, 0.5 / 16])


@pytest.mark.parametrize("bad", [annotation(2), annotation(0), annotation(99),
    annotation(ignore=1), annotation(ignored=1), annotation(difficult=1),
    annotation(bbox=(1, 2, 0, 1)), annotation(bbox=(1, 2, float("nan"), 1)),
    annotation(bbox=(21, 2, 2, 1)), dict(category_id=3, bbox=[1, 2, 1, 1])])
def test_bad_object_excludes_whole_frame_not_partial_positive_or_negative(bad):
    with pytest.raises(ingest.UnsafeData):
        ingest.aod_boxes([annotation(), bad], ingest.AOD_CATEGORIES, 20, 16)


def test_empty_annotations_never_negative():
    with pytest.raises(ingest.UnsafeData, match="not_negative"):
        ingest.aod_boxes([], ingest.AOD_CATEGORIES, 20, 16)


def test_truncated_aircraft_clipped_without_losing_tiny_fractional_intersection():
    assert ingest.box_xyxy((-2, -3, 21, 17), 20, 16) == (0, 0, 20, 16)
    assert ingest.box_xyxy((-5, 2, 0.125, 2.25), 20, 16) == (0, 2, .125, 2.25)
    boxes, _ = ingest.aod_boxes([annotation(1, (-5, 2, 5.125, .25))], ingest.AOD_CATEGORIES, 20, 16)
    assert boxes == [(0, 2, .125, 2.25)]
    with pytest.raises(ingest.UnsafeData, match="intersection"):
        ingest.box_xyxy((-5, 1, -2, 2), 20, 16)


def test_refined_mot_xywh_no_shift_and_whole_interpolation_exclusion():
    text = ("1,1,-2,3,2.25,0.5,1.0,-1,-1,-1\n"
            "2,1,2,3,1,1,1.0,-1,-1,-1\n"
            "2,2,5,6,1,1,0.5,-1,-1,-1\n"
            "4,1,19,15,2,2,1.0,-1,-1,-1\n")
    frames, bad, clipped = ingest.parse_purdue_refined(text, 4, 20, 16, "official.zip::refined")
    assert frames == [[(0, 3, .25, 3.5)], None, None, [(19, 15, 20, 16)]]
    assert clipped == [1, 0, 0, 1]
    assert [r["frame_index"] for r in bad] == [1, 2]
    assert "interpolated" in bad[0]["reason"]
    with pytest.raises(ingest.UnsafeData, match="outside_matched_video"):
        ingest.parse_purdue_refined("5,1,0,0,1,1,1,-1,-1,-1", 4, 20, 16, "bad")


def test_refined_malformed_object_quarantines_other_objects():
    text = "1,1,1,2,3,4,1,-1,-1,-1\n1,2,2,3,nan,4,1,-1,-1,-1\n"
    frames, bad, clipped = ingest.parse_purdue_refined(text, 1, 20, 16, "official")
    assert frames == [None] and clipped == [0] and len(bad) == 1


def test_refined_missing_requires_explicit_fallback_and_corrupt_never_falls_back(tmp_path):
    with pytest.raises(ingest.UnsafeData, match="explicitly"):
        ingest.plan_purdue(tmp_path, 8, [], [])
    archive = tmp_path / "corrupt.zip"
    archive.write_bytes(b"not official refined ground truth")
    with pytest.raises(ingest.UnsafeData, match="no_silent_fallback"):
        ingest.plan_purdue(tmp_path, 8, [], [], archive, allow_original=True)


def test_exact_source_groups_no_class_filename_heuristic():
    yt = dict(file_name="yt-0BFf4YWpzHM-0531_jpg.rf.abc.jpg", extra=dict(user_tags=["youtube:0BFf4YWpzHM"]))
    assert ingest.aod_source_group(yt) == "youtube:0BFf4YWpzHM"
    assert ingest.aod_source_group(dict(file_name="20190925_101846_1_1_000_jpg.rf.abc.jpg")) == ingest.aod_source_group(
        dict(file_name="20190925_101846_1_2_700_jpg.rf.def.jpg"))
    assert ingest.aod_source_group(dict(file_name="pexels-pixabay-62678_jpg.rf.abc.jpg")).startswith("photo:")
    assert ingest.aod_source_group(dict(file_name="bird-AAA-unsplash_jpg.rf.abc.jpg")).startswith("photo:")
    yt["extra"]["user_tags"] = ["youtube:wrong"]
    with pytest.raises(ingest.UnsafeData, match="disagreement"):
        ingest.aod_source_group(yt)
    for name in ("V_HELICOPTER_008232_073_png_jpg.rf.abc.jpg", "-_jpeg_jpg.rf.abc.jpg", "drone_001.jpg", "IR_001.jpg"):
        with pytest.raises(ingest.UnsafeData):
            ingest.aod_source_group(dict(file_name=name))


def test_cross_split_groups_and_transitive_duplicate_quarantine():
    def p(group, split, digest):
        return dict(source=group + digest, group=group, split=split, raw_image_sha256=digest)
    rows = [p("a", "train", "1"), p("a", "test", "2"), p("b", "train", "1"),
            p("b", "train", "3"), p("c", "train", "3"), p("safe", "val", "4")]
    good, bad = ingest.purge_split_conflicts(rows)
    assert [p["group"] for p in good] == ["safe"]
    assert len(bad) == 5


def test_purdue_coordinate_order_origin_and_first_frame_alignment(tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("time_layer: 1 detections: (0, 8, 1, 9), (3, 2, 5, 6), \n"
                  "time_layer: 2 detections: \n"
                  "time_layer: 3 detections: (1, 2, 2, 3), unknown\n")
    frames, bad = ingest.parse_purdue_gt(gt, 3, 20, 16)
    assert frames == [[(8, 0, 9, 1), (2, 3, 6, 5)], None, None]
    assert [b["frame_index"] for b in bad] == [1, 2]
    assert [b["original_frame_number"] for b in bad] == [2, 3]
    with pytest.raises(ingest.UnsafeData, match="alignment"):
        ingest.parse_purdue_gt(gt, 4, 20, 16)


@pytest.mark.parametrize("text", ["time_layer: 0 detections: ", "time_layer: 2 detections: ",
                                  "time_layer: 1 detections: \ntime_layer: 1 detections: "])
def test_purdue_missing_or_duplicate_ids_block_clip(tmp_path, text):
    gt = tmp_path / "gt.txt"
    gt.write_text(text + "\n")
    with pytest.raises(ingest.UnsafeData, match="frame_id"):
        ingest.parse_purdue_gt(gt, len(text.splitlines()), 20, 16)


def test_archive_existing_file_crc_size_and_hash(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_bytes(b"verified")
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("raw.txt", b"verified")
    member = ingest.verified_member(archive, "raw.txt", raw)
    assert member["sha256"] == hashlib.sha256(b"verified").hexdigest()
    raw.write_bytes(b"modified")
    with pytest.raises(ingest.UnsafeData, match="archive_member_mismatch"):
        ingest.verified_member(archive, "raw.txt", raw)


def test_stable_split_is_content_based_and_matches_existing_contract():
    hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(100)]
    splits = [ingest.stable_split(h) for h in hashes]
    assert set(splits) == {"train", "val"}
    assert splits == ["val" if int(hashlib.sha256(("public-uav-research-v1:" + h).encode()).hexdigest(), 16)
                      < (1 << 256) // 5 else "train" for h in hashes]
    with pytest.raises(ingest.UnsafeData):
        ingest.stable_split("path/not/content")


def make_plan(tmp_path, frames=None):
    source = tmp_path / "visible.png"
    pixels = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
    assert cv2.imwrite(str(source), pixels)
    plan = dict(dataset="aod4", source=source, split="test", group="photo:abc", width=20, height=16,
                aircraft_types=["airplane"], raw_image_sha256=ingest.sha256(source), xyxy=[(0, 0, .25, .5)])
    if frames is not None:
        plan.update(dataset="purdue_uav", frames=frames, frame_count=len(frames), annotation=source,
                    annotation_sha256=ingest.sha256(source), raw_video_sha256=ingest.sha256(source))
        plan.pop("xyxy")
        plan.pop("raw_image_sha256")
    return plan, pixels


def test_aod_symlink_and_positive_manifest_contract(tmp_path, ample_disk):
    plan, pixels = make_plan(tmp_path)
    out = output_dirs(tmp_path)
    row = ingest.emit_frame(plan, out, ingest.DiskGuard(out), plan["xyxy"])
    assert Path(row["image"]).is_symlink()
    assert Path(row["image"]).resolve() == plan["source"]
    assert np.array_equal(cv2.imread(row["image"]), pixels)
    assert row["split"] == "test" and type(row["boxes"]) is int and row["boxes"] == 1
    assert {"dataset", "split", "group", "source", "image", "label", "boxes", "width", "height"} <= row.keys()
    assert ingest.audit_manifest([row])["negative_frames"] == 0
    with pytest.raises(ingest.UnsafeData, match="empty_labels"):
        ingest.emit_frame(plan, out, ingest.DiskGuard(out), [])


class SequentialCapture:
    def __init__(self, frames):
        self.frames, self.index, self.released = frames, 0, False

    def isOpened(self):
        return True

    def read(self):
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


def test_purdue_sequential_decode_only_positive_jpeg95_and_provenance(tmp_path, ample_disk, monkeypatch):
    plan, pixels = make_plan(tmp_path, [[(0, 0, 1, 1)], None, [(8, 3, 9, 4)]])
    cap = SequentialCapture([pixels, pixels * 0, pixels])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: cap)
    original_encode, qualities = cv2.imencode, []

    def encode(ext, frame, params):
        qualities.append(params)
        return original_encode(ext, frame, params)

    monkeypatch.setattr(cv2, "imencode", encode)
    out = output_dirs(tmp_path)
    rows, audit = ingest.export_purdue(plan, out, ingest.DiskGuard(out))
    assert [r["frame_index"] for r in rows] == [0, 2]
    assert [r["gt_time_layer"] for r in rows] == [1, 3]
    assert qualities == [[cv2.IMWRITE_JPEG_QUALITY, 95]] * 2
    assert all(r["coordinate_convention"].startswith("zero_based_ymin") for r in rows)
    assert all(not Path(r["image"]).is_symlink() for r in rows)
    assert audit["alignment_verified"] and audit["sequential_decoded_frames"] == 3 and cap.released


def test_short_decode_fails_closed(tmp_path, ample_disk, monkeypatch):
    plan, pixels = make_plan(tmp_path, [[(0, 0, 1, 1)], [(0, 0, 1, 1)]])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: SequentialCapture([pixels]))
    out = output_dirs(tmp_path)
    with pytest.raises(ingest.UnsafeData, match="incomplete_decode"):
        ingest.export_purdue(plan, out, ingest.DiskGuard(out))
    assert not (out / "manifest.jsonl").exists()


def test_refined_export_verifies_archive_not_pseudo_annotation_path(tmp_path, ample_disk, monkeypatch):
    plan, pixels = make_plan(tmp_path, [[(0, 0, .25, 1)]])
    archive = tmp_path / "refined.zip"
    archive.write_bytes(b"archive-content-identity")
    plan.update(annotation=str(archive) + "::refined_gt/Clip_1_refined.txt", annotation_archive=archive,
                annotation_archive_sha256=ingest.sha256(archive), annotation_release="refined_v2",
                coordinate_convention="zero_based_xywh_size_is_difference", clipped_boxes_by_frame=[1])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: SequentialCapture([pixels]))
    out = output_dirs(tmp_path)
    rows, audit = ingest.export_purdue(plan, out, ingest.DiskGuard(out))
    assert rows[0]["clipped_boxes"] == 1
    assert rows[0]["annotation_release"] == audit["annotation_release"] == "refined_v2"
    assert rows[0]["coordinate_convention"] == "zero_based_xywh_size_is_difference"


def test_blocked_aod_never_starts_video_export(tmp_path, ample_disk, monkeypatch):
    monkeypatch.setattr(ingest, "plan_aod", lambda *args: [])
    monkeypatch.setattr(ingest, "plan_purdue", lambda *args: pytest.fail("must stop before video planning"))
    out = tmp_path / "blocked"
    with pytest.raises(ingest.UnsafeData, match="no_leak_free"):
        ingest.prepare(tmp_path / "raw", out, research_authorized=True)
    assert json.loads((out / "summary.json").read_text())["complete"] is False
    assert not (out / "manifest.jsonl").exists()


def test_disk_guard_reserve_and_no_overwrite(tmp_path, ample_disk, monkeypatch):
    guard = ingest.DiskGuard(tmp_path)
    path = tmp_path / "new"
    guard.write(path, b"abc")
    with pytest.raises(FileExistsError):
        guard.write(path, b"def")
    with pytest.raises(ValueError):
        ingest.DiskGuard(tmp_path, 39)
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _: SimpleNamespace(free=40 * ingest.GIB))
    with pytest.raises(ingest.UnsafeData, match="reserve"):
        guard.write(tmp_path / "never", b"x")
    assert not (tmp_path / "never").exists()


def test_prepare_complete_hash_and_failed_summary(tmp_path, ample_disk, monkeypatch):
    plan, _ = make_plan(tmp_path)
    monkeypatch.setattr(ingest, "plan_aod", lambda *args: [dict(plan)])
    out = tmp_path / "new"
    summary = ingest.prepare(tmp_path / "raw", out, datasets=["aod4"], research_authorized=True)
    assert summary["complete"] is True
    assert summary["manifest_sha256"] == ingest.sha256(out / "manifest.jsonl")
    assert json.loads((out / "summary.json").read_text())["complete"] is True
    assert "0: drone" in (out / "data.yaml").read_text()
    with pytest.raises(FileExistsError):
        ingest.prepare(tmp_path / "raw", out, datasets=["aod4"], research_authorized=True)

    def fail(*args):
        raise ingest.UnsafeData("unverified_origin")

    monkeypatch.setattr(ingest, "plan_aod", fail)
    failed = tmp_path / "failed"
    with pytest.raises(ingest.UnsafeData, match="origin"):
        ingest.prepare(tmp_path / "raw", failed, datasets=["aod4"], research_authorized=True)
    assert json.loads((failed / "summary.json").read_text())["complete"] is False
    assert not (failed / "manifest.jsonl").exists()


def test_auth_and_raw_overlap_rejected_before_writes(tmp_path):
    with pytest.raises(ingest.UnsafeData, match="research-authorized"):
        ingest.prepare(tmp_path, tmp_path / "output")
    with pytest.raises(ingest.UnsafeData, match="overlap"):
        ingest.prepare(tmp_path, tmp_path / "output", research_authorized=True)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name", ["../escape.jpg", "/absolute.jpg", "nested/file.jpg", "a\\b.jpg", None])
def test_source_path_traversal_rejected(tmp_path, name):
    with pytest.raises(ingest.UnsafeData):
        ingest.safe_child(tmp_path, name)


def make_aod_raw(tmp_path, monkeypatch, splits):
    raw = tmp_path / "raw"
    root = raw / "aod4/extracted/aod4_inner/AOD 4"
    archive = raw / "aod4/extracted/AOD 4 Dataset for Air Borne Object Detection/AOD 4.zip"
    archive.parent.mkdir(parents=True)
    members = []
    for split, entries in splits.items():
        base = root / "Images" / split
        base.mkdir(parents=True)
        images, annotations = [], []
        for index, (name, anns, shade) in enumerate(entries):
            source = base / name
            assert cv2.imwrite(str(source), np.full((16, 20, 3), shade, dtype=np.uint8))
            members.append(source)
            images.append(dict(id=index, file_name=name, width=20, height=16))
            for ann in anns:
                annotations.append(dict(ann, image_id=index, id=len(annotations)))
        path = root / "Annotations/COCO Annotation format" / split / "_annotations.coco.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(images=images, annotations=annotations,
                                       categories=[dict(id=i, name=n) for i, n in ingest.AOD_CATEGORIES.items()])))
        members.append(path)
    with zipfile.ZipFile(archive, "w") as z:
        for member in members:
            z.write(member, "AOD 4/" + str(member.relative_to(root)))
    monkeypatch.setattr(ingest, "AOD_COUNTS", {s: len(entries) for s, entries in splits.items()})
    return raw


def test_actual_aod_planner_and_cli_contract_end_to_end(tmp_path, ample_disk, monkeypatch):
    splits = {s: [(f"pilot-{s}-unsplash_jpg.rf.abc.jpg", [annotation(category)], shade)]
              for s, category, shade in (("train", 1, 30), ("valid", 3, 60), ("test", 4, 90))}
    raw = make_aod_raw(tmp_path, monkeypatch, splits)
    output = tmp_path / "converted"
    assert ingest.main(["--raw-root", str(raw), "--output", str(output), "--datasets", "aod4",
                        "--research-authorized", "--workers", "8"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert summary["complete"] is True and summary["manifest_sha256"] == ingest.sha256(output / "manifest.jsonl")
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert summary["aircraft_type_boxes"] == {"airplane": 1, "drone": 1, "helicopter": 1}
    assert all(Path(r["image"]).is_symlink() for r in rows)


def test_aod_holdout_bird_and_unknown_duplicate_are_audited_before_filtering(tmp_path, monkeypatch):
    splits = dict(train=[("yt-0BFf4YWpzHM-0001_jpg.rf.a.jpg", [annotation()], 40),
                         ("pilot-a-unsplash_jpg.rf.a.jpg", [annotation()], 50),
                         ("pilot-safe-unsplash_jpg.rf.a.jpg", [annotation(4)], 60)],
                  valid=[("pilot-val-unsplash_jpg.rf.a.jpg", [annotation(1)], 70)],
                  test=[("yt-0BFf4YWpzHM-0002_jpg.rf.b.jpg", [annotation(2)], 80),
                        ("-_jpeg_jpg.rf.b.jpg", [], 50)])
    raw = make_aod_raw(tmp_path, monkeypatch, splits)
    bad, provenance = [], []
    plans = ingest.plan_aod(raw, 8, bad, provenance)
    assert {p["source"].name for p in plans} == {"pilot-safe-unsplash_jpg.rf.a.jpg", "pilot-val-unsplash_jpg.rf.a.jpg"}
    assert any(b["reason"] == "cross_split_source_or_duplicate_group" and "pilot-a" in b["source"] for b in bad)
    assert len(provenance) == 4


def aod_train_only_splits():
    return dict(
        train=[("yt-0BFf4YWpzHM-0001_jpg.rf.a.jpg", [annotation(1, (-1, 2, 1.25, .5))], 40),
               ("yt-5KOIhxSH7lw-0001_jpg.rf.a.jpg", [annotation(3)], 50),
               ("yt-vpQEQ81cfoA-0001_jpg.rf.a.jpg", [annotation(4)], 60),
               ("V_AIRPLANE_01111_001_png_jpg.rf.a.jpg", [annotation(1)], 70),
               ("unknown-drone_jpg.rf.a.jpg", [annotation(3)], 80),
               ("pilot-bird-unsplash_jpg.rf.a.jpg", [annotation(2)], 90),
               ("pilot-mixed-unsplash_jpg.rf.a.jpg", [annotation(1), annotation(2)], 100),
               ("pilot-empty-unsplash_jpg.rf.a.jpg", [], 110),
               ("pilot-ignored-unsplash_jpg.rf.a.jpg", [annotation(4, ignore=1)], 120)],
        valid=[("yt-0BFf4YWpzHM-0002_jpg.rf.b.jpg", [annotation(1)], 130),
               ("pilot-val-unsplash_jpg.rf.b.jpg", [annotation(4)], 40)],
        test=[("yt-5KOIhxSH7lw-0002_jpg.rf.b.jpg", [annotation(3)], 140),
              ("V_DRONE_002112_206_png_jpg.rf.b.jpg", [annotation(3)], 150)])


def test_aod_train_only_retains_official_train_without_reassigning_holdouts(tmp_path, monkeypatch):
    raw = make_aod_raw(tmp_path, monkeypatch, aod_train_only_splits())
    bad, provenance, audit = [], [], {}
    plans = ingest.plan_aod(raw, 8, bad, provenance, train_only=True, audit=audit)
    assert len(plans) == 3
    assert {p["split"] for p in plans} == {"train"}
    assert {p["official_split"] for p in plans} == {"train"}
    assert all(p["evaluation_allowed"] is False for p in plans)
    assert all(p["source"].parent.name == "train" for p in plans)
    assert {t for p in plans for t in p["aircraft_types"]} == {"airplane", "drone", "helicopter"}
    assert sum(p["clipped_boxes"] for p in plans) == 1
    assert all("V_" not in p["source"].name for p in plans)
    assert len(audit["unused_holdout_images"]) == 4
    assert {r["official_split"] for r in audit["unused_holdout_images"]} == {"valid", "test"}
    assert all(r["training_allowed"] is False and r["evaluation_allowed"] is False for r in audit["unused_holdout_images"])
    assert audit["retained_train_shared_holdout_group_frames"] == 2
    assert audit["retained_train_shared_holdout_hash_frames"] == 1
    assert sum(r["reason"] == "aod_train_only_official_holdout_unused" for r in bad) == 4
    assert sum(r["reason"] == "bird_or_unknown_class_excluded_whole_frame" for r in bad) == 2


def test_aod_explicit_train_only_does_not_change_default_quarantine(tmp_path, monkeypatch):
    raw = make_aod_raw(tmp_path, monkeypatch, aod_train_only_splits())
    strict = ingest.plan_aod(raw, 8, [], [])
    train_only = ingest.plan_aod(raw, 8, [], [], train_only=True)
    strict_train = {p["source"].name for p in strict if p["split"] == "train"}
    assert strict_train == {"yt-vpQEQ81cfoA-0001_jpg.rf.a.jpg"}
    assert len(train_only) == 3 and all(p["split"] == "train" for p in train_only)
    assert strict == ingest.plan_aod(raw, 8, [], [], train_only=False)
    assert all("aod_split_policy" not in p for p in strict)


def test_aod_train_only_cli_protects_holdout_metadata_and_disclaims_evaluation(tmp_path, ample_disk, monkeypatch):
    raw = make_aod_raw(tmp_path, monkeypatch, aod_train_only_splits())
    out = tmp_path / "train_only_export"
    assert ingest.main(["--raw-root", str(raw), "--output", str(out), "--datasets", "aod4",
                        "--aod-train-only", "--research-authorized", "--workers", "8"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert summary["complete"] is True
    assert summary["manifest_sha256"] == ingest.sha256(out / "manifest.jsonl")
    assert summary["counts"]["aod4"] == {
        "train": {"images": 3, "boxes": 3}, "val": {"images": 0, "boxes": 0}, "test": {"images": 0, "boxes": 0}}
    policy = summary["aod_policy"]
    assert policy["mode"] == "official_train_only" and policy["source_group_quarantine"] is False
    assert policy["official_split_reassigned"] is False
    assert policy["official_video_split_independence_claimed"] is False
    assert policy["aod_performance_claim_allowed"] is False and policy["evaluation_allowed"] is False
    assert summary["leak_audit"]["scope"] == "exported_manifest_only_not_upstream_official_splits"
    protected = policy["protected_holdout_manifest"]
    assert protected["images"] == 4 and protected["official_split_counts"] == {"valid": 2, "test": 2}
    assert ingest.sha256(protected["path"]) == protected["sha256"]
    protected_rows = [json.loads(line) for line in Path(protected["path"]).read_text().splitlines()]
    assert all(row["raw_image_sha256"] == ingest.sha256(row["source"]) for row in protected_rows)
    for split in ("val", "test"):
        assert not list((out / "images" / split).iterdir())
        assert not list((out / "labels" / split).iterdir())
    assert all(Path(row["image"]).is_symlink() for row in rows)
    assert all(row["evaluation_allowed"] is False and row["official_split"] == "train" for row in rows)


@pytest.mark.parametrize("change", [dict(split="val"), dict(official_split="test"), dict(evaluation_allowed=True)])
def test_aod_train_only_manifest_rejects_holdout_or_evaluation_permission(change):
    row = dict(dataset="aod4", split="train", official_split="train", evaluation_allowed=False)
    row.update(change)
    with pytest.raises(ingest.UnsafeData, match="no_evaluation"):
        ingest.audit_manifest([row], aod_train_only=True)


def test_aod_train_only_flag_requires_aod_dataset_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="requires aod4"):
        ingest.prepare(tmp_path / "raw", tmp_path / "never", datasets=["purdue_uav"],
                       aod_train_only=True, research_authorized=True)
    assert not (tmp_path / "never").exists()
