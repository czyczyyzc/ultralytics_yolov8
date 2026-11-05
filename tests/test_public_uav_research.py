"""Torch-free tests; run pytest with --noconftest --import-mode=importlib."""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


from scripts.anti_uav import prepare_public_uav_research as ingest


def element(kind, payload=b"", children=()):
    def as_array():
        return np.frombuffer(payload, dtype={5: "<i4", 6: "<u4", 9: "<f8", 13: "<u8"}[kind])
    return SimpleNamespace(type_id=kind, payload=payload, children=children, as_array=as_array)


def numeric(kind, values):
    dtype = {5: "<i4", 6: "<u4", 9: "<f8", 13: "<u8"}[kind]
    return element(kind, np.asarray(values, dtype=dtype).tobytes())


def matrix(kind, dims, data):
    return element(14, children=(numeric(6, [kind, 0]), numeric(5, dims), element(1), *data))


def double(boxes=()):
    dims = (len(boxes), 4) if boxes else (0, 0)
    values = np.asarray(boxes).flatten(order="F") if boxes else []
    return matrix(6, dims, [numeric(9, values)])


def tree(frames, fps=30):
    cells = []
    for index, boxes in enumerate(frames):
        time = matrix(13, (6, 1), [numeric(6, [3707764736, 2, 1, 1, 8 + index, 1])])
        cells.extend([time, double(), double(), double(boxes), double()])
    names = b"".join(s.encode().ljust(11, b"\0") for s in ingest.FIELDS)
    table = matrix(2, (len(frames), 1), [numeric(5, [11]), element(1, names), *cells])
    times = matrix(6, (len(frames), 1), [numeric(9, np.arange(len(frames)) * 1000 / fps)])
    # 0.1.0's global size-based walker misreads this string as an empty frame.
    misleading_source = matrix(4, (1, 31), [element(16, b"C:\\Data\\Video_V\\V_DRONE_001.mp4")])
    opaque = element(14, children=(numeric(6, [17, 0]), element(1), element(1), element(1), table, times))
    return [element(14, children=(opaque, misleading_source))]


def make_dut(tmp_path, objects=None, suffix=".png"):
    image = tmp_path / ("00001" + suffix)
    pixels = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
    assert cv2.imwrite(str(image), pixels)
    objects = objects if objects is not None else [("UAV", "0", (2, 3, 2.5, 4))]
    obj = "".join("<object><name>" + name + "</name><difficult>" + difficult + "</difficult><bndbox>" +
                  "".join(f"<{k}>{v}</{k}>" for k, v in zip(("xmin", "ymin", "xmax", "ymax"), box)) +
                  "</bndbox></object>" for name, difficult, box in objects)
    xml = tmp_path / "00001.xml"
    xml.write_text(f"<annotation><filename>{image.name}</filename><size><width>20</width><height>16</height></size>{obj}</annotation>")
    return xml, image, pixels


def output_dirs(tmp_path):
    output = tmp_path / "out"
    for split in ingest.SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    return output


def test_sub_two_pixel_and_multiple_dut_boxes(tmp_path):
    xml, image, _ = make_dut(tmp_path, [("UAV", "0", (2, 3, 2.5, 4)), ("UAV", "0", (1, 1, 21, 17))])
    boxes, width, height, clipped = ingest.parse_dut_xml(xml, image)
    assert boxes == [(2, 3, 2.5, 4), (1, 1, 20, 16)]
    assert (width, height, clipped) == (20, 16, 1)
    values = [float(x) for x in ingest.yolo_line(boxes[0], width, height).split()[1:]]
    assert values == pytest.approx([2.25 / 20, 3.5 / 16, 0.5 / 20, 1 / 16])


@pytest.mark.parametrize("obj,reason", [
    (("AIRPLANE", "0", (1, 1, 3, 3)), "unknown_object_class"),
    (("UAV", "1", (1, 1, 3, 3)), "difficult_or_missing_flag"),
    (("UAV", "0", (1, 1, 1, 3)), "degenerate_box"),
    (("UAV", "0", (1, 1, float("nan"), 3)), "nonfinite_box"),
])
def test_bad_object_quarantines_entire_frame(tmp_path, obj, reason):
    xml, image, _ = make_dut(tmp_path, [("UAV", "0", (1, 1, 3, 3)), obj])
    out = output_dirs(tmp_path)
    row, bad = ingest.export_dut(("train", xml, image), out, ingest.DiskGuard(out))
    assert row is None and bad["reason"] == reason
    assert not list((out / "images/train").iterdir())
    assert not list((out / "labels/train").iterdir())


def test_dut_symlink_extension_pixels_and_absolute_contract(tmp_path):
    xml, image, pixels = make_dut(tmp_path)
    out = output_dirs(tmp_path)
    row, bad = ingest.export_dut(("test", xml, image), out, ingest.DiskGuard(out))
    assert bad is None and row["split"] == "test" and row["boxes"] == 1
    assert {"dataset", "split", "group", "source", "image", "label", "boxes"} <= row.keys()
    assert Path(row["image"]).is_symlink() and Path(row["image"]).suffix == ".png"
    assert Path(row["image"]).resolve() == image.resolve()
    assert all(Path(row[k]).is_absolute() for k in ("image", "label", "source"))
    assert np.array_equal(cv2.imread(row["image"]), pixels)


def test_matlab_one_based_xywh_pixels_and_tiny_box():
    box, _ = ingest.matlab_xywh((1, 1, 1, 1), 20, 16)
    assert box == (0, 0, 1, 1)
    box, _ = ingest.matlab_xywh((3.25, 4.5, 0.25, 0.5), 20, 16)
    assert box == (2.25, 3.5, 2.5, 4)


def test_dut_zero_origin_is_not_shifted(tmp_path):
    xml, image, _ = make_dut(tmp_path, [("UAV", "0", (0, 0, 1, 1))])
    boxes, _, _, clipped = ingest.parse_dut_xml(xml, image)
    assert boxes == [(0, 0, 1, 1)] and clipped == 0


def test_named_table_ignores_metadata_preserves_frame_one_and_all_boxes():
    expected = [((1, 2, 3, 4),), ((2, 3, 4, 5), (6, 7, 8, 9)), None]
    frames, times = ingest.parse_halmstad_tree(tree([expected[0], expected[1], ()]), 3, 30)
    assert frames == expected
    assert times == pytest.approx([0, 1000 / 30, 2000 / 30])


def test_mcos_rejects_frame_count_timestamp_and_unknown_class():
    with pytest.raises(ingest.BadRecord, match="frame_count"):
        ingest.parse_halmstad_tree(tree([((1, 2, 3, 4),)]), 2, 30)
    with pytest.raises(ingest.BadRecord, match="timestamps"):
        ingest.parse_halmstad_tree(tree([(), (), ()], fps=25), 3, 30)
    tops = tree([((1, 2, 3, 4),)])
    table = tops[0].children[0].children[4]
    cells = list(table.children)
    cells[6] = double(((1, 2, 3, 4),))
    table.children = tuple(cells)
    with pytest.raises(ingest.BadRecord, match="non_drone"):
        ingest.parse_halmstad_tree(tops, 1, 30)


def test_split_deterministic_and_no_group_video_leak():
    hashes = [__import__("hashlib").sha256(str(i).encode()).hexdigest() for i in range(100)]
    assert {ingest.stable_split(h) for h in hashes} == {"train", "val"}
    assert [ingest.stable_split(h) for h in hashes] == [ingest.stable_split(h) for h in hashes]
    rows = [dict(dataset="halmstad", split="train", group="g", raw_video_sha256=hashes[0], image="a", label="a.txt", boxes=1),
            dict(dataset="halmstad", split="val", group="other", raw_video_sha256=hashes[0], image="b", label="b.txt", boxes=1)]
    with pytest.raises(ingest.BadRecord, match="leak"):
        ingest.audit_manifest(rows)
    rows[1]["split"] = "train"
    assert ingest.audit_manifest(rows)["cross_split_group_overlap"] == 0
    rows[1]["boxes"] = 0
    with pytest.raises(ingest.BadRecord, match="negative"):
        ingest.audit_manifest(rows)


def test_archive_hash_completeness_and_corrupt_reused_member(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a").write_bytes(b"correct")
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a", b"correct")
    result = ingest.audit_archive(archive, raw, lambda _: True)
    assert result["complete"] and result["verified_members"] == 1
    assert result["sha256"] == ingest.sha256(archive)
    (raw / "a").write_bytes(b"corrupt")
    with pytest.raises(ingest.BadRecord, match="crc_mismatch"):
        ingest.audit_archive(archive, raw, lambda _: True)


def test_paths_existing_output_raw_nesting_and_disk_margin(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        ingest.check_paths(raw, raw / "new")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        ingest.check_paths(raw, existing)
    disk = ingest.DiskGuard(tmp_path)
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _: SimpleNamespace(total=100 * ingest.GIB, free=disk.margin + 9))
    with pytest.raises(OSError, match="disk_margin"):
        disk.check(10)
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _: SimpleNamespace(total=16 * 1024 * ingest.GIB, free=221 * ingest.GIB))
    shared_disk = ingest.DiskGuard(tmp_path)
    assert shared_disk.margin == 40 * ingest.GIB
    shared_disk.check(100 * ingest.GIB)


@pytest.mark.parametrize("truncate", [False, True])
def test_one_pass_video_alignment_jpeg95_unlabeled_never_negative(tmp_path, monkeypatch, truncate):
    output = output_dirs(tmp_path)
    video = tmp_path / "V_DRONE_001.mp4"
    video.write_bytes(b"video")
    annotation = tmp_path / "V_DRONE_001_LABELS.mat"
    annotation.write_bytes(b"annotations")
    pixels = [np.full((16, 20, 3), value, np.uint8) for value in (20, 80, 160)]
    reads, encodes = [], []

    class Capture:
        def __init__(self, path):
            self.index = 0
        def isOpened(self):
            return True
        def read(self):
            reads.append(self.index)
            self.index += 1
            if truncate and self.index == 3:
                return False, None
            return (True, pixels[self.index - 1]) if self.index <= 3 else (False, None)
        def release(self):
            pass

    real_encode = cv2.imencode

    def encode(ext, pixels, params):
        encodes.append(params)
        return real_encode(ext, pixels, params)

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(cv2, "imencode", encode)
    monkeypatch.setattr(ingest, "load_halmstad", lambda *args: ([((1, 1, 1, 1),), None, ((3, 4, 2, 2), (8, 9, 1, 1))], [0, 33, 66]))
    digest = ingest.sha256(video)
    plan = (video, annotation, digest, 3, 30, 20, 16)
    rows, bad = ingest.export_halmstad(plan, output, ingest.DiskGuard(output))
    if truncate:
        assert rows == [] and "quarantined_whole_clip" in bad[0]["reason"]
        assert not list((output / "images").rglob("*.jpg"))
        assert not list((output / "labels").rglob("*.txt"))
        return
    assert [r["original_frame_number"] for r in rows] == [1, 3]
    assert [r["boxes"] for r in rows] == [1, 2]
    assert len(bad) == 1 and bad[0]["reason"] == "unlabeled_frame_not_negative"
    assert reads == [0, 1, 2, 3] and encodes == [[cv2.IMWRITE_JPEG_QUALITY, 95]] * 2
    assert [int(cv2.imread(r["image"])[0, 0, 0]) for r in rows] == [20, 160]
    assert all(r["raw_video_sha256"] == digest and r["split"] == ingest.stable_split(digest) for r in rows)
    assert len({r["group"] for r in rows}) == 1


def test_complete_dut_run_keeps_official_splits_and_refuses_rerun(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    base = raw / "dut_anti_uav"
    (base / "archives").mkdir(parents=True)
    for split in ingest.SPLITS:
        image_dir = base / "extracted" / split / "img"
        xml_dir = base / "extracted" / split / "xml"
        image_dir.mkdir(parents=True)
        xml_dir.mkdir()
        xml, image, _ = make_dut(image_dir)
        xml.rename(xml_dir / xml.name)
        with zipfile.ZipFile(base / "archives" / (split + ".zip"), "w") as z:
            z.write(image, split + "/img/" + image.name)
            z.write(xml_dir / xml.name, split + "/xml/" + xml.name)
    monkeypatch.setattr(ingest, "DUT_COUNTS", dict.fromkeys(ingest.SPLITS, 1))
    output = tmp_path / "new_dataset"
    summary = ingest.prepare(raw, output, datasets=["dut"], workers=1)
    assert summary["complete"] and summary["research_only"]
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert {row["split"] for row in rows} == set(ingest.SPLITS)
    assert len(rows) == 3 and len(summary["dut_cross_split_exact_duplicate_hashes"]) == 1
    assert json.loads((output / "summary.json").read_text())["complete"]
    with pytest.raises(FileExistsError):
        ingest.prepare(raw, output, datasets=["dut"])


def test_incomplete_raw_never_marks_complete(tmp_path):
    raw = tmp_path / "raw"
    (raw / "dut_anti_uav/extracted/train/img").mkdir(parents=True)
    output = tmp_path / "new_dataset"
    with pytest.raises(ingest.BadRecord, match="incomplete"):
        ingest.prepare(raw, output, datasets=["dut"])
    assert json.loads((output / "summary.json").read_text())["complete"] is False
