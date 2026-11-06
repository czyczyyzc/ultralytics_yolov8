"""Torch-free tests; run pytest with --noconftest --import-mode=importlib."""

import json
import sys
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


def tree(frames, fps=30, timestamp_origin_ms=0):
    cells = []
    for index, boxes in enumerate(frames):
        time = matrix(13, (6, 1), [numeric(6, [3707764736, 2, 1, 1, 8 + index, 1])])
        classes = boxes if isinstance(boxes, dict) else {"DRONE": boxes}
        cells.extend([time, *(double(classes.get(name, ())) for name in ingest.FIELDS[1:])])
    names = b"".join(s.encode().ljust(11, b"\0") for s in ingest.FIELDS)
    table = matrix(2, (len(frames), 1), [numeric(5, [11]), element(1, names), *cells])
    times = matrix(6, (len(frames), 1), [numeric(9, timestamp_origin_ms + np.arange(len(frames)) * 1000 / fps)])
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


@pytest.mark.parametrize("name", ingest.HALMSTAD_CLASSES)
def test_aircraft_opt_in_preserves_every_box_and_source_class(name):
    boxes = ((1, 2, 3, 4), (5, 6, 0.25, 0.5))
    tops = tree([{name: boxes}, {}])
    frames, times, classes = ingest.parse_halmstad_tree(tops, 2, 30, ingest.HALMSTAD_CLASSES,
                                                       return_source_classes=True)
    assert frames == [boxes, None]
    assert classes == [(name, name), ()]
    assert times == pytest.approx([0, 1000 / 30])
    if name != "DRONE":
        with pytest.raises(ingest.BadRecord, match="non_drone"):
            ingest.parse_halmstad_tree(tops, 2, 30)


def test_mixed_aircraft_preserves_per_box_classes_not_filename_guesses():
    boxes = ((1, 1, 2, 2),)
    frames, _, classes = ingest.parse_halmstad_tree(
        tree([dict.fromkeys(ingest.HALMSTAD_CLASSES, boxes)]), 1, 30,
        ingest.HALMSTAD_CLASSES, return_source_classes=True)
    assert frames == [boxes * 3]
    assert classes == [("AIRPLANE", "DRONE", "HELICOPTER")]


@pytest.mark.parametrize("classes", [(), ("BIRD",), ("DRONE", "BIRD"), ("UNKNOWN",), "DRONE"])
def test_unsafe_class_selection_rejected_before_output(tmp_path, classes):
    with pytest.raises(ValueError, match="nonempty_aircraft_list"):
        ingest.parse_halmstad_tree([], 1, 30, classes)
    with pytest.raises(ValueError, match="nonempty_aircraft_list"):
        ingest.prepare(tmp_path, tmp_path / "new", halmstad_classes=classes)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("origin", [1000 / 30, -1000 / 30, 100, 1000])
@pytest.mark.parametrize("classes", [("DRONE",), ingest.HALMSTAD_CLASSES])
def test_nonzero_timestamp_origins_stay_quarantined_no_automatic_offset(origin, classes):
    with pytest.raises(ingest.BadRecord, match="timestamps"):
        ingest.parse_halmstad_tree(tree([(), (), ()], timestamp_origin_ms=origin), 3, 30, classes)


def test_bird_mixed_frame_and_unknown_named_fields_quarantine_whole_clip():
    boxes = ((1, 1, 2, 2),)
    for bird_frame in ({"BIRD": boxes}, {"DRONE": boxes, "BIRD": boxes}):
        with pytest.raises(ingest.BadRecord, match="disallowed_halmstad_object_class: BIRD"):
            ingest.parse_halmstad_tree(tree([{"AIRPLANE": boxes}, bird_frame]), 2, 30, ingest.HALMSTAD_CLASSES)
    tops = tree([{"DRONE": boxes}])
    table = tops[0].children[0].children[4]
    children = list(table.children)
    children[4] = element(1, children[4].payload.replace(b"BIRD", b"DUCK"))
    table.children = tuple(children)
    with pytest.raises(ingest.BadRecord, match="fields"):
        ingest.parse_halmstad_tree(tops, 1, 30, ingest.HALMSTAD_CLASSES)


def wrapped_position(boxes=((10, 273, 160, 98),), fields=b"Position\0\0LabelUIDs\0",
                     uid=b"34555d2a-2e8e-4acc-bb98-d5393d63955b"):
    return matrix(2, (1, 1), [numeric(5, [10]), element(1, fields), double(boxes),
                              matrix(4, (1, 36), [element(16, uid)])])


def test_verified_airplane_position_uid_wrapper_preserves_box():
    tops = tree([{}, {"AIRPLANE": ((1, 1, 2, 2),)}, {}])
    table = tops[0].children[0].children[4]
    children = list(table.children)
    children[11] = wrapped_position()
    table.children = tuple(children)
    frames, _, classes = ingest.parse_halmstad_tree(tops, 3, 30, ("AIRPLANE",), return_source_classes=True)
    assert frames == [None, ((10, 273, 160, 98),), None]
    assert classes == [(), ("AIRPLANE",), ()]
    with pytest.raises(ingest.BadRecord, match="non_drone"):
        ingest.parse_halmstad_tree(tops, 3, 30)


@pytest.mark.parametrize("wrapper,reason", [
    (wrapped_position(fields=b"Location\0\0LabelUIDs\0"), "unsupported_label_struct"),
    (wrapped_position(uid=b"not a verified UUID"), "unsupported_label_uid"),
    (wrapped_position(boxes=((1, 1, 2, 2), (3, 3, 4, 4))), "unsupported_label_struct_position"),
])
def test_unverified_label_struct_variants_quarantined(wrapper, reason):
    tops = tree([{}])
    table = tops[0].children[0].children[4]
    children = list(table.children)
    children[6] = wrapper
    table.children = tuple(children)
    with pytest.raises(ingest.BadRecord, match=reason):
        ingest.parse_halmstad_tree(tops, 1, 30, ingest.HALMSTAD_CLASSES)


@pytest.mark.parametrize("name,classes", [("DRONE", ("DRONE",)), ("HELICOPTER", ingest.HALMSTAD_CLASSES)])
def test_load_forwards_allowed_classes_and_keeps_default_pair(monkeypatch, name, classes):
    tops = tree([{name: ((1, 1, 2, 2),)}])
    fake_io = SimpleNamespace(loadmat=lambda path: {"__function_workspace__": np.zeros(8, np.uint8)})
    monkeypatch.setitem(sys.modules, "scipy", SimpleNamespace(io=fake_io))
    monkeypatch.setitem(sys.modules, "scipy.io", fake_io)
    monkeypatch.setitem(sys.modules, "mcos_decoder", SimpleNamespace(
        MatStream=lambda data, skip_header: SimpleNamespace(walk=lambda: iter(tops))))
    kwargs = {} if name == "DRONE" else {"allowed_classes": classes}
    assert len(ingest.load_halmstad("unused.mat", 1, 30, **kwargs)) == 2
    frames, _, source_classes = ingest.load_halmstad("unused.mat", 1, 30, **kwargs, return_source_classes=True)
    assert frames == [((1, 1, 2, 2),)] and source_classes == [(name,)]


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
    monkeypatch.setattr(ingest, "load_halmstad", lambda *args, **kwargs: (
        [((1, 1, 1, 1),), None, ((3, 4, 2, 2), (8, 9, 1, 1))], [0, 33, 66],
        [("DRONE",), (), ("DRONE", "DRONE")]))
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
    assert all(r["source_class"] == "DRONE" and r["source_classes"] == ["DRONE"] for r in rows)


def fake_halmstad_capture(monkeypatch, count=3):
    class Capture:
        def __init__(self, path):
            self.index = 0
        def isOpened(self):
            return True
        def get(self, key):
            return {cv2.CAP_PROP_FRAME_COUNT: count, cv2.CAP_PROP_FPS: 30,
                    cv2.CAP_PROP_FRAME_WIDTH: 20, cv2.CAP_PROP_FRAME_HEIGHT: 16}[key]
        def read(self):
            self.index += 1
            return (True, np.full((16, 20, 3), self.index, np.uint8)) if self.index <= count else (False, None)
        def release(self):
            pass
    monkeypatch.setattr(cv2, "VideoCapture", Capture)


@pytest.mark.parametrize("filename", ["V_BIRD_001.mp4", "V_UNKNOWN_001.mp4"])
def test_export_never_opens_bird_or_unknown_clip(tmp_path, monkeypatch, filename):
    output = output_dirs(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("excluded source must not be opened")

    monkeypatch.setattr(ingest, "load_halmstad", forbidden)
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    video = tmp_path / filename
    plan = (video, video.with_suffix(".mat"), "0" * 64, 3, 30, 20, 16)
    rows, bad = ingest.export_halmstad(plan, output, ingest.DiskGuard(output), ingest.HALMSTAD_CLASSES)
    assert rows == [] and "excluded_or_unknown_clip_class_not_negative" in bad[0]["reason"]
    assert not list((output / "labels").rglob("*.txt"))


@pytest.mark.parametrize("name", ["AIRPLANE", "HELICOPTER"])
def test_aircraft_export_class_zero_clips_all_boxes_preserves_tiny(tmp_path, monkeypatch, name):
    output = output_dirs(tmp_path)
    video = tmp_path / ("V_" + name + "_001.mp4")
    annotation = video.with_name(video.stem + "_LABELS.mat")
    video.write_bytes(b"video")
    annotation.write_bytes(b"labels")
    fake_halmstad_capture(monkeypatch)
    boxes = ((0, 0, 3, 3), (20.5, 16.5, 2, 2), (3.25, 4.5, 0.25, 0.5))
    tops = tree([{name: boxes}, {}, {name: boxes, "DRONE": ((1, 1, 1, 1),)}])
    monkeypatch.setattr(ingest, "load_halmstad", lambda path, count, fps, classes, **kwargs:
                        ingest.parse_halmstad_tree(tops, count, fps, classes, **kwargs))
    plan = (video, annotation, ingest.sha256(video), 3, 30, 20, 16)
    rows, bad = ingest.export_halmstad(plan, output, ingest.DiskGuard(output), ingest.HALMSTAD_CLASSES)
    assert [r["original_frame_number"] for r in rows] == [1, 3]
    assert [r["boxes"] for r in rows] == [3, 4]
    assert all(r["source_class"] == name and r["clipped_boxes"] == 2 for r in rows)
    assert rows[0]["source_classes"] == [name]
    assert rows[1]["source_classes"] == sorted([name, "DRONE"])
    assert len(bad) == 1 and bad[0]["reason"] == "unlabeled_frame_not_negative"
    for row in rows:
        lines = [line.split() for line in Path(row["label"]).read_text().splitlines()]
        assert len(lines) == row["boxes"] and all(line[0] == "0" for line in lines)
        assert len(row["box_source_classes"]) == len(lines)
        for line, raw in zip(lines, row["raw_boxes_matlab_xywh"]):
            x, y, w, h = map(float, line[1:])
            assert 0 < w <= 1 and 0 < h <= 1
            decoded = ((x - w / 2) * 20, (y - h / 2) * 16, (x + w / 2) * 20, (y + h / 2) * 16)
            assert decoded == pytest.approx(ingest.matlab_xywh(raw, 20, 16)[0])
        assert any(float(line[3]) == pytest.approx(0.25 / 20) for line in lines)


@pytest.mark.parametrize("bad_frame", [
    {"BIRD": ((1, 1, 2, 2),)},
    {"AIRPLANE": ((1, 1, 2, 2),), "BIRD": ((2, 2, 3, 3),)},
])
def test_bird_in_later_frame_exports_nothing_not_even_earlier_positives(tmp_path, monkeypatch, bad_frame):
    output = output_dirs(tmp_path)
    video = tmp_path / "V_AIRPLANE_001.mp4"
    annotation = video.with_suffix(".mat")
    video.write_bytes(b"video")
    annotation.write_bytes(b"labels")
    tops = tree([{"AIRPLANE": ((1, 1, 2, 2),)}, bad_frame, {}])
    fake_halmstad_capture(monkeypatch)
    monkeypatch.setattr(ingest, "load_halmstad", lambda path, count, fps, classes, **kwargs:
                        ingest.parse_halmstad_tree(tops, count, fps, classes, **kwargs))
    plan = (video, annotation, ingest.sha256(video), 3, 30, 20, 16)
    rows, bad = ingest.export_halmstad(plan, output, ingest.DiskGuard(output), ingest.HALMSTAD_CLASSES)
    assert rows == [] and "BIRD" in bad[0]["reason"]
    assert "quarantined_whole_clip" in bad[0]["reason"]
    assert not list((output / "images").rglob("*.jpg"))
    assert not list((output / "labels").rglob("*.txt"))


@pytest.mark.parametrize("classes", [("DRONE",), ("AIRPLANE", "HELICOPTER"), ingest.HALMSTAD_CLASSES])
def test_prepare_selects_only_requested_visible_clips_and_estimates_only_new_output(tmp_path, monkeypatch, classes):
    raw = tmp_path / "raw"
    base = raw / "halmstad_drone_detection/extracted" / ingest.HALMSTAD_DIR / "Data/Video_V"
    base.mkdir(parents=True)
    for name in ingest.HALMSTAD_COUNTS:
        (base / ("V_" + name + "_001.mp4")).write_bytes(name.encode())
        (base / ("V_" + name + "_001_LABELS.mat")).write_bytes(b"labels")
    ir = base.parent / "Video_IR"
    ir.mkdir()
    (ir / "V_AIRPLANE_999.mp4").write_bytes(b"IR must not be read")
    monkeypatch.setitem(sys.modules, "mcos_decoder", SimpleNamespace(__version__="0.1.0"))
    monkeypatch.setattr(ingest, "HALMSTAD_COUNTS", dict.fromkeys(ingest.HALMSTAD_COUNTS, 1))
    monkeypatch.setattr(ingest, "audit_archive", lambda *args: dict(complete=True, verified_members=8))
    fake_halmstad_capture(monkeypatch, count=1)

    def load(path, count, fps, allowed_classes, **kwargs):
        name = path.name.split("_")[1]
        return ingest.parse_halmstad_tree(tree([{name: ((1, 1, 1, 1),)}]), count, fps, allowed_classes, **kwargs)

    monkeypatch.setattr(ingest, "load_halmstad", load)
    kwargs = {} if classes == ("DRONE",) else {"halmstad_classes": classes}
    output = tmp_path / "new"
    summary = ingest.prepare(raw, output, datasets=["halmstad"], workers=1, **kwargs)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert summary["complete"] and len(rows) == len(classes)
    assert {r["source_class"] for r in rows} == set(classes)
    assert summary["policy"]["halmstad_classes"] == list(classes)
    assert summary["policy"]["halmstad_modality"] == "visible_only"
    assert summary["estimated_output_upper_bytes"] == len(classes) * (20 * 16 * 3 + 65536)
    assert {r["source_class"] for r in summary["excluded_clips"]} == set(ingest.HALMSTAD_COUNTS) - set(classes)


def test_cli_classes_default_and_incremental_birds_never_accepted(tmp_path, monkeypatch):
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return dict(complete=True, output=str(kwargs["output"]), counts={})

    monkeypatch.setattr(ingest, "prepare", prepare)
    args = ["--raw-root", str(tmp_path / "raw"), "--output", str(tmp_path / "new")]
    assert ingest.main(args) == 0
    assert calls[-1]["halmstad_classes"] == ["DRONE"]
    assert ingest.main(args + ["--halmstad-classes", "AIRPLANE", "HELICOPTER"]) == 0
    assert calls[-1]["halmstad_classes"] == ["AIRPLANE", "HELICOPTER"]
    for name in ("BIRD", "UNKNOWN"):
        with pytest.raises(SystemExit) as exc:
            ingest.main(args + ["--halmstad-classes", name])
        assert exc.value.code == 2


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
