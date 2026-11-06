#!/usr/bin/env python3
"""Strict, CPU-only DUT detection/Halmstad visible-light research ingester.

Requires numpy, opencv-python-headless, scipy, and mcos-decoder==0.1.0.
Never imports ultralytics/torch, extracts archives, downloads, or starts training.
Only a NEW output directory is accepted. Consumers must require summary.complete.

Halmstad 0.1.0 load_groundtruth is intentionally NOT used: it mistakes a source
string for an empty frame and a 2x4 double matrix for an empty label. Instead we
read the explicitly named MCOS struct table, validate timestamps against FPS, and
check its row count against both the video header and a single sequential decode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import sys
import threading
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


DUT_COUNTS = {"train": 5200, "val": 2600, "test": 2200}
HALMSTAD_COUNTS = {"DRONE": 114, "AIRPLANE": 59, "BIRD": 51, "HELICOPTER": 61}
HALMSTAD_CLASSES = ("DRONE", "AIRPLANE", "HELICOPTER")
HALMSTAD_DIR = "DroneDetectionThesis-Drone-detection-dataset-e7a6eaf"
FIELDS = ("Time", "AIRPLANE", "BIRD", "DRONE", "HELICOPTER")
VIDEO_NAME = re.compile(r"V_(DRONE|AIRPLANE|BIRD|HELICOPTER)_\d{3}\.mp4\Z")
SPLITS = ("train", "val", "test")
GIB = 1024**3


class BadRecord(ValueError):
    """An untrusted annotation must not become a negative training example."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(video_sha):
    """Fixed 20% whole-content-hash holdout, independent of paths and workers."""
    if not re.fullmatch(r"[0-9a-f]{64}", video_sha):
        raise BadRecord("invalid_video_sha256")
    score = int(hashlib.sha256(("public-uav-research-v1:" + video_sha).encode()).hexdigest(), 16)
    return "val" if score < (1 << 256) // 5 else "train"


def clipped_box(values, width, height):
    if len(values) != 4:
        raise BadRecord("box_must_have_four_values")
    box = tuple(float(x) for x in values)
    if not all(math.isfinite(x) for x in box):
        raise BadRecord("nonfinite_box")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise BadRecord("degenerate_box")
    clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise BadRecord("box_outside_image")
    return clipped, clipped != box


def matlab_xywh(values, width, height):
    """MATLAB one-based xywh -> zero-based xyxy; translate origin, NOT size."""
    if len(values) != 4:
        raise BadRecord("matlab_box_must_have_four_values")
    x, y, w, h = (float(v) for v in values)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise BadRecord("invalid_matlab_xywh")
    return clipped_box((x - 1, y - 1, x - 1 + w, y - 1 + h), width, height)


def yolo_line(box, width, height):
    x1, y1, x2, y2 = box
    return "0 " + " ".join(format(x, ".17g") for x in (
        (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
        (x2 - x1) / width, (y2 - y1) / height,
    ))


def parse_dut_xml(path, image):
    import xml.etree.ElementTree as ET
    import cv2

    try:
        data = Path(path).read_bytes()
        if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
            raise BadRecord("xml_entities_not_allowed")
        root = ET.fromstring(data)
        if root.tag != "annotation" or root.findtext("filename") != image.name:
            raise BadRecord("xml_filename_or_root_mismatch")
        width, height = (int(root.findtext("size/" + key, "0")) for key in ("width", "height"))
        if width <= 0 or height <= 0:
            raise BadRecord("invalid_xml_size")
        pixels = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.shape[:2] != (height, width):
            raise BadRecord("image_decode_or_xml_size_mismatch")
        boxes, clipped = [], 0
        for obj in root.findall("object"):
            if obj.findtext("name") != "UAV":
                raise BadRecord("unknown_object_class")
            if obj.findtext("difficult") != "0":
                raise BadRecord("difficult_or_missing_flag")
            bnd = obj.find("bndbox")
            if bnd is None:
                raise BadRecord("missing_bndbox")
            # Preserve DUT's published xyxy coordinates; no unverified VOC
            # origin shift or inclusive +1. Boundary clipping is audited.
            values = [float(bnd.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")]
            box, changed = clipped_box(values, width, height)
            boxes.append(box)
            clipped += changed
        return boxes, width, height, clipped
    except (ET.ParseError, TypeError, ValueError, OSError) as exc:
        if isinstance(exc, BadRecord):
            raise
        raise BadRecord("invalid_dut_record: " + str(exc)) from exc


def _walk(element):
    yield element
    for child in element.children:
        yield from _walk(child)


def _matrix(element):
    """Validate MAT5 matrix headers before interpreting their contents."""
    c = element.children
    if element.type_id != 14 or len(c) < 3 or c[0].type_id != 6 or c[1].type_id != 5 or c[2].type_id != 1:
        raise BadRecord("unsupported_mcos_matrix_header")
    flags = c[0].as_array()
    if len(flags) != 2 or int(flags[0]) & ~255:
        raise BadRecord("complex_sparse_or_flagged_mcos_matrix")
    return int(flags[0]), tuple(int(x) for x in c[1].as_array()), c


def validate_halmstad_classes(allowed_classes):
    """Birds can never be opted in, including through the Python API."""
    if isinstance(allowed_classes, str):
        raise ValueError("halmstad_classes_must_be_a_nonempty_aircraft_list")
    classes = tuple(allowed_classes)
    if not classes or any(name not in HALMSTAD_CLASSES for name in classes):
        raise ValueError("halmstad_classes_must_be_a_nonempty_aircraft_list")
    return tuple(name for name in HALMSTAD_CLASSES if name in classes)


def parse_halmstad_tree(tops, frame_count, fps, allowed_classes=("DRONE",), *, return_source_classes=False):
    """Only the firsthand-verified R2020a named struct format is accepted.

    The table stores frame-major rows with five fields, including Time. Boxes
    are MATLAB column-major Nx4 doubles. This supports ALL boxes, not just N=1.
    Disallowed nonempty classes invalidate the whole clip, never negatives.
    Nonzero timestamp origins remain quarantined: observing a one-frame offset
    alone does not establish annotation/video alignment.
    """
    allowed_classes = validate_halmstad_classes(allowed_classes)
    if frame_count < 1 or not math.isfinite(fps) or fps <= 0:
        raise BadRecord("invalid_video_metadata")
    tables, timestamps = [], []
    for top in tops:
        for element in _walk(top):
            if element.type_id != 14 or len(element.children) < 4:
                continue
            c = element.children
            # MCOS opaque containers have a different header (no dimensions).
            # Traverse them, but interpret only standard MAT matrix headers.
            if c[0].type_id != 6 or c[1].type_id != 5 or c[2].type_id != 1:
                continue
            kind, dims, c = _matrix(element)
            if kind == 2 and len(c) >= 5 and c[3].type_id == 5 and c[4].type_id == 1:
                lengths = c[3].as_array()
                if len(lengths) != 1 or int(lengths[0]) <= 0:
                    raise BadRecord("invalid_struct_field_length")
                length = int(lengths[0])
                if len(c[4].payload) % length:
                    raise BadRecord("invalid_struct_field_names")
                names = tuple(c[4].payload[i:i + length].rstrip(b"\0").decode("ascii")
                              for i in range(0, len(c[4].payload), length))
                if "DRONE" in names and "Time" in names:
                    if names != FIELDS or dims != (frame_count, 1):
                        raise BadRecord("unsupported_label_table_fields_or_frame_count")
                    tables.append(c[5:])
            if kind == 6 and dims == (frame_count, 1) and len(c) == 4 and c[3].type_id == 9:
                times = c[3].as_array()
                if len(times) == frame_count and all(
                    math.isfinite(float(t)) and abs(float(t) - i * 1000 / fps) <= 1.0
                    for i, t in enumerate(times)
                ):
                    timestamps.append(tuple(float(t) for t in times))
    if len(tables) != 1 or len(timestamps) != 1:
        raise BadRecord("ambiguous_or_missing_label_table_or_millisecond_timestamps")
    cells = tables[0]
    if len(cells) != frame_count * len(FIELDS):
        raise BadRecord("label_table_cell_count_mismatch")
    frames, frame_classes = [], []
    first_time_reference = None
    for index in range(frame_count):
        row = cells[index * len(FIELDS):(index + 1) * len(FIELDS)]
        time_kind, time_dims, tc = _matrix(row[0])
        if time_kind != 13 or time_dims != (6, 1) or len(tc) != 4 or tc[3].type_id != 6:
            raise BadRecord("unsupported_row_timestamp_reference")
        reference = tuple(int(v) for v in tc[3].as_array())
        if len(reference) != 6 or reference[:4] != (3707764736, 2, 1, 1) or reference[-1] != 1:
            raise BadRecord("unsupported_mcos_time_reference_encoding")
        if first_time_reference is None:
            first_time_reference = reference[4]
        if reference[4] != first_time_reference + index:
            raise BadRecord("nonsequential_row_timestamp_reference")
        boxes, source_classes = [], []
        for name, cell in zip(FIELDS[1:], row[1:]):
            kind, dims, c = _matrix(cell)
            if kind == 2:
                # Verified in visible V_AIRPLANE_002 frame 263: one Position
                # box plus a UUID, NOT a generic struct/size-based fallback.
                if (dims != (1, 1) or len(c) != 7 or c[3].type_id != 5 or
                        tuple(c[3].as_array()) != (10,) or c[4].type_id != 1 or
                        c[4].payload != b"Position\0\0LabelUIDs\0"):
                    raise BadRecord("unsupported_label_struct")
                uid_kind, uid_dims, uid = _matrix(c[6])
                if (uid_kind != 4 or uid_dims != (1, 36) or len(uid) != 4 or uid[3].type_id != 16 or
                        not re.fullmatch(rb"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uid[3].payload)):
                    raise BadRecord("unsupported_label_uid")
                kind, dims, c = _matrix(c[5])
                if dims != (1, 4):
                    raise BadRecord("unsupported_label_struct_position")
            if kind != 6:
                raise BadRecord("nonnumeric_label_cell")
            if dims == (0, 0) and len(c) == 4 and c[3].type_id == 9 and not c[3].payload:
                continue
            if len(dims) != 2 or dims[0] < 1 or dims[1] != 4 or len(c) != 4 or c[3].type_id != 9:
                raise BadRecord("unsupported_label_cell_shape")
            if name not in allowed_classes:
                reason = "non_drone_object_in_drone_clip" if allowed_classes == ("DRONE",) else "disallowed_halmstad_object_class"
                raise BadRecord(reason + ": " + name)
            values = c[3].as_array()
            n = dims[0]
            if len(values) != n * 4:
                raise BadRecord("label_payload_size_mismatch")
            boxes.extend(tuple(float(values[j * n + i]) for j in range(4)) for i in range(n))
            source_classes.extend([name] * n)
        # Empty/None is unlabeled, NEVER a negative.
        frames.append(tuple(boxes) if boxes else None)
        frame_classes.append(tuple(source_classes))
    if return_source_classes:
        return frames, timestamps[0], frame_classes
    return frames, timestamps[0]


def load_halmstad(path, frame_count, fps, allowed_classes=("DRONE",), *, return_source_classes=False):
    allowed_classes = validate_halmstad_classes(allowed_classes)
    from scipy.io import loadmat
    from mcos_decoder import MatStream

    raw = loadmat(str(path))
    if "__function_workspace__" not in raw:
        raise BadRecord("not_mcos_groundtruth")
    return parse_halmstad_tree(MatStream(raw["__function_workspace__"].tobytes(), skip_header=8).walk(),
                               frame_count, fps, allowed_classes, return_source_classes=return_source_classes)


def audit_archive(path, extracted, keep):
    """Read only: hash archive, verify ZIP CRCs, and compare reused raw members."""
    result = {"source": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
              "verified_members": 0, "hash_scope": "local_fingerprint_not_upstream_authentication"}
    with zipfile.ZipFile(path) as archive:
        names = [m.filename for m in archive.infolist()]
        if len(set(names)) != len(names):
            raise BadRecord("duplicate_archive_members: " + str(path))
        bad = archive.testzip()
        if bad:
            raise BadRecord("archive_crc_failure: " + bad)
        for member in archive.infolist():
            if member.is_dir() or not keep(member.filename):
                continue
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise BadRecord("unsafe_archive_member")
            target = extracted.joinpath(*relative.parts)
            if not target.is_file() or target.stat().st_size != member.file_size:
                raise BadRecord("missing_or_size_mismatched_raw_member: " + str(target))
            crc = 0
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b""):
                    crc = zlib.crc32(block, crc)
            if crc & 0xFFFFFFFF != member.CRC:
                raise BadRecord("raw_member_crc_mismatch: " + str(target))
            result["verified_members"] += 1
    result["complete"] = True
    return result


class DiskGuard:
    def __init__(self, path):
        self.path = path
        usage = shutil.disk_usage(path)
        self.margin = max(5 * GIB, min(40 * GIB, math.ceil(usage.total * 0.05)))
        self.lock = threading.Lock()

    def check(self, needed=0):
        if shutil.disk_usage(self.path).free < self.margin + needed:
            raise OSError("insufficient_disk_margin: need free >= margin + estimated output bytes")

    def write(self, path, payload):
        with self.lock:
            self.check(len(payload) + 8192)
            with path.open("xb") as stream:
                stream.write(payload)


def check_paths(raw_root, output):
    raw_root, output = raw_root.resolve(), output.resolve()
    if raw_root == output or raw_root in output.parents or output in raw_root.parents:
        raise ValueError("output_must_be_disjoint_from_raw_root")
    if os.path.lexists(output):
        raise FileExistsError("output_must_not_exist: " + str(output))
    if not raw_root.is_dir():
        raise FileNotFoundError(str(raw_root))
    return raw_root, output


def _record(dataset, split, group, source, image, label, boxes, **metadata):
    return dict(dataset=dataset, split=split, group=group, source=str(source.resolve()),
                image=str(image.absolute()), label=str(label.absolute()), boxes=len(boxes), **metadata)


def export_dut(job, output, disk):
    split, xml, image = job
    try:
        boxes, width, height, clipped = parse_dut_xml(xml, image)
    except BadRecord as exc:
        return None, dict(dataset="dut", split=split, source=str(xml), reason=str(exc))
    name = "dut_" + split + "_" + image.name
    dst = output / "images" / split / name
    label = output / "labels" / split / (Path(name).stem + ".txt")
    disk.write(label, "".join(yolo_line(b, width, height) + "\n" for b in boxes).encode("ascii"))
    dst.symlink_to(image.resolve())
    return _record("dut", split, "dut/" + split + "/" + image.stem, image, dst, label, boxes,
                   annotation=str(xml.resolve()), raw_image_sha256=sha256(image),
                   width=width, height=height, clipped_boxes=clipped,
                   coordinates="dut_published_xyxy_no_origin_shift_clip_to_image"), None


def export_halmstad(plan, output, disk, allowed_classes=("DRONE",)):
    import cv2

    allowed_classes = validate_halmstad_classes(allowed_classes)
    video, annotation, digest, count, fps, width, height = plan
    group, split = "halmstad/" + digest, stable_split(digest)
    rows, bad, created = [], [], []
    cap = None
    match = VIDEO_NAME.fullmatch(video.name)
    source_class = match[1] if match else None
    try:
        if source_class not in allowed_classes:
            raise BadRecord("excluded_or_unknown_clip_class_not_negative")
        frames, times, frame_classes = load_halmstad(annotation, count, fps, allowed_classes, return_source_classes=True)
        if len(frames) != count or len(times) != count or len(frame_classes) != count:
            raise BadRecord("label_frame_metadata_count_mismatch")
        ann_sha = sha256(annotation)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise BadRecord("video_open_failed")
        for index, raw_boxes in enumerate(frames):
            ok, pixels = cap.read()
            if not ok or pixels is None or pixels.shape[:2] != (height, width):
                raise BadRecord("sequential_decode_failed_at_frame_" + str(index + 1))
            if not raw_boxes:
                bad.append(dict(dataset="halmstad", source=str(video), original_frame_number=index + 1,
                                source_class=source_class,
                                reason="unlabeled_frame_not_negative"))
                continue
            classes = frame_classes[index]
            if len(classes) != len(raw_boxes) or any(name not in allowed_classes for name in classes):
                raise BadRecord("invalid_box_source_classes")
            try:
                converted = [matlab_xywh(b, width, height) for b in raw_boxes]
            except BadRecord as exc:
                bad.append(dict(dataset="halmstad", source=str(video), original_frame_number=index + 1,
                                source_class=source_class, source_classes=sorted(set(classes)), reason=str(exc)))
                continue
            boxes = [b for b, changed in converted]
            name = "halmstad_" + video.stem + "_" + digest[:16] + "_" + f"{index + 1:08d}"
            image = output / "images" / split / (name + ".jpg")
            label = output / "labels" / split / (name + ".txt")
            ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise BadRecord("jpeg_encode_failed")
            created.append(image)
            disk.write(image, encoded.tobytes())
            created.append(label)
            disk.write(label, "".join(yolo_line(b, width, height) + "\n" for b in boxes).encode("ascii"))
            rows.append(_record("halmstad", split, group, video, image, label, boxes,
                                original_frame_number=index + 1, raw_video_sha256=digest,
                                annotation=str(annotation.resolve()), annotation_sha256=ann_sha,
                                source_class=source_class, source_classes=sorted(set(classes)),
                                box_source_classes=classes,
                                timestamp_ms=times[index], raw_boxes_matlab_xywh=raw_boxes,
                                width=width, height=height, jpeg_quality=95,
                                clipped_boxes=sum(changed for b, changed in converted),
                                coordinates="matlab_one_based_xywh_translate_xy_minus_one_keep_wh"))
        if cap.read()[0]:
            raise BadRecord("extra_decoded_frames_beyond_label_table")
    except OSError:
        raise
    except Exception as exc:
        for path in created:
            path.unlink(missing_ok=True)
        return [], [dict(dataset="halmstad", source=str(video), group=group,
                         source_class=source_class,
                         reason="quarantined_whole_clip: " + str(exc))]
    finally:
        if cap is not None:
            cap.release()
    return rows, bad


def audit_manifest(rows):
    groups, videos, images, labels = {}, {}, set(), set()
    for row in rows:
        if row["image"] in images or row["label"] in labels:
            raise BadRecord("duplicate_output_path")
        images.add(row["image"])
        labels.add(row["label"])
        for key, registry in ((row["group"], groups), (row.get("raw_video_sha256"), videos)):
            if key is not None and registry.setdefault(key, row["split"]) != row["split"]:
                raise BadRecord("cross_split_group_or_video_leak")
        if row["dataset"] == "halmstad" and row["boxes"] < 1:
            raise BadRecord("halmstad_negative_forbidden")
    return {"groups": len(groups), "halmstad_unique_video_hashes": len(videos),
            "cross_split_group_overlap": 0,
            "scope": "group_and_raw_video_hash; DUT official content duplicates reported separately, not resplit"}


def prepare(raw_root, output, datasets=("dut", "halmstad"), workers=2, halmstad_decoder_target=None,
            halmstad_classes=("DRONE",)):
    if workers < 1 or workers > 8 or not datasets or not set(datasets) <= {"dut", "halmstad"}:
        raise ValueError("workers_must_be_1_to_8_and_datasets_dut_or_halmstad")
    allowed_classes = validate_halmstad_classes(halmstad_classes)
    raw_root, output = check_paths(Path(raw_root), Path(output))
    if halmstad_decoder_target:
        sys.path.insert(0, str(Path(halmstad_decoder_target).resolve()))
        importlib.invalidate_caches()
    import cv2

    cv2.setNumThreads(1)
    if "halmstad" in datasets:
        import mcos_decoder
        if getattr(mcos_decoder, "__version__", None) != "0.1.0":
            raise ValueError("requires_verified_mcos_decoder_0.1.0")
    output.parent.mkdir(parents=True, exist_ok=True)
    disk = DiskGuard(output.parent)
    disk.check()
    output.mkdir(exist_ok=False)
    summary = dict(complete=False, research_only=True, raw_root=str(raw_root), output=str(output),
                   datasets=sorted(set(datasets)), workers=workers, archives=[], bad_records=[],
                   disk_margin_bytes=disk.margin, source_completeness={},
                   licenses={"dut": {"repository_license": "Apache-2.0", "dataset_rights": "not independently cleared",
                                      "source": "https://github.com/wangdongdut/DUT-Anti-UAV"},
                             "halmstad": {"license": "CC0-1.0", "source": "https://github.com/DroneDetectionThesis/Drone-detection-dataset"}},
                   policy={"use": "research_only_not_a_commercial_rights_clearance", "halmstad_classes": list(allowed_classes),
                           "halmstad_modality": "visible_only", "halmstad_positive_class_id": 0,
                           "birds": "excluded_never_negative", "halmstad_timestamp_origin_ms": 0,
                           "negative_aircraft": False, "sampling": "none", "halmstad_validation": "sha256_content_hash_20_percent_v1",
                           "frame_number_base": 1, "jpeg_quality": 95, "dut_splits": "official_unchanged"})

    def save_summary():
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / "summary.json")

    save_summary()
    try:
        for split in SPLITS:
            (output / "images" / split).mkdir(parents=True)
            (output / "labels" / split).mkdir(parents=True)
        dut_jobs, plans, excluded = [], [], []
        if "dut" in datasets:
            root = raw_root / "dut_anti_uav"
            counts = {}
            for split, expected in DUT_COUNTS.items():
                base = root / "extracted" / split
                xmls = sorted((base / "xml").glob("*.xml"))
                images = sorted(p for p in (base / "img").iterdir() if p.is_file())
                lookup = {}
                for image in images:
                    if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"} or image.stem in lookup:
                        raise BadRecord("unsupported_or_ambiguous_dut_image: " + str(image))
                    lookup[image.stem] = image
                if len(xmls) != expected or len(images) != expected or {p.stem for p in xmls} != set(lookup):
                    raise BadRecord("dut_official_split_incomplete: " + split)
                archive = audit_archive(root / "archives" / (split + ".zip"), root / "extracted",
                                        lambda name, s=split: name.startswith(s + "/img/") or name.startswith(s + "/xml/"))
                if archive["verified_members"] != expected * 2:
                    raise BadRecord("dut_archive_member_count_mismatch")
                summary["archives"].append(archive)
                dut_jobs.extend((split, p, lookup[p.stem]) for p in xmls)
                counts[split] = {"expected": expected, "images": len(images), "xml": len(xmls)}
            summary["source_completeness"]["dut"] = counts
        estimate = len(dut_jobs) * 16384
        if "halmstad" in datasets:
            root = raw_root / "halmstad_drone_detection"
            base = root / "extracted" / HALMSTAD_DIR / "Data" / "Video_V"
            videos = sorted(base.glob("*.mp4"))
            counts = Counter()
            seen = set()
            for video in videos:
                match = VIDEO_NAME.fullmatch(video.name)
                if not match:
                    raise BadRecord("unsupported_halmstad_video_name: " + video.name)
                counts[match[1]] += 1
                if not video.with_name(video.stem + "_LABELS.mat").is_file():
                    raise BadRecord("missing_halmstad_labels: " + video.name)
            if dict(counts) != HALMSTAD_COUNTS or len(list(base.glob("*_LABELS.mat"))) != len(videos):
                raise BadRecord("halmstad_visible_inventory_incomplete")
            archive = audit_archive(root / "archives" / "Drone-detection-dataset-v1.0.0.zip", root / "extracted",
                                    lambda name: name.startswith(HALMSTAD_DIR + "/Data/Video_V/"))
            if archive["verified_members"] != len(videos) * 2:
                raise BadRecord("halmstad_archive_member_count_mismatch")
            summary["archives"].append(archive)
            for video in videos:
                source_class = VIDEO_NAME.fullmatch(video.name)[1]
                if source_class not in allowed_classes:
                    excluded.append(dict(source=str(video), source_class=source_class,
                                         reason="non_drone_clip_excluded_not_negative" if allowed_classes == ("DRONE",)
                                         else "unselected_class_excluded_not_negative"))
                    continue
                digest = sha256(video)
                if digest in seen:
                    excluded.append(dict(source=str(video), reason="duplicate_video_sha256"))
                    continue
                seen.add(digest)
                cap = cv2.VideoCapture(str(video))
                try:
                    values = [cap.get(key) for key in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS,
                                                       cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
                    if not cap.isOpened() or not all(math.isfinite(v) and v > 0 for v in values):
                        raise BadRecord("invalid_halmstad_video_header: " + str(video))
                    n, fps, width, height = values
                    if any(v != int(v) for v in (n, width, height)):
                        raise BadRecord("noninteger_halmstad_video_dimensions")
                finally:
                    cap.release()
                plans.append((video, video.with_name(video.stem + "_LABELS.mat"), digest, int(n), fps, int(width), int(height)))
                estimate += int(n) * (int(width) * int(height) * 3 + 65536)
            summary["source_completeness"]["halmstad"] = dict(videos=len(videos), labels=len(videos), classes=dict(counts))
        summary["excluded_clips"] = excluded
        summary["estimated_output_upper_bytes"] = estimate
        disk.check(estimate)
        save_summary()
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row, bad in pool.map(lambda job: export_dut(job, output, disk), dut_jobs):
                if row:
                    rows.append(row)
                if bad:
                    summary["bad_records"].append(bad)
            for clip_rows, bad in pool.map(lambda plan: export_halmstad(plan, output, disk, allowed_classes), plans):
                rows.extend(clip_rows)
                summary["bad_records"].extend(bad)
        rows.sort(key=lambda row: (row["dataset"], row["split"], row["group"], row.get("original_frame_number", 0)))
        summary["leak_audit"] = audit_manifest(rows)
        content_splits = {}
        for row in rows:
            if row["dataset"] == "dut":
                content_splits.setdefault(row["raw_image_sha256"], set()).add(row["split"])
        summary["dut_cross_split_exact_duplicate_hashes"] = sorted(k for k, splits in content_splits.items() if len(splits) > 1)
        summary["counts"] = {d: {s: {"images": sum(r["dataset"] == d and r["split"] == s for r in rows),
                                      "boxes": sum(r["boxes"] for r in rows if r["dataset"] == d and r["split"] == s)}
                                 for s in SPLITS} for d in sorted(set(datasets))}
        summary["bad_reason_counts"] = dict(Counter(r["reason"] for r in summary["bad_records"]))
        with (output / "manifest.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        summary["manifest_sha256"] = sha256(output / "manifest.jsonl")
        summary["complete"] = True
        save_summary()
        return summary
    except BaseException as exc:
        summary["complete"] = False
        summary["failure"] = type(exc).__name__ + ": " + str(exc)
        save_summary()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--halmstad-decoder-target", type=Path)
    parser.add_argument("--halmstad-classes", nargs="+", choices=HALMSTAD_CLASSES, default=["DRONE"],
                        help="Visible-light positive aircraft classes mapped to class 0; use AIRPLANE HELICOPTER for an incremental export.")
    parser.add_argument("--datasets", nargs="+", choices=("dut", "halmstad"), default=["dut", "halmstad"])
    args = parser.parse_args(argv)
    try:
        summary = prepare(**vars(args))
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"complete": summary["complete"], "output": summary["output"], "counts": summary["counts"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
