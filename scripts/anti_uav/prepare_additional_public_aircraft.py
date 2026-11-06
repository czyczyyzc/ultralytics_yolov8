#!/usr/bin/env python3
"""CPU-only, visible-light AOD4/Purdue ingester; never imports the old converter.

Usage (NEW output only):
  python scripts/anti_uav/prepare_additional_public_aircraft.py \
    --raw-root /mnt/chenziye/datasets/anti_uav/external_rgb/raw \
    --output /path/to/new/additional_aircraft --research-authorized --workers 8
For the audited raw tree use --datasets purdue_uav: AOD4's source-video groups
cross official splits, and no aircraft positives survived strict isolation.
Place official Video_Annotation-v2.zip in purdue_uav/archives (or supply
--purdue-refined-archive). The pinned refined release is preferred; if absent,
--allow-original-purdue-gt is an explicit, documented research-only fallback.
Never fall back on a corrupt, incomplete or mismatched refined archive.

AOD4 v1: https://data.mendeley.com/datasets/cd5z895tr2/1 (CC BY 4.0).
The downloaded COCO export, NOT the paper's inconsistent split prose, defines
train=15761, valid=4514, test=2241. COCO bbox is zero-based xywh. Preserve its
fractional coordinates and every positive size; clip intersecting boundary boxes
without an origin shift and report clipped_boxes. Null/bird/unknown/ignored or
invalid-object frames are excluded whole, not turned into negative examples.
Keep official splits; quarantine ALL members of source groups spanning splits.
YouTube ids and Anti-UAV recording timestamps group video frames conservatively.
V_* names do not prove a unique original video: exclude that entire family.
Only explicit Unsplash/Pexels photo identities are accepted as still sources.
No heuristic about a filename is EVER used to determine an object class.

Purdue: https://engineering.purdue.edu/~bouman/UAV_Dataset/ (BSD-3-Clause).
The already-downloaded ORIGINAL Video_Annotation.zip is pinned below. It is not
the newer refined MOT release, and this distinction is retained in provenance.
Refined v2 is also pinned, contains all 50 matching clip names, and uses MOT
frame,id,left,top,width,height,confidence,-1,-1,-1. Its official renderer proves
zero-based xywh, one-based frame ids, and conf<1 denotes interpolation. Exclude
an entire frame if ANY object is interpolated, unknown or invalid. All missing
MOT frames are excluded, never negatives. Before image-boundary validation its
72131 boxes yield 52566 annotated frames, of which 732 contain interpolation;
51834 non-interpolated frames are candidates, not a promise of exported counts.
Read-only parser audit using all 50 actual video dimensions retained 51834
frames/70670 boxes (863 clipped), excluding 17684 unannotated and 732 interpolated
frames. Full sequential runtime decode is still required before completion.
Official Video_Annotation-v2.zip (SHA256 below) includes render_gt_video.py:
load_raw interprets (y1,x1,y2,x2) as (x1,y1,x2-x1,y2-y1), with NO origin shift
or inclusive +1. render starts frame_no=0 then increments BEFORE looking up GT.
This proves raw GT line time_layer=1 maps to the first decoded frame (index 0).
Renderer SHA256: 05bb7c8ed8bfff7c8224f87a271f6ac88cf97f0d30854db5473bcb280ab4e86f.
v2 archive SHA256: 61581df9426c1827acfa8dfc1e4f930dfa265d9b6c2dcbfef0993f9d82f388d6.
Read-only audit found 50 clips, 70250 matching GT rows, 52306 positive frames,
71432 boxes, 17944 empty frames; continuous first-frame overlays checked on
Clips 1,10,50. Runtime requires consecutive GT ids, header agreement AND a full
sequential decode to EOF. Missing row/count disagreement fails the run closed.
No official split is published for these archives: custom whole-video holdout,
val iff SHA256('public-uav-research-v1:' + raw_video_sha256) < 2**256 // 5.
Identical raw video SHA groups cannot cross splits. Never seek/subsample frames.

Consumers MUST verify parent summary.complete and manifest_sha256. This output
is internal research data, not a claim of commercial clearance or an independent
benchmark against datasets reused by AOD4. Raw inputs are read-only. No downloads,
archive extraction, training, IR, or edits to Halmstad/native data are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import threading
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path, PurePosixPath


SPLITS = ("train", "val", "test")
AOD_COUNTS = {"train": 15761, "valid": 4514, "test": 2241}
AOD_CATEGORIES = {0: "airplane-helicopter-drone-bird", 1: "airplane", 2: "bird", 3: "drone", 4: "helicopter"}
AIRCRAFT = {"airplane", "drone", "helicopter"}
PURDUE_GT_SHA = "abc9110cd287d76d2c6ed70f1163386d881bd567e39930431d393a2b383271b1"
PURDUE_V2_SHA = "61581df9426c1827acfa8dfc1e4f930dfa265d9b6c2dcbfef0993f9d82f388d6"
PURDUE_URL = "https://engineering.purdue.edu/~bouman/UAV_Dataset/"
RAW_BOX = re.compile(r"\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\),\s*")
GIB = 1024**3
BSD_NOTICE = """BSD 3-Clause License
Copyright (c) 2022, Charles A. Bouman
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


class UnsafeData(ValueError):
    """Unproven data must never become an empty-label negative."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(digest):
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise UnsafeData("invalid_raw_video_sha256")
    score = int(hashlib.sha256(("public-uav-research-v1:" + digest).encode()).hexdigest(), 16)
    return "val" if score < (1 << 256) // 5 else "train"


def box_xyxy(values, width, height):
    if len(values) != 4 or any(type(v) not in (float, int) for v in values):
        raise UnsafeData("invalid_box_values")
    x1, y1, x2, y2 = map(float, values)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise UnsafeData("nonfinite_box")
    if x2 <= x1 or y2 <= y1:
        raise UnsafeData("degenerate_box")
    clipped = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise UnsafeData("box_has_no_image_intersection")
    return clipped


def yolo_line(box, width, height):
    x1, y1, x2, y2 = box
    return "0 " + " ".join(format(v, ".17g") for v in (
        (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
        (x2 - x1) / width, (y2 - y1) / height))


def safe_child(base, name):
    if not isinstance(name, str):
        raise UnsafeData("unsafe_source_filename")
    p = PurePosixPath(name)
    if len(p.parts) != 1 or p.name != name or "\\" in name or name in (".", ".."):
        raise UnsafeData("unsafe_source_filename")
    result = base / name
    if not result.is_file() or not result.resolve().is_relative_to(base.resolve()):
        raise UnsafeData("missing_or_escaping_source_image")
    return result


@lru_cache(maxsize=8)
def archive_index(archive):
    """Index once: AOD4 has 90k entries, so reopening per image is prohibitive."""
    with zipfile.ZipFile(archive) as z:
        index = {}
        for info in z.infolist():
            if info.filename in index or info.flag_bits & 1:
                raise UnsafeData("duplicate_or_encrypted_archive_member: " + info.filename)
            index[info.filename] = info
    return index


def verified_member(archive, member, path):
    """Bind existing extracted files to archive CRC+size without extracting."""
    try:
        info = archive_index(archive)[member]
    except KeyError as exc:
        raise UnsafeData("missing_archive_member: " + member) from exc
    crc, size = 0, 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            crc = zlib.crc32(block, crc)
            size += len(block)
            digest.update(block)
    if size != info.file_size or crc & 0xFFFFFFFF != info.CRC:
        raise UnsafeData("archive_member_mismatch: " + str(path))
    return dict(member=member, crc32=f"{info.CRC:08x}", size=size, sha256=digest.hexdigest())


class DiskGuard:
    def __init__(self, output, reserve_gib=40):
        if not math.isfinite(reserve_gib) or reserve_gib < 40:
            raise ValueError("reserve_gib must be >= 40")
        self.output, self.reserve = output, math.ceil(reserve_gib * GIB)
        self.lock = threading.Lock()

    def check(self, size=0):
        if shutil.disk_usage(self.output).free < self.reserve + size + 1024**2:
            raise UnsafeData("insufficient_disk_reserve_40GiB_minimum")

    def write(self, path, data):
        # Serialize actual writes, not JPEG encoding, to protect the reserve
        # from concurrent workers without a pessimistic full-RGB disk estimate.
        with self.lock:
            self.check(len(data) + 4096)
            with path.open("xb") as stream:
                stream.write(data)

    def link(self, path, source):
        with self.lock:
            self.check(4096)
            path.symlink_to(source)


def aod_source_group(info):
    name = info["file_name"].split(".rf.")[0]
    tags = info.get("extra", {}).get("user_tags", [])
    if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
        raise UnsafeData("ambiguous_source_tags")
    yt = re.fullmatch(r"yt-([A-Za-z0-9_-]{11})-\d+_jpg", name)
    if yt:
        if tags and set(tags) != {"youtube:" + yt[1]}:
            raise UnsafeData("source_tag_filename_disagreement")
        return "youtube:" + yt[1]
    if tags:
        raise UnsafeData("unresolved_source_tags")
    anti = re.fullmatch(r"(20\d{6}_\d{6})_\d+_\d+_\d+_jpg", name)
    if anti:
        return "anti-uav-recording:" + anti[1]
    if name.startswith("V_"):
        raise UnsafeData("unproven_original_video_identity")
    if re.fullmatch(r"[A-Za-z0-9_-]+-unsplash_jpg", name) or re.fullmatch(r"pexels-[A-Za-z0-9_-]+-\d+_jpg", name):
        return "photo:" + name
    raise UnsafeData("unresolved_source_identity")


def aod_boxes(annotations, categories, width, height):
    boxes, names = [], []
    if not annotations:
        raise UnsafeData("empty_annotation_excluded_not_negative")
    for ann in annotations:
        name = categories.get(ann.get("category_id"))
        if name not in AIRCRAFT:
            raise UnsafeData("bird_or_unknown_class_excluded_whole_frame")
        if ann.get("iscrowd") != 0 or any(ann.get(k, 0) != 0 for k in ("ignore", "ignored", "difficult", "is_ignore")):
            raise UnsafeData("ignored_or_missing_annotation_flag")
        values = ann.get("bbox", [])
        if len(values) != 4 or any(type(v) not in (float, int) for v in values):
            raise UnsafeData("invalid_coco_xywh")
        x, y, w, h = values
        if w <= 0 or h <= 0:
            raise UnsafeData("invalid_coco_xywh")
        boxes.append(box_xyxy((x, y, x + w, y + h), width, height))
        names.append(name)
    return boxes, names


def purge_split_conflicts(plans):
    """Drop whole source groups, including groups connected by identical bytes."""
    groups, content_groups = defaultdict(set), defaultdict(set)
    for p in plans:
        groups[p["group"]].add(p["split"])
        content_groups[p["raw_image_sha256"]].add(p["group"])
    bad = {g for g, splits in groups.items() if len(splits) > 1}
    # A duplicate connects a source family even when only a subset of it overlaps.
    changed = True
    while changed:
        changed = False
        for linked in content_groups.values():
            splits = set().union(*(groups[g] for g in linked))
            if len(splits) > 1 or linked & bad:
                if not linked <= bad:
                    bad.update(linked)
                    changed = True
    return [p for p in plans if p["group"] not in bad], [
        dict(source=str(p["source"]), reason="cross_split_source_or_duplicate_group", group=p["group"])
        for p in plans if p["group"] in bad]


def plan_aod(raw, workers, excluded, provenance):
    import cv2

    root = raw / "aod4/extracted/aod4_inner/AOD 4"
    archive = raw / "aod4/extracted/AOD 4 Dataset for Air Borne Object Detection/AOD 4.zip"
    provenance.append(dict(dataset="aod4", archive=str(archive), sha256=sha256(archive),
                           origin="https://data.mendeley.com/datasets/cd5z895tr2/1", license="CC BY 4.0",
                           attribution="Vama Soni, Dhruval Shah, Jeel Joshi; AOD4 v1, DOI 10.17632/cd5z895tr2.1"))
    jobs, identities = [], []
    for source_split, count in AOD_COUNTS.items():
        path = root / "Annotations/COCO Annotation format" / source_split / "_annotations.coco.json"
        verified_member(archive, "AOD 4/" + str(path.relative_to(root)), path)
        data = json.loads(path.read_text())
        cats = {c["id"]: c["name"] for c in data["categories"]}
        if cats != AOD_CATEGORIES or len(cats) != len(data["categories"]):
            raise UnsafeData("unverified_aod_categories")
        images = data["images"]
        if len(images) != count or len({i["id"] for i in images}) != count:
            raise UnsafeData("aod_official_split_incomplete_or_duplicate_ids")
        image_dir = root / "Images" / source_split
        if {p.name for p in image_dir.iterdir() if p.is_file()} != {i["file_name"] for i in images}:
            raise UnsafeData("aod_image_annotation_inventory_mismatch")
        by_image, ids = defaultdict(list), set()
        valid_ids = {i["id"] for i in images}
        for ann in data["annotations"]:
            if ann["id"] in ids or ann["image_id"] not in valid_ids:
                raise UnsafeData("duplicate_or_orphan_aod_annotation")
            ids.add(ann["id"])
            by_image[ann["image_id"]].append(ann)
        provenance.append(dict(dataset="aod4", split=source_split, annotation=str(path),
                               annotation_sha256=sha256(path), images=count, annotations=len(ids)))
        for info in images:
            group = None
            try:
                group = aod_source_group(info)
            except (UnsafeData, TypeError, AttributeError) as exc:
                excluded.append(dict(source=str(image_dir / str(info.get("file_name"))), reason=str(exc)))
            source = safe_child(image_dir, info["file_name"])
            identities.append(dict(source=source, group=group or "unresolved:" + str(source),
                                   split="val" if source_split == "valid" else source_split))
            if group is not None:
                jobs.append((source_split, info, by_image[info["id"]], image_dir, cats, group))

    def identity_hash(p):
        return dict(p, raw_image_sha256=sha256(p["source"]))

    # Include EVERY raw image in the duplicate audit, even a bird, unknown or
    # already excluded video frame in an official holdout. Filtering first
    # could otherwise retain an exact copy under a different source identity.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        identities = list(pool.map(identity_hash, identities))
    identities, bad = purge_split_conflicts(identities)
    excluded.extend(bad)
    accepted_sources = {str(p["source"]) for p in identities}
    kept_jobs = []
    for job in jobs:
        if str(job[3] / job[1]["file_name"]) in accepted_sources:
            kept_jobs.append(job)

    def inspect(job):
        split, info, anns, base, cats, group = job
        try:
            image = safe_child(base, info["file_name"])
            width, height = info["width"], info["height"]
            if type(width) is not int or type(height) is not int or min(width, height) <= 0:
                raise UnsafeData("invalid_image_dimensions")
            # Audit groups before filtering objects so an excluded bird/empty
            # frame in a holdout still prevents same-source train leakage.
            member = verified_member(archive, "AOD 4/" + str(image.relative_to(root)), image)
            pixels = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if pixels is None or pixels.ndim != 3 or pixels.shape != (height, width, 3):
                raise UnsafeData("not_verified_visible_rgb_image")
            try:
                boxes, names = aod_boxes(anns, cats, width, height)
                reason = None
            except (UnsafeData, TypeError, ValueError) as exc:
                boxes, names, reason = [], [], str(exc)
            return dict(dataset="aod4", split="val" if split == "valid" else split, group=group,
                        source=image, width=width, height=height, xyxy=boxes, aircraft_types=names,
                        clipped_boxes=sum(b != (a["bbox"][0], a["bbox"][1],
                                                a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3])
                                          for b, a in zip(boxes, anns)),
                        annotation_image_id=info["id"], raw_image_sha256=member["sha256"],
                        archive_member=member, exclusion=reason), None
        except (UnsafeData, OSError, TypeError, ValueError) as exc:
            return None, dict(source=str(base / str(info.get("file_name"))), reason=str(exc))

    plans = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for p, bad in pool.map(inspect, kept_jobs):
            if p:
                plans.append(p)
            if bad:
                excluded.append(bad)
    plans, bad = purge_split_conflicts(plans)
    excluded.extend(bad)
    good = []
    for p in plans:
        reason = p.pop("exclusion")
        if reason:
            excluded.append(dict(source=str(p["source"]), reason=reason))
        else:
            good.append(p)
    return good


def parse_purdue_gt(path, count, width, height):
    lines = path.read_text().splitlines()
    if len(lines) != count:
        raise UnsafeData("purdue_gt_video_count_alignment_mismatch")
    frames, excluded = [], []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"time_layer:\s*(\d+)\s+detections:\s*(.*)", line)
        if not match or int(match[1]) != index + 1:
            raise UnsafeData("purdue_missing_duplicate_or_nonconsecutive_frame_id")
        try:
            payload = match[2]
            if RAW_BOX.sub("", payload).strip():
                raise UnsafeData("purdue_partial_or_unknown_gt_payload")
            boxes = [box_xyxy((int(x), int(y), int(xx), int(yy)), width, height)
                     for y, x, yy, xx in RAW_BOX.findall(payload)]
            if not boxes:
                raise UnsafeData("empty_annotation_excluded_not_negative")
            frames.append(boxes)
        except UnsafeData as exc:
            frames.append(None)
            excluded.append(dict(source=str(path), frame_index=index, original_frame_number=index + 1, reason=str(exc)))
    return frames, excluded


def parse_purdue_refined(text, count, width, height, source):
    """Official v2 MOT: frame,id,left,top,width,height,conf,-1,-1,-1.

    Renderer establishes zero-based xywh, one-based video frame ids. A frame
    containing ANY interpolated/confidence<1 or malformed object is excluded
    whole, including its otherwise valid objects. Missing rows are not negatives.
    """
    frames, clipped = [[] for _ in range(count)], [0] * count
    bad_frames, seen = {}, set()
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        try:
            frame_number = int(fields[0])
        except (ValueError, IndexError) as exc:
            raise UnsafeData("unproven_refined_frame_identity") from exc
        if not 1 <= frame_number <= count:
            raise UnsafeData("refined_frame_outside_matched_video")
        index = frame_number - 1
        try:
            if len(fields) != 10:
                raise UnsafeData("invalid_refined_mot_columns")
            track = int(fields[1])
            x, y, w, h, conf = map(float, fields[2:7])
            if track < 1 or (index, track) in seen or any(float(v) != -1 for v in fields[7:]):
                raise UnsafeData("unknown_or_duplicate_refined_object")
            seen.add((index, track))
            if conf != 1.0:
                raise UnsafeData("refined_interpolated_or_unknown_confidence_frame")
            raw_box = (x, y, x + w, y + h)
            box = box_xyxy(raw_box, width, height)
            frames[index].append(box)
            clipped[index] += box != raw_box
        except (ValueError, TypeError) as exc:
            bad_frames[index] = str(exc)
    excluded = []
    for index, boxes in enumerate(frames):
        if index in bad_frames or not boxes:
            frames[index], clipped[index] = None, 0
            excluded.append(dict(source=str(source), frame_index=index, original_frame_number=index + 1,
                                 reason=bad_frames.get(index, "empty_annotation_excluded_not_negative")))
    return frames, excluded, clipped


def plan_purdue(raw, workers, excluded, provenance, refined_archive=None, allow_original=False):
    import cv2

    root = raw / "purdue_uav"
    refined_archive = Path(refined_archive).resolve() if refined_archive else root / "archives/Video_Annotation-v2.zip"
    refined = refined_archive.is_file()
    if not refined and not allow_original:
        raise UnsafeData("preferred_purdue_v2_not_local: provide --purdue-refined-archive or explicitly --allow-original-purdue-gt")
    if refined and sha256(refined_archive) != PURDUE_V2_SHA:
        raise UnsafeData("purdue_v2_sha256_mismatch_no_silent_fallback")
    refined_labels = {}
    if refined:
        expected = {f"Video_Annotation-v2/refined_gt/Clip_{i}_refined.txt" for i in range(1, 51)}
        with zipfile.ZipFile(refined_archive) as z:
            members = [n for n in z.namelist() if n.startswith("Video_Annotation-v2/refined_gt/") and n.endswith(".txt")]
            if set(members) != expected or len(members) != 50:
                raise UnsafeData("purdue_v2_not_complete_50_matched_clips")
            refined_labels = {n: z.read(n) for n in members}
    gt_archive, videos_archive = (root / "archives" / n for n in ("Video_Annotation.zip", "Videos.zip"))
    if sha256(gt_archive) != PURDUE_GT_SHA:
        raise UnsafeData("purdue_original_official_annotation_archive_sha256_mismatch")
    names = {f"Clip_{i}.mov" for i in range(1, 51)}
    if {p.name for p in (root / "extracted/Videos").iterdir() if p.is_file()} != names:
        raise UnsafeData("purdue_video_inventory_not_official_50")
    expected_gt = {n.replace(".mov", "_gt.txt") for n in names}
    if {p.name for p in (root / "extracted/Video_Annotation").glob("*.txt")} != expected_gt:
        raise UnsafeData("purdue_gt_inventory_not_official_50")
    provenance.append(dict(dataset="purdue_uav", origin=PURDUE_URL,
                           annotation_release="refined_v2" if refined else "original_not_refined_v2",
                           original_fallback_reason=None if refined else "refined_release_unavailable_locally; explicit_operator_override",
                           refined_archive=str(refined_archive) if refined else None,
                           refined_archive_sha256=PURDUE_V2_SHA if refined else None,
                           annotation_archive=str(gt_archive), video_archive=str(videos_archive),
                           annotation_archive_sha256=PURDUE_GT_SHA, video_archive_sha256=sha256(videos_archive),
                           license="BSD-3-Clause", license_notice=BSD_NOTICE,
                           license_url=PURDUE_URL + "pubs/LICENSE.txt",
                           alignment_evidence=PURDUE_URL + "Video_Annotation-v2.zip#render_gt_video.py",
                           alignment_evidence_archive_sha256="61581df9426c1827acfa8dfc1e4f930dfa265d9b6c2dcbfef0993f9d82f388d6",
                           alignment_renderer_sha256="05bb7c8ed8bfff7c8224f87a271f6ac88cf97f0d30854db5473bcb280ab4e86f",
                           coordinate_convention="zero_based_xywh_size_is_difference" if refined else "zero_based_ymin_xmin_ymax_xmax_size_is_difference",
                           frame_alignment="GT time_layer=decoded_frame_index+1; full sequential EOF count required"))

    def inspect(name):
        video = root / "extracted/Videos" / name
        gt = root / "extracted/Video_Annotation" / (video.stem + "_gt.txt")
        member = verified_member(videos_archive, "Videos/" + name, video)
        verified_member(gt_archive, "Video_Annotation/" + gt.name, gt)
        digest = member["sha256"]
        cap = cv2.VideoCapture(str(video))
        try:
            n, width, height, fps = [cap.get(k) for k in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH,
                                                        cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)]
            if not cap.isOpened() or not all(math.isfinite(v) and v > 0 for v in (n, width, height, fps)):
                raise UnsafeData("invalid_purdue_video_header")
            if any(v != int(v) for v in (n, width, height)):
                raise UnsafeData("noninteger_purdue_video_header")
        finally:
            cap.release()
        frames, bad = parse_purdue_gt(gt, int(n), int(width), int(height))
        clipped = [0] * int(n)
        if refined:
            member_name = "Video_Annotation-v2/refined_gt/" + video.stem + "_refined.txt"
            data = refined_labels[member_name]
            frames, bad, clipped = parse_purdue_refined(data.decode("utf-8"), int(n), int(width), int(height),
                                                       str(refined_archive) + "::" + member_name)
            annotation = dict(annotation=str(refined_archive) + "::" + member_name,
                              annotation_sha256=hashlib.sha256(data).hexdigest(),
                              annotation_archive=refined_archive, annotation_archive_sha256=PURDUE_V2_SHA,
                              annotation_rows=len(data.splitlines()))
        else:
            raw_lines = gt.read_text().splitlines()
            for index, boxes in enumerate(frames):
                if boxes:
                    values = RAW_BOX.findall(raw_lines[index].split("detections:", 1)[1])
                    clipped[index] = sum(b != (int(x), int(y), int(xx), int(yy))
                                         for b, (y, x, yy, xx) in zip(boxes, values))
            annotation = dict(annotation=gt, annotation_sha256=sha256(gt), annotation_rows=len(raw_lines))
        return dict(dataset="purdue_uav", source=video, **annotation,
                    annotation_release="refined_v2" if refined else "original_not_refined_v2",
                    coordinate_convention="zero_based_xywh_size_is_difference" if refined else "zero_based_ymin_xmin_ymax_xmax_size_is_difference",
                    clipped_boxes_by_frame=clipped,
                    raw_video_sha256=digest, group="sha256:" + digest, split=stable_split(digest),
                    width=int(width), height=int(height), fps=fps, frame_count=int(n), frames=frames), bad

    plans, seen, total_frames = [], set(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for p, bad in pool.map(inspect, sorted(names)):
            excluded.extend(bad)
            total_frames += p["frame_count"]
            if p["raw_video_sha256"] in seen:
                excluded.append(dict(source=str(p["source"]), reason="duplicate_raw_video_sha256"))
            else:
                seen.add(p["raw_video_sha256"])
                plans.append(p)
    if total_frames != 70250:
        raise UnsafeData("purdue_total_frame_inventory_mismatch")
    return plans


def emit_frame(plan, output, disk, boxes, pixels=None, frame_index=None):
    if not boxes:
        raise UnsafeData("empty_labels_forbidden")
    stem = plan["dataset"] + "_" + hashlib.sha256(str(plan["source"]).encode()).hexdigest()
    if frame_index is not None:
        stem += f"_{frame_index:08d}"
    suffix = plan["source"].suffix.lower() if pixels is None else ".jpg"
    image = output / "images" / plan["split"] / (stem + suffix)
    label = output / "labels" / plan["split"] / (stem + ".txt")
    if pixels is None:
        disk.link(image, plan["source"])
    else:
        import cv2

        ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise UnsafeData("jpeg_encode_failed")
        disk.write(image, encoded.tobytes())
    disk.write(label, ("\n".join(yolo_line(b, plan["width"], plan["height"]) for b in boxes) + "\n").encode())
    row = {k: str(v) if isinstance(v, Path) else v for k, v in plan.items()
           if k not in {"xyxy", "frames", "frame_count", "clipped_boxes_by_frame"}}
    row.update(image=str(image), label=str(label), boxes=len(boxes))
    if frame_index is not None:
        row.update(frame_index=frame_index, original_frame_number=frame_index + 1,
                   gt_time_layer=frame_index + 1, jpeg_quality=95, aircraft_types=["drone"] * len(boxes),
                   coordinate_convention=plan.get("coordinate_convention", "zero_based_ymin_xmin_ymax_xmax_size_is_difference"))
    return row


def export_purdue(plan, output, disk):
    import cv2

    cap = cv2.VideoCapture(str(plan["source"]))
    rows, index = [], 0
    try:
        if not cap.isOpened():
            raise UnsafeData("purdue_decoder_open_failed")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= len(plan["frames"]) or frame.shape != (plan["height"], plan["width"], 3):
                raise UnsafeData("purdue_sequential_decode_alignment_mismatch")
            if plan["frames"][index]:
                row = emit_frame(plan, output, disk, plan["frames"][index], frame, index)
                row["clipped_boxes"] = plan["clipped_boxes_by_frame"][index] if "clipped_boxes_by_frame" in plan else 0
                rows.append(row)
            index += 1
        if index != plan["frame_count"] or sha256(plan["source"]) != plan["raw_video_sha256"]:
            raise UnsafeData("purdue_incomplete_decode_or_source_changed")
        annotation = plan.get("annotation_archive", plan["annotation"])
        expected = plan.get("annotation_archive_sha256", plan["annotation_sha256"])
        if sha256(annotation) != expected:
            raise UnsafeData("purdue_gt_changed")
    finally:
        cap.release()
    return rows, dict(source=str(plan["source"]), raw_video_sha256=plan["raw_video_sha256"],
                      gt_timeline_frames=len(plan["frames"]), annotation_rows=plan.get("annotation_rows", len(plan["frames"])),
                      annotation_release=plan.get("annotation_release", "original_not_refined_v2"), sequential_decoded_frames=index,
                      positive_frames=len(rows), alignment_verified=True)


def audit_manifest(rows):
    groups, content, paths = {}, {}, set()
    for row in rows:
        if row["split"] not in SPLITS or type(row["boxes"]) is not int or row["boxes"] <= 0:
            raise UnsafeData("invalid_split_or_nonpositive_manifest_row")
        for table, key in ((groups, (row["dataset"], row["group"])),
                           (content, row.get("raw_image_sha256", row.get("raw_video_sha256")))):
            if key in table and table[key] != row["split"]:
                raise UnsafeData("cross_split_leak")
            table[key] = row["split"]
        for key in ("image", "label"):
            p = Path(row[key])
            if not p.is_absolute() or not p.is_file() or row[key] in paths:
                raise UnsafeData("invalid_or_duplicate_output_path")
            paths.add(row[key])
        lines = Path(row["label"]).read_text().splitlines()
        if len(lines) != row["boxes"] or any(line.split()[0] != "0" for line in lines):
            raise UnsafeData("manifest_label_mismatch")
    return dict(cross_split_group_overlap=0, cross_split_raw_hash_overlap=0, negative_frames=0)


def prepare(raw_root, output, datasets=("aod4", "purdue_uav"), workers=8, reserve_gib=40, research_authorized=False,
            purdue_refined_archive=None, allow_original_purdue_gt=False):
    if not research_authorized:
        raise UnsafeData("explicit_--research-authorized_required")
    if workers not in (8, 16) or not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= {"aod4", "purdue_uav"}:
        raise ValueError("use unique supported datasets and workers 8 or 16")
    if not math.isfinite(reserve_gib) or reserve_gib < 40:
        raise ValueError("reserve_gib must be >= 40")
    raw, output = Path(raw_root).resolve(), Path(output).resolve()
    if output.is_relative_to(raw) or raw.is_relative_to(output):
        raise UnsafeData("output_must_not_overlap_raw_root")
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(complete=False, output=str(output), raw_root=str(raw), datasets=list(datasets),
                   visible_only=True, research_authorized=True, workers=workers, reserve_gib=reserve_gib,
                   names={"0": "drone"}, class_0_semantics=sorted(AIRCRAFT),
                   official_splits="AOD4 directory train/valid/test -> train/val/test; no reassignment",
                   custom_split="Purdue only: SHA256(public-uav-research-v1:+raw_video_sha256) < 2**256//5 => val, else train",
                   limitations=["AOD4 reuses other datasets; not an independent cross-dataset benchmark",
                                "AOD4 ambiguous/video-family and cross-split groups excluded, not reassigned",
                                "Purdue refined v2 preferred; original fallback requires explicit operator flag; positive-only export"],
                   provenance=[], excluded_records=[], video_alignment=[])

    def save():
        temp = output / "summary.json.tmp"
        content = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode()
        if shutil.disk_usage(output).free < reserve_gib * GIB + len(content) + 4096:
            raise UnsafeData("insufficient_disk_for_summary_and_reserve")
        temp.write_bytes(content)
        temp.replace(output / "summary.json")

    save()
    try:
        import cv2

        cv2.setNumThreads(1)
        disk = DiskGuard(output, reserve_gib)
        disk.check()
        for split in SPLITS:
            for kind in ("images", "labels"):
                (output / kind / split).mkdir(parents=True)
        excluded, provenance = summary["excluded_records"], summary["provenance"]
        aod = plan_aod(raw, workers, excluded, provenance) if "aod4" in datasets else []
        if "aod4" in datasets and not aod:
            raise UnsafeData("aod4_blocked_no_leak_free_positive_frames; run --datasets purdue_uav independently")
        purdue = plan_purdue(raw, workers, excluded, provenance, purdue_refined_archive,
                             allow_original_purdue_gt) if "purdue_uav" in datasets else []
        save()
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows.extend(pool.map(lambda p: emit_frame(p, output, disk, p["xyxy"]), aod))
            for clip_rows, alignment in pool.map(lambda p: export_purdue(p, output, disk), purdue):
                rows.extend(clip_rows)
                summary["video_alignment"].append(alignment)
        if not rows or any(not any(r["dataset"] == d for r in rows) for d in datasets):
            raise UnsafeData("requested_dataset_has_no_verified_positive_frames")
        rows.sort(key=lambda r: (r["dataset"], r["split"], r["group"], r.get("frame_index", -1), r["source"]))
        summary["leak_audit"] = audit_manifest(rows)
        summary["counts"] = {d: {s: dict(images=sum(r["dataset"] == d and r["split"] == s for r in rows),
                                           boxes=sum(r["boxes"] for r in rows if r["dataset"] == d and r["split"] == s))
                                     for s in SPLITS} for d in datasets}
        summary["aircraft_type_boxes"] = dict(Counter(t for r in rows for t in r["aircraft_types"]))
        summary["clipped_boxes"] = sum(r.get("clipped_boxes", 0) for r in rows)
        summary["exclusion_counts"] = dict(Counter(r["reason"] for r in excluded))
        manifest = "".join(json.dumps(r, allow_nan=False, sort_keys=True) + "\n" for r in rows)
        disk.write(output / "manifest.jsonl", manifest.encode())
        disk.write(output / "data.yaml", (f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: drone\n").encode())
        summary.update(manifest_sha256=sha256(output / "manifest.jsonl"), complete=True)
        disk.check()
        save()
        return summary
    except BaseException as exc:
        summary.update(complete=False, failure=type(exc).__name__ + ": " + str(exc))
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("aod4", "purdue_uav"), default=["aod4", "purdue_uav"])
    parser.add_argument("--workers", type=int, choices=(8, 16), default=8)
    parser.add_argument("--reserve-gib", type=float, default=40)
    parser.add_argument("--research-authorized", action="store_true")
    parser.add_argument("--purdue-refined-archive", type=Path, help="Preferred pinned official Video_Annotation-v2.zip; read in place, no extraction")
    parser.add_argument("--allow-original-purdue-gt", action="store_true", help="Explicit research fallback ONLY when refined archive is absent locally")
    args = parser.parse_args(argv)
    try:
        summary = prepare(**vars(args))
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps({k: summary[k] for k in ("complete", "output", "counts", "manifest_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
