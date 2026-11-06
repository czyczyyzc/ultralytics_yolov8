#!/usr/bin/env python3
"""Export only audited missing reviewed negatives; never change the native curriculum."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
from threading import Event

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.prepare_labeler_training_expansion import frame_plan

MISSING_SCHEMA = "anti_uav.missing_approved_frames.v1"
GIB = 1 << 30


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    payload = Path(path).read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def metadata(video):
    return dict(task_id=video["task_id"], source_sha256=video["source_sha256"],
                source=str(Path(video["video"]).resolve()), width=video["frame_width"],
                height=video["frame_height"], status=video["status"], revision=video["revision"],
                approved_revision=video["approved_revision"])


def load_plan(audit_path, snapshot_path):
    """Recompute the exact (source SHA, zero-based frame) complement of the pinned cache."""
    import numpy as np

    audit_path, snapshot_path = Path(audit_path).resolve(), Path(snapshot_path).resolve()
    audit, audit_hash = read_json(audit_path)
    snapshot, snapshot_hash = read_json(snapshot_path)
    if (audit.get("complete") is not True or audit.get("coverage_verified") is not True
            or audit.get("all_current_approved_training_sources_covered") is not True):
        raise ValueError("Approved audit is incomplete or has unverified source coverage")
    if snapshot_hash != audit["snapshot_sha256"]:
        raise ValueError("Approved snapshot SHA256 changed")
    source = Path(audit["source_dataset"]).resolve()
    train_path, cache_path = source / "train_hardneg.txt", source / "train_hardneg.cache"
    if sha256(train_path) != audit["source_train_sha256"]:
        raise ValueError("Source train list SHA256 changed")
    payload = cache_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != audit["source_cache_sha256"]:
        raise ValueError("Source label cache SHA256 changed")
    # This pickle is the trusted, cryptographically pinned cache produced by our pipeline.
    cache = np.load(io.BytesIO(payload), allow_pickle=True).item()
    del payload
    videos = snapshot["videos"]
    by_sha = {v["source_sha256"]: v for v in videos}
    audited = {v["task_id"]: v for v in audit["videos"]}
    heldout = set(snapshot["heldout_sha256"]) | set(audit["heldout_sha256"])
    if (len(by_sha) != len(videos) or len(videos) != audit["approved_training_videos"]
            or len(audited) != len(videos) or {v["task_id"] for v in videos} != set(audited)
            or heldout & set(by_sha) or snapshot.get("frame_index_base") != 0):
        raise ValueError("Approved source scope, frame index base, or holdout isolation changed")
    for video in videos:
        if (not re.fullmatch(r"[0-9a-f]{64}", video["source_sha256"])
                or video["status"] != "approved" or type(video["revision"]) is not int
                or video["revision"] != video["approved_revision"]
                or any(type(video[k]) is not int or video[k] <= 0
                       for k in ("frame_width", "frame_height", "frame_count"))):
            raise ValueError("Invalid source geometry, SHA256, or approved revision")
    groups = defaultdict(set)
    for row in cache["labels"]:
        path = Path(row["im_file"])
        groups[str(path.parent)].add(path.stem)
    seen, mapped = defaultdict(set), defaultdict(set)
    for parent, stems in groups.items():
        name = Path(parent).name
        candidates = [s for s, v in by_sha.items() if name == s or name.endswith("_" + s[:12])
                      or (name == Path(v["source_relpath"]).stem and name.endswith("_" + s[:8]))]
        if len(candidates) > 1:
            raise ValueError("Ambiguous cache source group")
        if candidates:
            digest = candidates[0]
            if any(not stem.isdigit() for stem in stems):
                raise ValueError("Nonnumeric source frame filename")
            seen[digest].update(int(stem) for stem in stems)
            mapped[digest].add(parent)
    jobs = []
    eligible_positive = eligible_negative = 0
    for video in videos:
        digest = video["source_sha256"]
        detail = audited[video["task_id"]]
        if (detail["source_sha256"] != digest or set(detail["cache_image_groups"]) != mapped[digest]
                or not mapped[digest] or detail.get("errors")):
            raise ValueError("Source/cache group map differs from the completed audit")
        plan, _ = frame_plan(video["annotations"], video["frame_count"], video["frame_width"],
                             video["frame_height"], True)
        eligible_positive += sum(bool(boxes) for boxes in plan.values())
        eligible_negative += sum(not boxes for boxes in plan.values())
        if not seen[digest] <= plan.keys():
            raise ValueError("Cached source contains a currently excluded or out-of-range frame")
        missing = sorted(plan.keys() - seen[digest])
        if len(missing) != detail["current_frames_not_in_cache"]:
            raise ValueError("Missing-frame count differs from the completed audit")
        statuses = defaultdict(set)
        for annotation in video["annotations"]:
            statuses[int(annotation["frame_index"])].add(annotation["status"])
        if any(plan[index] != [] or statuses[index] != {"absent"} for index in missing):
            raise ValueError("Missing frames include positive, uncertain, invalid, or unlabelled annotations")
        if missing:
            jobs.append(dict(video=video, frame_indices=missing))
    count = sum(len(job["frame_indices"]) for job in jobs)
    if not count or count != audit["current_frames_not_in_cache"]:
        raise ValueError("No missing reviewed negatives or inconsistent aggregate missing count")
    pins = {str(audit_path): audit_hash, str(snapshot_path): snapshot_hash,
            str(train_path): audit["source_train_sha256"], str(cache_path): audit["source_cache_sha256"]}
    return dict(jobs=jobs, pins=pins, source_dataset=str(source), audit_path=str(audit_path),
                approved_audit_sha256=audit_hash, snapshot_path=str(snapshot_path), snapshot_sha256=snapshot_hash,
                source_train_sha256=audit["source_train_sha256"], source_cache_sha256=audit["source_cache_sha256"],
                missing_frames=count, eligible_positive_frames=eligible_positive,
                eligible_negative_frames=eligible_negative, heldout_sha256=sorted(heldout))


def missing_contract(plan):
    """Frozen auditmissing_frames.json: full per-frame metadata plus source summaries."""
    sources, frames = [], []
    for job in plan["jobs"]:
        video, indices = job["video"], job["frame_indices"]
        meta = metadata(video)
        annotations = defaultdict(list)
        for row in video["annotations"]:
            annotations[int(row["frame_index"])].append(row)
        sources.append(dict(meta, name=video["name"], source_relpath=video["source_relpath"],
                            fps=video["fps"], frame_count=video["frame_count"],
                            missing_positive_frames=0, missing_negative_frames=len(indices)))
        frames.extend(dict(meta, frame_index=index, boxes=[], reviewed_status="absent",
                           annotations=annotations[index]) for index in indices)
    return dict(schema=MISSING_SCHEMA, complete=True, approved_audit_path=plan["audit_path"],
                approved_audit_sha256=plan["approved_audit_sha256"], source_dataset=plan["source_dataset"],
                source_train_sha256=plan["source_train_sha256"], source_cache_sha256=plan["source_cache_sha256"],
                snapshot_path=plan["snapshot_path"], snapshot_sha256=plan["snapshot_sha256"],
                source_snapshot_sha256=plan["snapshot_sha256"], heldout_sha256=plan["heldout_sha256"],
                frame_index_base=0, missing_positive_frames=0, missing_negative_frames=plan["missing_frames"],
                missing_frame_count=plan["missing_frames"], decode_source_count=len(sources), sources=sources,
                eligible_positive_frames=plan["eligible_positive_frames"],
                eligible_negative_frames=plan["eligible_negative_frames"], frames=frames)


def validate_missing_contract(path, plan):
    frozen, digest = read_json(path)
    for key, expected in dict(schema=MISSING_SCHEMA, complete=True, frame_index_base=0,
                              approved_audit_sha256=plan["approved_audit_sha256"],
                              source_train_sha256=plan["source_train_sha256"],
                              source_cache_sha256=plan["source_cache_sha256"],
                              source_snapshot_sha256=plan["snapshot_sha256"],
                              snapshot_sha256=plan["snapshot_sha256"], missing_positive_frames=0,
                              missing_negative_frames=plan["missing_frames"],
                              missing_frame_count=plan["missing_frames"]).items():
        if frozen.get(key) != expected or type(frozen.get(key)) is not type(expected):
            raise ValueError(f"Frozen missing-frame contract changed: {key}")
    expected = {(j["video"]["source_sha256"], i): metadata(j["video"])
                for j in plan["jobs"] for i in j["frame_indices"]}
    seen = set()
    for row in frozen["frames"]:
        identity = row.get("source_sha256"), row.get("frame_index")
        if (type(row.get("frame_index")) is not int or identity not in expected or identity in seen
                or row.get("boxes") != [] or row.get("reviewed_status") != "absent"
                or any(row.get(k) != v for k, v in expected[identity].items())):
            raise ValueError("Frozen missing-frame identities or reviewed metadata changed")
        seen.add(identity)
    if seen != set(expected):
        raise ValueError("Frozen missing-frame list is not the exact audited complement")
    plan["pins"][str(Path(path).resolve())] = digest
    return digest


def png_bound(width, height):
    raw = height * (3 * width + 1)
    return raw + (raw // 16383 + 1) * 5 + 65536


def verify_pins(plan):
    if any(sha256(path) != expected for path, expected in plan["pins"].items()):
        raise ValueError("An audited input changed during supplement preparation")


def source_stamp(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def check_video(job):
    video = job["video"]
    path = Path(video["video"]).resolve()
    before = source_stamp(path)
    digest = sha256(path)
    if digest != video["source_sha256"] or source_stamp(path) != before:
        raise ValueError(f"Raw source video SHA256 changed: {path}")
    return before


def extract_job(job, output, snapshot_hash, reserve, write_headroom, stamp, stop):
    import cv2

    video = job["video"]
    source = Path(video["video"]).resolve()
    digest = video["source_sha256"]
    cap = None
    try:
        if source_stamp(source) != stamp:
            raise ValueError(f"Source video changed after raw SHA256 verification: {source}")
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot decode source video: {source}")
        images, labels = output / "images/train" / digest, output / "labels/train" / digest
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        requested = set(job["frame_indices"])
        rows, index = [], 0
        while True:
            if stop.is_set():
                raise RuntimeError("Another supplement worker failed")
            ok, image = cap.read()
            if not ok:
                break
            if index >= video["frame_count"] or image.shape != (video["frame_height"], video["frame_width"], 3):
                raise ValueError(f"Source frame geometry/count differs from the frozen snapshot: {source}")
            if index in requested:
                if shutil.disk_usage(output).free < reserve + write_headroom:
                    raise OSError("Insufficient disk reserve while writing negative PNGs")
                image_path, label_path = images / f"{index:09d}.png", labels / f"{index:09d}.txt"
                if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise OSError(f"Cannot write PNG: {image_path}")
                with label_path.open("x"):
                    pass
                rows.append(dict(metadata(video), dataset="approved_negative_supplement", split="train",
                                 group=digest, raw_video_sha256=digest, frame_index=index,
                                 image=str(image_path), label=str(label_path), boxes=0, reviewed_status="absent",
                                 snapshot_sha256=snapshot_hash))
            index += 1
        if index != video["frame_count"] or len(rows) != len(requested) or source_stamp(source) != stamp:
            raise ValueError(f"Source frame coverage or video identity changed: {source}")
        return rows
    except BaseException:
        stop.set()
        raise
    finally:
        if cap is not None:
            cap.release()


def prepare(audit_path, snapshot_path, missing_frames_path, output, workers=8, reserve_gib=40, dry_run=False):
    if type(workers) is not int or not 1 <= workers <= 16 or type(reserve_gib) is not int or reserve_gib < 40:
        raise ValueError("Use 1-16 workers and a reserve of at least 40 GiB")
    output = Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Use a fresh output directory under an existing parent")
    plan = load_plan(audit_path, snapshot_path)
    frozen_hash = validate_missing_contract(missing_frames_path, plan)
    source = Path(plan["source_dataset"])
    if source == output or source in output.parents:
        raise ValueError("Never write a supplement inside the frozen source dataset")
    bounds = [png_bound(j["video"]["frame_width"], j["video"]["frame_height"]) for j in plan["jobs"]]
    worst = sum(bound * len(job["frame_indices"]) for bound, job in zip(bounds, plan["jobs"]))
    reserve = reserve_gib * GIB
    if shutil.disk_usage(output.parent).free < reserve + worst:
        raise OSError("Insufficient disk space for worst-case lossless PNGs plus the 40 GiB reserve")
    summary = dict(complete=False, dry_run=dry_run, dataset="approved_negative_supplement",
                   source_snapshot_sha256=plan["snapshot_sha256"], snapshot_path=plan["snapshot_path"],
                   approved_audit_sha256=plan["approved_audit_sha256"],
                   source_train_sha256=plan["source_train_sha256"], source_cache_sha256=plan["source_cache_sha256"],
                   missing_frames_sha256=frozen_hash, missing_frames_path=str(Path(missing_frames_path).resolve()),
                   source_videos=len(plan["jobs"]), negative_frames=plan["missing_frames"], positive_frames=0,
                   worst_case_png_bytes=worst, reserved_disk_bytes=reserve, workers=workers,
                   frame_index_base=0, raw_video_hashes_verified=False, lossless_png_compression=3,
                   resizing=False, color_conversion=False, existing_schedule_and_repeats_modified=False,
                   negative_pool_modified=False, training_started=False)
    if dry_run:
        verify_pins(plan)
        return dict(summary, preflight_complete=True)
    import cv2

    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        stamps = list(executor.map(check_video, plan["jobs"]))
    verify_pins(plan)
    output.mkdir()
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    stop = Event()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(extract_job, job, output, plan["snapshot_sha256"], reserve,
                                       max(bounds) * min(workers, len(plan["jobs"])), stamp, stop)
                       for job, stamp in zip(plan["jobs"], stamps)]
            rows = [row for future in futures for row in future.result()]
        verify_pins(plan)
        if len(rows) != plan["missing_frames"]:
            raise ValueError("Supplement frame count changed")
        manifest = output / "manifest.jsonl"
        with manifest.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        summary.update(complete=True, manifest=str(manifest), manifest_sha256=sha256(manifest),
                       raw_video_hashes_verified=True, finished_at=datetime.now(timezone.utc).isoformat())
        write_json(summary_path, summary)
    except BaseException as error:
        summary.update(complete=False, error=str(error))
        write_json(summary_path, summary)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-audit", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--missing-frames", type=Path, required=True, help="Frozen auditmissing_frames.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-gib", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true", help="Validate exact scope and disk budget; do not decode or write")
    args = parser.parse_args()
    result = prepare(args.approved_audit, args.snapshot, args.missing_frames, args.output,
                     args.workers, args.reserve_gib, args.dry_run)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
