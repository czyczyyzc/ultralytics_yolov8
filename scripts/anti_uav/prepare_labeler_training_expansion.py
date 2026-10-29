#!/usr/bin/env python3
"""Append immutable labeler snapshots without treating unfinished labels as negatives."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def excluded_calibration_source(path, token):
    """Sequence names match components, not a dataset root mentioning its holdout."""
    if token in {"Video00004", "Video00009"}:
        return any(part.casefold() == token.casefold() for part in Path(path).parts)
    return token in str(path)


def frame_plan(annotations, frame_count, width, height, approved):
    """Keep all valid box sizes; only reviewed absent labels may supply negatives."""
    grouped = defaultdict(list)
    for annotation in annotations:
        index = int(annotation["frame_index"])
        if not 0 <= index < frame_count:
            raise ValueError("Out-of-range annotation frame")
        grouped[index].append(annotation)
    plan, excluded = {}, Counter(unlabelled=frame_count-len(grouped))
    for index, rows in grouped.items():
        statuses = {row["status"] for row in rows}
        if not statuses <= {"confirmed", "absent", "uncertain"}:
            raise ValueError("Unexpected annotation status")
        if "uncertain" in statuses:
            excluded["uncertain"] += 1
            continue
        boxes, invalid = [], False
        for row in rows:
            if row["status"] != "confirmed":
                continue
            values = [row[key] for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
            if any(value is None or not math.isfinite(value) for value in values):
                invalid = True
                break
            x1, y1, x2, y2 = values
            x1, x2 = max(0., x1), min(float(width), x2)
            y1, y2 = max(0., y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                invalid = True
                break
            boxes.append([(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height])
        if invalid:
            excluded["invalid_bbox"] += 1
        elif boxes:
            plan[index] = boxes
        elif approved and statuses == {"absent"}:
            plan[index] = []
        else:
            excluded["unreviewed_absent"] += 1
    return plan, dict(excluded)


def snapshot_tasks(database, video_root, blocked):
    con = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN")
        rows = con.execute("""SELECT t.id task_id,t.status,t.revision,t.approved_revision,t.updated_at,
            v.name,v.source_relpath,v.source_sha256,v.frame_count,v.frame_width,v.frame_height,v.fps,
            (SELECT mi.display_path FROM media_imports mi WHERE mi.task_id=t.id AND mi.status='succeeded'
             ORDER BY mi.created_at DESC LIMIT 1) original_path
            FROM tasks t JOIN videos v ON v.id=t.video_id
            WHERE t.is_published=1 AND t.status IN ('approved','submitted','in_progress')
            ORDER BY t.created_at,t.id""").fetchall()
        selected, seen = [], set()
        for row in rows:
            item = dict(row)
            digest = item["source_sha256"]
            if digest in blocked or digest in seen:
                continue
            labels = [dict(a) for a in con.execute("""SELECT frame_index,object_id,status,
                bbox_x1,bbox_y1,bbox_x2,bbox_y2,source,updated_at
                FROM annotations WHERE task_id=? ORDER BY frame_index,object_id""", (item["task_id"],))]
            if not labels:
                continue
            video = (video_root / item["source_relpath"]).resolve()
            video.relative_to(video_root.resolve())
            if not video.is_file():
                raise FileNotFoundError(video)
            item.update(video=str(video), annotations=labels)
            selected.append(item)
            seen.add(digest)
        con.rollback()
        return selected
    finally:
        con.close()


def extract(item, output):
    import cv2
    import numpy as np

    video = Path(item["video"])
    if sha256(video) != item["source_sha256"]:
        raise ValueError(f"Video bytes changed: {video}")
    plan, excluded = frame_plan(item["annotations"], item["frame_count"], item["frame_width"],
                                item["frame_height"], item["status"] == "approved")
    if not plan:
        return dict(task_id=item["task_id"], included=False, excluded=excluded), [], [], []
    key = video.stem + "_" + item["source_sha256"][:12]
    images, labels = output / "images/labeler_snapshot" / key, output / "labels/labeler_snapshot" / key
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {video}")
    positive, negative, cached = [], [], []
    index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if image.shape[:2] != (item["frame_height"], item["frame_width"]):
                raise ValueError(f"Frame geometry differs from annotation source: {video}")
            if index in plan:
                path = images / f"{index:09d}.jpg"
                boxes = plan[index]
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(path)
                text = "".join("0 " + " ".join(f"{v:.10f}" for v in box) + "\n" for box in boxes)
                (labels / f"{index:09d}.txt").write_text(text)
                (positive if boxes else negative).append(str(path))
                cached.append(dict(im_file=str(path), shape=image.shape[:2],
                    cls=np.zeros((len(boxes), 1), dtype=np.float32),
                    bboxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4), segments=[], keypoints=None,
                    normalized=True, bbox_format="xywh"))
            index += 1
    finally:
        cap.release()
    if index != item["frame_count"] or len(cached) != len(plan):
        raise ValueError(f"Frame coverage mismatch: {video}: {index} vs {item['frame_count']}")
    record = dict(task_id=item["task_id"], included=True, status=item["status"], revision=item["revision"],
        video=str(video), sha256=item["source_sha256"], original_path=item["original_path"],
        decoded_frames=index, positive_samples=len(positive), negative_samples=len(negative),
        included_frames=len(cached), excluded=excluded, image_directory=str(images), label_directory=str(labels))
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record, positive, negative, cached


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--baseline-protocol", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    a.source, a.output = a.source.resolve(), a.output.resolve()
    if a.output.exists() or not 1 <= a.workers <= 16:
        raise ValueError("Use a fresh output directory and 1-16 extraction workers")
    if shutil.disk_usage(a.output.parent).free < 80 * (1 << 30):
        raise OSError("Less than 80 GiB free; refuse extraction")
    import cv2
    import numpy as np
    import yaml
    from scripts.anti_uav.append_approved_gray_native import append_samples, expansion_sampling

    cv2.setNumThreads(1)
    config = yaml.safe_load((a.source / "train_online_gray_monitor.yaml").read_text())
    source = json.loads((a.source / "manifest.json").read_text())
    baseline_protocol = json.loads(a.baseline_protocol.read_text())
    for split in ("train", "val"):
        listing = Path(config[split])
        record = baseline_protocol["trusted_label_caches"][str(listing)]
        if sha256(listing) != record["source_sha256"] or sha256(listing.with_suffix(".cache")) != record["cache_sha256"]:
            raise ValueError(f"Baseline {split} cache/list differs from the 52-video training protocol")
    base = Path(config["train"]).read_text().splitlines()
    validation = Path(config["val"]).read_text().splitlines()
    blocked = set(source["train_video_hashes"]) | {source["test_sha256"], source["validation_sha256"]}
    items = snapshot_tasks(a.database, a.video_root, blocked)
    if not items:
        raise ValueError("No new labelled videos")
    a.output.mkdir(parents=True)
    snapshot = dict(captured_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        source_database=str(a.database), baseline_manifest=str(a.source / "manifest.json"),
        frame_index_base=0, unreviewed_absent_policy="exclude", uncertain_policy="exclude_entire_frame",
        size_filter=False, source_annotations_modified=False, videos=items)
    dump(a.output / "annotation_snapshot.json", snapshot)
    dump(a.output / "status.json", dict(stage="extracting", candidates=len(items)))
    records, additions, added_negatives, added_cache = [], [], [], []
    positive_count = negative_count = 0
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        for record, pos, neg, cache in executor.map(lambda item: extract(item, a.output), items):
            records.append(record)
            additions.extend(pos + neg)
            added_negatives.extend(neg)
            added_cache.extend(cache)
            positive_count += len(pos)
            negative_count += len(neg)
            dump(a.output / "preparation_progress.json", records)
    included = [r for r in records if r["included"]]
    if not included:
        raise ValueError("No safe labelled frames remain")
    schedule = append_samples(base, additions, validation)
    merged_pool, positives, desired, budget, anchors = expansion_sampling(
        source, config, base, positive_count, added_negatives, .15)
    train_list, val_list, pool = a.output / "train_hardneg.txt", a.output / "val_monitor.txt", a.output / "negative_pool.txt"
    train_list.write_text("\n".join(schedule) + "\n")
    val_list.write_text("\n".join(validation) + "\n")
    pool.write_text("\n".join(sorted(merged_pool)) + "\n")
    baseline_cache_path = Path(config["train"]).with_suffix(".cache")
    cache = np.load(baseline_cache_path, allow_pickle=True).item()
    if sorted(row["im_file"] for row in cache["labels"]) != sorted(base):
        raise ValueError("Pinned baseline cache entries do not match the complete repeated baseline schedule")
    labels = sorted(cache["labels"] + added_cache, key=lambda row: row["im_file"])
    identity = hashlib.sha256((sha256(baseline_cache_path) + sha256(train_list) +
                               sha256(a.output / "annotation_snapshot.json")).encode()).hexdigest()
    # This identity is consumed only through cryptographically pinned trusted_label_caches.
    # It intentionally avoids restatting every unchanged NFS image and label.
    cache.update(labels=labels, hash=identity, results=(len(labels), 0,
        sum(not len(row["cls"]) for row in labels), 0, len(labels)), msgs=[])
    with train_list.with_suffix(".cache").open("wb") as stream:
        np.save(stream, cache, allow_pickle=True)
    shutil.copy2(Path(config["val"]).with_suffix(".cache"), val_list.with_suffix(".cache"))
    final_config = dict(config, path=str(a.output), train=str(train_list), val=str(val_list),
        label_sampling=dict(negative_pool=str(pool), negative_pool_count=len(merged_pool),
            negatives_per_epoch=budget, anchor_slots=anchors, target_negative_fraction=.15))
    pinned = {}
    for listing in (train_list, val_list):
        cached = np.load(listing.with_suffix(".cache"), allow_pickle=True).item()
        pinned[str(listing)] = dict(path=str(listing.with_suffix(".cache")), source_sha256=sha256(listing),
            cache_sha256=sha256(listing.with_suffix(".cache")), dataset_hash=str(cached["hash"]),
            entries=int(cached["results"][4]))
    final_config["trusted_label_caches"] = pinned
    (a.output / "train_online_gray_monitor.yaml").write_text(yaml.safe_dump(final_config, sort_keys=False))
    manifest = dict(source, schema="anti_uav.labeler_snapshot_expansion.v1", source_dataset=str(a.source),
        train_video_hashes=sorted(set(source["train_video_hashes"]) | {r["sha256"] for r in included}),
        latest_appended_videos=included, appended_videos=source.get("appended_videos", []) + included,
        original_gray_training_videos=len(source["train_video_hashes"]) + len(included),
        append_only_entries=len(schedule), candidate_entries=len(schedule), append_only_positive=positives,
        negative=source["negative"] + negative_count, final_entries=positives + desired,
        epoch_positive=positives, epoch_negative=desired, additional_negatives_per_epoch=budget,
        final_negative_fraction=desired / (positives + desired), fixed_negative_anchors=source["negative"]-(len(merged_pool)-negative_count),
        added_positive_samples=positive_count, added_negative_samples=negative_count, positive_repeat=1,
        baseline_prefix_exactly_preserved=True, training_started=False,
        native_schedule_sha256=sha256(train_list), validation_schedule_sha256=sha256(val_list),
        snapshot_sha256=sha256(a.output / "annotation_snapshot.json"), label_snapshot_policy=snapshot.copy())
    manifest["label_snapshot_policy"].pop("videos")
    manifest["label_snapshot_policy"]["pending_labels_are_not_approved"] = True
    dump(a.output / "manifest.json", manifest)
    status = dict(stage="complete", training_videos=manifest["original_gray_training_videos"],
        added_videos=len(included), added_positive=positive_count, added_negative=negative_count,
        source_statuses=dict(Counter(r["status"] for r in included)),
        excluded_frames=dict(sum((Counter(r["excluded"]) for r in records), Counter())),
        train_slots=len(schedule), epoch_slots=positives+desired, negative_fraction=manifest["final_negative_fraction"],
        replacement_policy="Existing 328-asset cache/p=0.5 retained; uncached new frames remain original pixels")
    dump(a.output / "status.json", status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
