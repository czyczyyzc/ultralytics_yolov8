#!/usr/bin/env python3
"""Extend immutable background packs using exact frozen labeler frame identities."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.prepare_labeler_training_expansion import dump, frame_plan, sha256

_PLAN = None
_TEMPORAL = {}


def pixel_plan(item):
    normalized, excluded = frame_plan(item["annotations"], item["frame_count"], item["frame_width"],
                                      item["frame_height"], item["status"] == "approved")
    w, h = item["frame_width"], item["frame_height"]
    return {f: [[(x-bw/2)*w, (y-bh/2)*h, bw*w, bh*h] for x, y, bw, bh in boxes]
            for f, boxes in normalized.items()}, excluded


def snapshot_temporal(item, boxes, heldout):
    import cv2
    from scripts.anti_uav.gray_temporal_background import TemporalBackgrounds

    if item["source_sha256"] in heldout:
        raise ValueError("Heldout temporal source rejected")
    temporal = TemporalBackgrounds.__new__(TemporalBackgrounds)
    temporal.cache, temporal.offsets = {}, (10, -10, 20, -20, 40, -40, 80, -80)
    capture = cv2.VideoCapture(item["video"])
    if not capture.isOpened():
        raise ValueError("Cannot open frozen temporal video")
    temporal.records = {Path(item["video"]).stem: dict(video=Path(item["video"]), metadata=item,
        boxes=boxes, eligible=set(boxes), capture=capture, video_sha256=item["source_sha256"],
        coco_sha256="frozen_labeler_snapshot", approved_manifest_sha256="not_claimed_approved",
        annotation_provenance="frozen_human_confirmed_labels_pending_tasks_explicitly_allowed")}
    return temporal


def worker_init(plan):
    global _PLAN, _TEMPORAL
    _PLAN, _TEMPORAL = plan, {}
    import cv2
    cv2.setNumThreads(1)


def prepare_chunk(job):
    import cv2
    import numpy as np
    from scripts.anti_uav.online_gray_replacement import decode_prepared, encode_prepared
    from scripts.anti_uav.synthesize_gray_drone_replacements import SkipSample, prepare_target, stable_seed

    item = _PLAN["videos"][job["video_index"]]
    digest = item["source_sha256"]
    if digest not in _TEMPORAL:
        if len(_TEMPORAL) >= 2:
            _TEMPORAL.pop(next(iter(_TEMPORAL))).close()
        _TEMPORAL[digest] = snapshot_temporal(item, item["frame_boxes"], _PLAN["heldout_sha256"])
    temporal = _TEMPORAL[digest]
    output = Path(_PLAN["output"])
    path = output/job["pack"]
    signature = hashlib.sha256(json.dumps(job, sort_keys=True).encode()+_PLAN["snapshot_sha256"].encode()).hexdigest()
    db = sqlite3.connect(path, timeout=60)
    db.executescript("""CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS backgrounds(frame INTEGER PRIMARY KEY,image TEXT UNIQUE NOT NULL,
        label TEXT NOT NULL,source_sha TEXT NOT NULL,label_sha TEXT NOT NULL,box TEXT NOT NULL,
        state TEXT NOT NULL,reason TEXT,payload BLOB,payload_sha TEXT);""")
    try:
        old = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if old and old[0] != signature:
            raise ValueError("Chunk belongs to another frozen snapshot")
        with db:
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('signature',?)", (signature,))
        done = {f for f, in db.execute("SELECT frame FROM backgrounds")}
        for offset, row in enumerate(job["rows"]):
            frame, image_path, label_path = row["frame"], Path(row["image"]), Path(row["label"])
            if frame in done:
                continue
            if shutil.disk_usage(output).free < 25*(1 << 30):
                raise OSError("Less than 25 GiB free for background cache")
            image_hash, label_hash = sha256(image_path), sha256(label_path)
            image = cv2.imread(str(image_path))
            if image is None or image.shape[:2] != (item["frame_height"], item["frame_width"]):
                raise ValueError("Frozen image dimensions changed")
            boxes = item["frame_boxes"][frame]
            actual = [[float(v) for v in line.split()] for line in label_path.read_text().splitlines()]
            if len(actual) != len(boxes) or any(v[0] != 0 for v in actual):
                raise ValueError("Frozen image labels differ from snapshot")
            expected = [[(x+bw/2)/item["frame_width"], (y+bh/2)/item["frame_height"],
                         bw/item["frame_width"], bh/item["frame_height"]] for x, y, bw, bh in boxes]
            if not np.allclose(np.asarray(actual)[:, 1:], expected, atol=1e-8, rtol=0):
                raise ValueError("Frozen bbox geometry differs from snapshot")
            payload = payload_sha = reason = None
            state = "ready"
            try:
                if len(boxes) != 1:
                    raise SkipSample("multi_target_original_retained")
                prepared = prepare_target(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), boxes[0],
                    stable_seed(digest, frame), temporal=temporal, video_id=Path(item["video"]).stem, frame_index=frame)
                payload = encode_prepared(prepared)
                restored = decode_prepared(payload)
                if any(not np.array_equal(restored[k], prepared[k]) for k in ("patch", "clean", "erase")):
                    raise ValueError("Background payload roundtrip changed pixels")
                payload_sha = hashlib.sha256(payload).hexdigest()
            except SkipSample as error:
                state, reason = "original_only", str(error)
            finally:
                temporal.cache.clear()
            if sha256(image_path) != image_hash or sha256(label_path) != label_hash:
                raise ValueError("Frozen source changed during background preparation")
            db.execute("INSERT INTO backgrounds VALUES (?,?,?,?,?,?,?,?,?,?)", (frame, str(image_path),
                str(label_path), image_hash, label_hash, json.dumps(boxes[0]), state, reason, payload, payload_sha))
            if offset % 16 == 15:
                db.commit()
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Corrupt background chunk")
        states = dict(db.execute("SELECT state,count(*) FROM backgrounds GROUP BY state"))
        if sum(states.values()) != len(job["rows"]):
            raise ValueError("Missing frozen positive frame in background chunk")
        result = dict(pack=job["pack"], video_sha256=digest, states=states, processed=sum(states.values()))
        dump(output/(job["pack"]+".status.json"), dict(stage="complete", **result))
        return result
    finally:
        db.close()


def finalize(plan, jobs):
    root, previous = Path(plan["output"]), Path(plan["previous_cache"])
    old_index = json.loads((previous/"index.json").read_text())
    old_summary = json.loads((previous/"summary.json").read_text())
    if sha256(previous/"index.json") != plan["previous_index_sha256"]:
        raise ValueError("Previous background index changed")
    images = dict(old_index["images"])
    for pack in {v["pack"] for v in images.values()}:
        if Path(pack).name != pack or not (previous/pack).is_file():
            raise ValueError("Invalid reused background pack")
        (root/pack).symlink_to((previous/pack).resolve())
    reasons, processed, ready, seen = Counter(), 0, 0, set()
    for job in jobs:
        item = plan["videos"][job["video_index"]]
        with sqlite3.connect(f"file:{(root/job['pack']).resolve()}?mode=ro", uri=True) as db:
            for frame, image, label, source, label_sha, box, state, reason, payload_sha in db.execute(
                    "SELECT frame,image,label,source_sha,label_sha,box,state,reason,payload_sha FROM backgrounds"):
                if image in seen:
                    raise ValueError("Duplicate augmented positive path")
                seen.add(image)
                processed += 1
                if state == "ready":
                    ready += 1
                    if image in images:
                        raise ValueError("New background would overwrite an old cache entry")
                    images[image] = dict(pack=job["pack"], frame=frame, video_sha256=item["source_sha256"],
                        source_sha256=source, label_sha256=label_sha, box=json.loads(box), payload_sha=payload_sha)
                else:
                    reasons[reason.split(":")[0]] += 1
    if processed != sum(len(job["rows"]) for job in jobs):
        raise ValueError("Missing backgrounds at finalization")
    for path, digest in plan["protected_inputs"].items():
        if sha256(path) != digest:
            raise ValueError("Frozen snapshot/list/cache changed during preparation")
    index = dict(old_index, images=images)
    dump(root/"index.json", index)
    summary = dict(stage="complete", is_smoke_subset=False, index_sha256=sha256(root/"index.json"),
        processed_positive_frames=old_summary["processed_positive_frames"]+processed,
        ready_backgrounds=len(images), original_only=old_summary["original_only"]+processed-ready,
        new_processed_positive_frames=processed, new_ready_backgrounds=ready,
        new_original_only=processed-ready, new_rejection_reasons=dict(reasons),
        source_videos=old_summary["source_videos"]+len(plan["videos"]), asset_count=len(index["asset_ids"]),
        original_images_preserved=True, synthetic_full_images_written=0, replacement_probability=.5,
        manual_visual_review_pending=True, pending_task_provenance_explicit=True)
    dump(root/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--previous-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 32 or not 16 <= args.chunk_size <= 512:
        raise ValueError("Use a fresh cache, 1..32 workers and chunks of 16..512")
    data, previous, output = args.data_dir.resolve(), args.previous_cache.resolve(), args.output.resolve()
    manifest = json.loads((data/"manifest.json").read_text())
    snapshot = json.loads((data/"annotation_snapshot.json").read_text())
    heldout = {manifest["test_sha256"], manifest["validation_sha256"]}
    old_index = json.loads((previous/"index.json").read_text())
    old_summary = json.loads((previous/"summary.json").read_text())
    if (old_summary["stage"] != "complete" or old_summary["is_smoke_subset"]
            or sha256(previous/"index.json") != old_summary["index_sha256"]
            or sha256(old_index["catalog"]) != old_index["catalog_sha256"]
            or heldout != set(old_index["heldout_sha256"])):
        raise ValueError("Unsafe/incompatible previous background cache")
    if any(v["video_sha256"] not in manifest["train_video_hashes"] for v in old_index["images"].values()):
        raise ValueError("Previous background source is outside the training split")
    videos, jobs = [], []
    for item in snapshot["videos"]:
        digest = item["source_sha256"]
        if digest in heldout or digest not in manifest["train_video_hashes"]:
            raise ValueError("Snapshot background source is not allowed for training")
        if sha256(item["video"]) != digest:
            raise ValueError("Frozen source video checksum changed")
        boxes, _ = pixel_plan(item)
        key = Path(item["video"]).stem+"_"+digest[:12]
        rows = [dict(frame=f, image=str(data/"images/labeler_snapshot"/key/f"{f:09d}.jpg"),
                     label=str(data/"labels/labeler_snapshot"/key/f"{f:09d}.txt"))
                for f, target in sorted(boxes.items()) if target]
        if not rows:
            continue
        position = len(videos)
        videos.append(dict(item, frame_boxes=boxes))
        for offset in range(0, len(rows), args.chunk_size):
            jobs.append(dict(video_index=position, pack=f"snapshot_{digest}_{offset//args.chunk_size:04d}.sqlite",
                             rows=rows[offset:offset+args.chunk_size]))
    output.mkdir(parents=True)
    plan = dict(schema="frozen_labeler_background_extension.v1", videos=videos,
        output=str(output), previous_cache=str(previous), previous_index_sha256=sha256(previous/"index.json"),
        snapshot_sha256=sha256(data/"annotation_snapshot.json"), heldout_sha256=sorted(heldout),
        protected_inputs={str(data/name): sha256(data/name) for name in
            ("annotation_snapshot.json", "train_hardneg.txt", "train_hardneg.cache", "val_monitor.txt", "val_monitor.cache")},
        source_annotations_modified=False, original_retention_not_training_size_filter=True,
        code_sha256=sha256(__file__))
    dump(output/"plan.json", plan)
    dump(output/"jobs.json", jobs)
    completed = []
    def status(stage, **extra):
        dump(output/"job_status.json", dict(stage=stage, pid=os.getpid(), total_chunks=len(jobs),
            completed_chunks=len(completed), processed=sum(x["processed"] for x in completed),
            states=dict(sum((Counter(x["states"]) for x in completed), Counter())), **extra))
    try:
        status("running")
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(plan,)) as pool:
            futures = [pool.submit(prepare_chunk, job) for job in jobs]
            for future in as_completed(futures):
                completed.append(future.result())
                status("running")
        summary = finalize(plan, jobs)
        status("complete", summary=summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
