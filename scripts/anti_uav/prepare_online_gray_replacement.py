#!/usr/bin/env python3
"""Prepare one background ROI per reviewed positive frame, not 53 full images."""
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
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.build_gray_replacement_batch import load_assets, training_records
from scripts.anti_uav.gray_temporal_background import TemporalBackgrounds, legacy_annotations
from scripts.anti_uav.online_gray_replacement import decode_prepared, encode_prepared
from scripts.anti_uav.synthesize_gray_drone_replacements import (
    SkipSample, normalize_box, parse_boxes, prepare_target, sha256, stable_seed,
)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(path)


def reviewed_frames(row):
    """Read all reviewed annotations, independently of historical frame strides."""
    if "legacy_manifest" in row:
        rec, boxes = legacy_annotations(row)
        capture = cv2.VideoCapture(row["video"])
        try:
            if not capture.isOpened():
                raise ValueError("Cannot open legacy source")
            w, h = [int(capture.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
        finally:
            capture.release()
        shape = {f: (w, h) for f in boxes}
        excluded = set()
        annotation_sha = rec["annotation_sha256"]
    else:
        task = Path(row["task"])
        if sha256(task/"manifest.json") != row["manifest_sha256"]:
            raise ValueError("Approved manifest changed")
        meta = json.loads((task/"manifest.json").read_text())
        if meta["video"]["sha256"] != row["sha256"] or meta["frames"]["indexBase"] != 0:
            raise ValueError("Approved identity/frame index mismatch")
        annotation_sha = next(f["sha256"] for f in meta["files"] if f["path"] == "coco/annotations.json")
        if sha256(task/"coco/annotations.json") != annotation_sha:
            raise ValueError("COCO annotations changed")
        data = json.loads((task/"coco/annotations.json").read_text())
        frames = meta["frames"]
        excluded = set(frames.get("excludedUncertainFrameIndices", []))
        excluded.update(frames.get("excludedUnreviewedFrameIndices", []))
        included = set(frames["includedFrameIndices"])-excluded
        images = {i["id"]: i for i in data["images"]}
        frame_ids = {i["frame_index"] for i in images.values()}
        if len(images) != len(data["images"]) or len(frame_ids) != len(images) or not included <= frame_ids:
            raise ValueError("Missing/duplicate frame identities")
        boxes = {f: [] for f in included}
        shape = {i["frame_index"]: (i["width"], i["height"]) for i in images.values()}
        for ann in data["annotations"]:
            frame = images[ann["image_id"]]["frame_index"]
            if frame in included:
                boxes[frame].append(ann["bbox"])
    positive, counts = [], Counter()
    for frame, targets in sorted(boxes.items()):
        if not targets:
            counts["negative_not_augmented"] += 1
            continue
        if len(targets) != 1:
            raise ValueError("Multi-target sources require a separate label-preserving compositor")
        box, (w, h) = targets[0], shape[frame]
        if frame < 0 or len(box) != 4 or not np.isfinite(box).all():
            raise ValueError("Invalid frame or box")
        parse_boxes(normalize_box(box, w, h), w, h)
        positive.append(dict(frame=frame, box=box, width=w, height=h))
        if min(box[2:]) < 3 or max(box[2:]) > 160:
            counts["outside_repair_size_range_original_retained"] += 1
    counts.update(positive=len(positive), uncertain_unreviewed_excluded=len(excluded))
    return positive, dict(counts), annotation_sha


def build_plan(dataset, catalog):
    meta, rows, groups = training_records(dataset, True)
    if {r["sha256"] for r in rows} != set(meta["train_video_hashes"]):
        raise ValueError("Missing training video annotation sources")
    assets = load_assets(catalog, {"24", "25"})
    videos = []
    for row in rows:
        positive, counts, annotation_sha = reviewed_frames(row)
        existing = {int(k): str(p) for k, p in groups.get(row["image_directory"], {}).items() if k.isdecimal()}
        for item in positive:
            if item["frame"] in existing:
                image = Path(existing[item["frame"]])
                item.update(image=str(image), label=str(Path(row["label_directory"])/(image.stem+".txt")))
        videos.append(dict(record=row, positive=positive, counts=counts, annotation_sha256=annotation_sha))
    return dict(schema="online_gray_background_cache.v1", videos=videos,
        source_dataset=str(dataset), catalog=str(catalog), catalog_sha256=sha256(catalog),
        asset_ids=[a["id"] for a, _ in assets], heldout_sha256=[meta["test_sha256"], meta["validation_sha256"]],
        counts=dict(sum((Counter(v["counts"]) for v in videos), Counter())),
        protected_inputs={str(dataset/n): sha256(dataset/n) for n in
            ("manifest.json", "train_hardneg.txt", "val_monitor.txt", "train_hardneg_gray_monitor.yaml")},
        code_sha256={n: sha256(Path(__file__).parent/n) for n in (
            "prepare_online_gray_replacement.py", "online_gray_replacement.py",
            "synthesize_gray_drone_replacements.py", "gray_temporal_background.py")},
        frame_stride=1, per_video_cap=None, original_images_preserved=True, negatives_replaced=False,
        training_started=False, repair_size_range="short >=3px and long <=160px in original pixels",
        quality_failure_policy="retain original image and box; do not force synthesis")


def prepare_video(plan, item, output, only_frames=()):
    cv2.setNumThreads(2)
    root, row = Path(output), item["record"]
    digest = row["sha256"]
    positive, _, annotation_sha = reviewed_frames(row)
    if annotation_sha != item["annotation_sha256"] or [p["frame"] for p in positive] != [p["frame"] for p in item["positive"]]:
        raise ValueError("Reviewed annotations changed after planning")
    chosen = item["positive"]
    if only_frames:
        if not set(only_frames) <= {p["frame"] for p in chosen}:
            raise ValueError("Smoke selection includes a negative/unreviewed frame")
        chosen = [p for p in chosen if p["frame"] in only_frames]
    db_path = root/f"{digest}.sqlite"
    db = sqlite3.connect(db_path, timeout=60)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS backgrounds (
            frame INTEGER PRIMARY KEY, image TEXT UNIQUE NOT NULL, label TEXT NOT NULL,
            source_sha TEXT NOT NULL, label_sha TEXT NOT NULL, box TEXT NOT NULL,
            state TEXT NOT NULL, reason TEXT, payload BLOB, payload_sha TEXT);
    """)
    signature = dict(plan_sha=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                     only_frames=list(only_frames))
    capture, temporal = None, None
    start = time.monotonic()
    status_path = root/f"{digest}.status.json"
    def status(stage, **extra):
        counts = dict(db.execute("SELECT state, count(*) FROM backgrounds GROUP BY state"))
        value = dict(stage=stage, video=row["video"], total_positive_frames=len(chosen),
                     processed=sum(counts.values()), states=counts, seconds=round(time.monotonic()-start, 2), **extra)
        atomic_json(status_path, value)
        return value
    try:
        previous = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if previous and json.loads(previous[0]) != signature:
            raise ValueError("Cannot resume a different annotation/configuration")
        with db:
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('signature',?)", (json.dumps(signature),))
        done = {r[0] for r in db.execute("SELECT frame FROM backgrounds")}
        needed = {p["frame"]: p for p in chosen if p["frame"] not in done}
        if needed:
            entry = row if "legacy_manifest" in row else dict(video=row["video"],
                approved_manifest=str(Path(row["task"])/"manifest.json"), coco=str(Path(row["task"])/"coco/annotations.json"))
            registry = root/f"{digest}.registry.json"
            atomic_json(registry, dict(videos=[entry]))
            temporal = TemporalBackgrounds(registry, blocked=(), blocked_hashes=plan["heldout_sha256"])
            capture = cv2.VideoCapture(row["video"])
            if not capture.isOpened():
                raise ValueError("Cannot decode source")
        for frame in range(max(needed, default=-1)+1):
            if not capture.grab():
                raise ValueError(f"Video ended before frame {frame}")
            if frame not in needed:
                continue
            if shutil.disk_usage(root).free < 25*1024**3:
                raise RuntimeError("Less than 25 GiB free; resume after freeing space")
            source = needed[frame]
            if source.get("image"):
                image, label = Path(source["image"]), Path(source["label"])
            else:
                image = root/"images"/digest/f"{frame:09d}.png"
                label = root/"labels"/digest/f"{frame:09d}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                ok, bgr = capture.retrieve()
                if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame+1:
                    raise ValueError("Decoded frame index mismatch")
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                temp = image.with_suffix(".tmp.png")
                if not cv2.imwrite(str(temp), gray):
                    raise IOError("Cannot save missing original frame")
                temp.replace(image)
                label.write_text(normalize_box(source["box"], source["width"], source["height"]))
            source_hash, label_hash = sha256(image), sha256(label)
            bgr = cv2.imread(str(image))
            if bgr is None or bgr.shape[:2] != (source["height"], source["width"]):
                raise ValueError("Source image dimensions changed")
            boxes = parse_boxes(label.read_text(), source["width"], source["height"])
            if len(boxes) != 1 or not np.allclose(boxes[0], source["box"], atol=.25, rtol=0):
                raise ValueError(f"Training label disagrees with reviewed annotation: {label}")
            payload, payload_sha, state, reason = None, None, "ready", None
            try:
                prepared = prepare_target(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), source["box"],
                    stable_seed(digest, frame), temporal=temporal, video_id=Path(row["video"]).stem, frame_index=frame)
                payload = encode_prepared(prepared)
                restored = decode_prepared(payload)
                if any(not np.array_equal(restored[k], prepared[k]) for k in ("patch", "clean", "erase")):
                    raise AssertionError("Background serialization is not lossless")
                payload_sha = hashlib.sha256(payload).hexdigest()
            except SkipSample as error:
                state, reason = "original_only", str(error)
            temporal.cache.clear()
            if sha256(image) != source_hash or sha256(label) != label_hash:
                raise ValueError("Source image/label changed during preparation")
            with db:
                db.execute("INSERT INTO backgrounds VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (frame, str(image), str(label), source_hash, label_hash, json.dumps(source["box"]),
                     state, reason, payload, payload_sha))
            status("running", current_frame=frame, pid=os.getpid())
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise AssertionError("Cache database integrity check failed")
        result = status("complete")
        if result["processed"] != len(chosen):
            raise AssertionError("Missing positive frames")
        return result
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        if capture is not None:
            capture.release()
        if temporal is not None:
            temporal.close()
        db.close()


def finalize(plan, selected, root, is_smoke):
    images, additions, reasons, total = {}, [], Counter(), 0
    base = Path(plan["source_dataset"])
    baseline = (base/"train_hardneg.txt").read_text().splitlines()
    baseline_set = set(baseline)
    for item in selected:
        row = item["record"]
        name = f"{row['sha256']}.sqlite"
        db = sqlite3.connect(f"file:{root/name}?mode=ro", uri=True)
        try:
            for frame, image, label, source_sha, label_sha, box, state, reason, payload_sha in db.execute(
                    "SELECT frame,image,label,source_sha,label_sha,box,state,reason,payload_sha FROM backgrounds"):
                total += 1
                if image not in baseline_set:
                    additions.append(image)
                if state == "ready":
                    images[image] = dict(pack=name, frame=frame, video_sha256=row["sha256"],
                        source_sha256=source_sha, label_sha256=label_sha, box=json.loads(box), payload_sha=payload_sha)
                else:
                    reasons[reason.split(":")[0]] += 1
        finally:
            db.close()
    if len(additions) != len(set(additions)):
        raise ValueError("Duplicate new originals")
    if any(sha256(Path(p)) != h for p, h in plan["protected_inputs"].items()):
        raise ValueError("Protected baseline changed")
    index = dict(images=images, heldout_sha256=plan["heldout_sha256"], catalog=plan["catalog"],
                 catalog_sha256=plan["catalog_sha256"], asset_ids=plan["asset_ids"])
    atomic_json(root/"index.json", index)
    summary = dict(stage="complete", is_smoke_subset=is_smoke, processed_positive_frames=total,
        ready_backgrounds=len(images), original_only=total-len(images), rejection_reasons=dict(reasons),
        added_original_positive_frames=len(additions), source_videos=len(selected),
        asset_count=len(plan["asset_ids"]), index_sha256=sha256(root/"index.json"), training_started=False,
        synthetic_full_images_written=0, replacement_probability=.5, manual_visual_review_pending=True)
    if not is_smoke:
        import yaml
        from scripts.anti_uav.append_approved_gray_native import expansion_sampling
        source = json.loads((base/"manifest.json").read_text())
        config = yaml.safe_load((base/"train_hardneg_gray_monitor.yaml").read_text())
        if config.get("online_scale") or config.get("online_replacement"):
            raise ValueError("Baseline already contains another online augmentation")
        pool, positives, negatives, budget, anchors = expansion_sampling(
            source, config, baseline, len(additions), [], .15)
        train = root/"train_hardneg.txt"
        train.write_text("\n".join(baseline+sorted(additions))+"\n")
        (root/"val_monitor.txt").write_bytes((base/"val_monitor.txt").read_bytes())
        (root/"new_negative_pool.txt").write_text("\n".join(pool)+"\n")
        config.update(path=str(root), train=str(train), val=str(root/"val_monitor.txt"),
            label_sampling=dict(negative_pool=str(root/"new_negative_pool.txt"), negative_pool_count=len(pool),
                negatives_per_epoch=budget, anchor_slots=anchors, target_negative_fraction=.15),
            online_replacement=dict(cache=str(root), replacement_probability=.5, seed=20260918))
        (root/"train_online_gray_monitor.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        control = dict(config)
        control.pop("online_replacement")
        (root/"train_control_gray_monitor.yaml").write_text(yaml.safe_dump(control, sort_keys=False))
        summary.update(epoch_positive=positives, epoch_negative=negatives, epoch_total=positives+negatives,
                       negative_fraction=negatives/(positives+negatives), baseline_prefix_exactly_preserved=True)
        atomic_json(root/"manifest.json", dict(source, schema="online_gray_positive_expansion.v1",
            source_dataset=str(base), background_cache=str(root), added_positive_samples=len(additions),
            append_only_positive=positives, epoch_positive=positives, epoch_negative=negatives,
            candidate_entries=len(baseline)+len(additions), final_entries=positives+negatives,
            final_negative_fraction=negatives/(positives+negatives), training_started=False,
            native_schedule_sha256=sha256(train), original_gray_training_videos=len(selected),
            online_replacement=config["online_replacement"], positive_stride=1))
    atomic_json(root/"summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--video-sha256", help="One allowed video for a smoke test only; omit for all videos")
    p.add_argument("--frames", help="Smoke frame IDs only; omit for uncapped full coverage")
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        raise ValueError("Use 1..8 CPU-only workers")
    plan = build_plan(a.dataset.resolve(), a.catalog.resolve())
    selected = [v for v in plan["videos"] if not a.video_sha256 or v["record"]["sha256"] == a.video_sha256]
    frames = tuple(int(f) for f in a.frames.split(",")) if a.frames else ()
    if not selected or (frames and len(selected) != 1):
        raise ValueError("Explicit smoke frames require one allowed training video")
    if a.output.exists() and not a.resume:
        raise FileExistsError("Use a new folder or --resume")
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/"plan.json").exists() and json.loads((root/"plan.json").read_text()) != plan:
        raise ValueError("Cannot resume changed inputs/code")
    atomic_json(root/"plan.json", plan)
    if a.plan_only:
        print(json.dumps(dict(videos=len(selected), counts=plan["counts"], assets=len(plan["asset_ids"]))))
        return
    # Keep expensive registration out of GPU training; independent CPU workers
    # write one transactional DB each, with no shared writer or NFS WAL file.
    status = dict(stage="running", pid=os.getpid(), total_videos=len(selected), completed_videos=[],
                  is_smoke_subset=bool(a.video_sha256 or frames), training_started=False)
    atomic_json(root/"job_status.json", status)
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(prepare_video, plan, v, str(root), frames) for v in selected]
            for future in as_completed(futures):
                status["completed_videos"].append(future.result())
                atomic_json(root/"job_status.json", status)
        summary = finalize(plan, selected, root, status["is_smoke_subset"])
        status.update(stage="complete", summary=summary)
        atomic_json(root/"job_status.json", status)
    except BaseException as error:
        status.update(stage="failed", error=repr(error))
        atomic_json(root/"job_status.json", status)
        raise


if __name__ == "__main__":
    main()
