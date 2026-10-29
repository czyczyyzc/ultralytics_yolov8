#!/usr/bin/env python3
"""Snapshot imported video tasks, render detections, and score reviewed frame labels."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.evaluate_detector_clip_csv import match_frame


def dump(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_tasks(database, videos_dir, since, prefix):
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        rows = connection.execute(
            """SELECT mi.id import_id, mi.display_path, mi.name original_name,
                      mi.created_at imported_at, v.source_relpath, v.frame_count,
                      v.frame_width, v.frame_height, v.fps, v.source_sha256,
                      t.id task_id, t.status task_status, t.revision, t.approved_revision
               FROM media_imports mi JOIN tasks t ON t.id=mi.task_id
               JOIN videos v ON v.id=t.video_id
               WHERE mi.status='succeeded' AND mi.created_at>=strftime('%s', ?)
               ORDER BY mi.display_path, mi.id""",
            (since,),
        ).fetchall()
        output, seen = [], set()
        for row in rows:
            item = dict(row)
            if not item["display_path"].startswith(prefix) or item["task_id"] in seen:
                continue
            seen.add(item["task_id"])
            source = (videos_dir / item["source_relpath"]).resolve()
            source.relative_to(videos_dir.resolve())
            if not source.is_file():
                raise FileNotFoundError(source)
            item["video"] = str(source)
            item["key"] = f"clip_{len(output)+1:02d}_{Path(item['original_name']).stem}"
            item["annotations"] = [dict(record) for record in connection.execute(
                """SELECT frame_index, object_id, status, bbox_x1, bbox_y1, bbox_x2, bbox_y2
                   FROM annotations WHERE task_id=? ORDER BY frame_index, object_id""",
                (item["task_id"],),
            )]
            output.append(item)
        connection.rollback()
        return output
    finally:
        connection.close()


def score_predictions(item, predictions_path, thresholds=(.03, .05, .10, .25)):
    labels = {}
    for annotation in item["annotations"]:
        labels.setdefault(annotation["frame_index"], []).append(annotation)
    eligible = item["task_status"] in ("submitted", "approved")
    totals = {f"{c:.2f}": dict(tp=0, fp=0, fn=0, frames=0, gt_boxes=0) for c in thresholds}
    coverage = {f"{c:.2f}": 0 for c in thresholds}
    frames, detections, uncertain, unlabelled = 0, 0, 0, 0
    for line in predictions_path.open():
        prediction = json.loads(line)
        index = prediction["frame_index"]
        if index != frames:
            raise ValueError("Prediction indices must be complete and sequential from zero")
        frames += 1
        boxes = prediction["boxes_xyxy_score"]
        detections += len(boxes)
        records = labels.get(index, [])
        if not records:
            unlabelled += 1
        has_uncertain = any(a["status"] == "uncertain" for a in records)
        uncertain += has_uncertain
        ground_truth = [tuple(a[k] for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
                        for a in records if a["status"] == "confirmed"]
        for confidence in thresholds:
            key = f"{confidence:.2f}"
            filtered = [{"box": tuple(b[:4]), "confidence": b[4]}
                        for b in boxes if b[4] >= confidence]
            coverage[key] += bool(filtered)
            if not eligible or not records or has_uncertain:
                continue
            tp, fp, fn = match_frame(ground_truth, filtered, .5)
            counter = totals[key]
            counter["tp"] += tp
            counter["fp"] += fp
            counter["fn"] += fn
            counter["frames"] += 1
            counter["gt_boxes"] += len(ground_truth)
    if frames != item["frame_count"]:
        raise ValueError(f"Prediction count {frames} != source count {item['frame_count']}")
    for counter in totals.values():
        counter["recall"] = counter["tp"] / counter["gt_boxes"] if counter["gt_boxes"] else None
        denominator = counter["tp"] + counter["fp"]
        counter["precision"] = counter["tp"] / denominator if denominator else None
    return dict(
        frames=frames, detections=detections, task_status=item["task_status"],
        provisional_labels=item["task_status"] == "submitted",
        unlabelled_frames=unlabelled, uncertain_frames=uncertain,
        coverage_warning="Frames with predictions is not recall; predictions may be false positives.",
        frames_with_predictions=coverage,
        metrics=totals if eligible else None,
        evaluation_iou=.5, annotation_revision=item["revision"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--since", default="2026-10-28")
    parser.add_argument("--display-prefix", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "2"])
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if sha256(args.model) != args.expected_sha256:
        raise ValueError("Model SHA256 does not match the delivered checkpoint")
    if args.output.exists() or args.batch < 1:
        raise ValueError("Output must be fresh and batch must be positive")
    items = snapshot_tasks(args.database, args.videos_dir, args.since, args.display_prefix)
    if not items:
        raise ValueError("No successfully imported videos match the prefix")
    args.output.mkdir(parents=True)
    protocol = dict(
        model=str(args.model), model_sha256=args.expected_sha256,
        model_label="52-video public-repro P3 Student FP32", input_hw=[544, 960],
        confidence=.03, nms_iou=.45, max_det=100, tracker=False, frame_stride=1,
        devices=args.devices, source_database=str(args.database),
        warning="Submitted labels are provisional, not approved ground truth. Uncertain/unlabelled frames excluded.",
        videos=items,
    )
    dump(args.output / "manifest.json", protocol)

    def worker(arguments):
        device, subset = arguments
        results = []
        for item in subset:
            directory = args.output / item["key"]
            command = [sys.executable, str(ROOT / "scripts/anti_uav/render_pt_detector_video.py"),
                       "--source-video", item["video"], "--model", str(args.model),
                       "--output", str(directory), "--output-name", item["key"] + "_P3_FP32_conf003.mp4",
                       "--label", "52-video P3 Student | " + item["key"],
                       "--device", device, "--batch", str(args.batch), "--conf", ".03"]
            with (args.output / (item["key"] + ".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            rendered = json.loads((directory / "summary.json").read_text())
            if rendered["weights_sha256"] != args.expected_sha256 or rendered["source_sha256"] != item["source_sha256"]:
                raise ValueError("Rendered model/source hash mismatch")
            metrics = score_predictions(item, directory / "predictions.jsonl")
            result = dict(key=item["key"], original_path=item["display_path"], video=rendered["video"], **metrics)
            dump(directory / "evaluation.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
        return results

    groups = [(device, items[index::len(args.devices)]) for index, device in enumerate(args.devices)]
    results = []
    try:
        with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
            for group in executor.map(worker, groups):
                results.extend(group)
    except BaseException as error:
        dump(args.output / "status.json", dict(stage="failed", error=repr(error)))
        raise
    results.sort(key=lambda r: r["key"])
    pooled = {}
    for confidence in ("0.03", "0.05", "0.10", "0.25"):
        aggregate = {key: sum(r["metrics"][confidence][key] for r in results if r["metrics"])
                     for key in ("tp", "fp", "fn", "frames", "gt_boxes")}
        aggregate["recall"] = aggregate["tp"] / aggregate["gt_boxes"] if aggregate["gt_boxes"] else None
        denominator = aggregate["tp"] + aggregate["fp"]
        aggregate["precision"] = aggregate["tp"] / denominator if denominator else None
        pooled[confidence] = aggregate
    dump(args.output / "results.json", dict(videos=results, provisional_pooled=pooled))
    dump(args.output / "status.json", dict(stage="complete", videos=len(results), frames=sum(r["frames"] for r in results)))


if __name__ == "__main__":
    main()
