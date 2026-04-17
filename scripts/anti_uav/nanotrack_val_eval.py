#!/usr/bin/env python3
"""Tracker-only validation for NanoTrack checkpoints on Anti-UAV300 splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.convert_anti_uav300_nanotrack import discover_sequences, read_boxes, split_sequences
from ultralytics import solutions


DEFAULT_SOURCE_ROOT = Path("/mnt/chenziye/datasets/anti_uav/Anti-UAV300")
DEFAULT_CONVERTED_ROOT = Path("/mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Original Anti-UAV300 root.")
    parser.add_argument("--converted-root", type=Path, default=DEFAULT_CONVERTED_ROOT, help="Converted NanoTrack dataset root.")
    parser.add_argument("--modality", choices=("rgb", "ir"), default="rgb", help="Validation modality.")
    parser.add_argument("--split", choices=("train", "val"), default="val", help="Validation split.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fallback split ratio when no manifest is present.")
    parser.add_argument("--config", type=Path, default=None, help="NanoTrack config yaml.")
    parser.add_argument("--snapshot", type=Path, default=None, help="NanoTrack checkpoint to evaluate.")
    parser.add_argument("--nanotrack-root", type=Path, default=None, help="Optional NanoTrack workspace root.")
    parser.add_argument("--device", default="cpu", help="Torch device for NanoTrack, for example cpu or cuda:0.")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="NanoTrack score threshold for considering a prediction valid.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold used for success / tp accounting.")
    parser.add_argument("--center-threshold", type=float, default=20.0, help="Center error threshold in pixels.")
    parser.add_argument("--max-sequences", type=int, default=0, help="Optional cap on the number of sequences.")
    parser.add_argument("--per-sequence-dir", type=Path, default=None, help="Optional directory for per-sequence JSON summaries.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional aggregate output JSON.")
    parser.add_argument("--dry-run", action="store_true", help="List resolved sequences without loading NanoTrack.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_split_sequences(source_root: Path, converted_root: Path, modality: str, split: str, val_ratio: float) -> list[dict]:
    manifest_path = converted_root / modality / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get(split, [])

    sequences = [sequence for sequence in discover_sequences(source_root) if modality in sequence["modalities"]]
    train_names, val_names = split_sequences([sequence["name"] for sequence in sequences], val_ratio)
    selected_names = train_names if split == "train" else val_names
    entries = []
    for sequence in sequences:
        if sequence["name"] not in selected_names:
            continue
        entries.append(
            {
                "name": sequence["name"],
                "source_dir": str(sequence["dir"].resolve()),
                "video": str(sequence["modalities"][modality]["video"].resolve()),
                "label": str(sequence["modalities"][modality]["label"].resolve()),
            }
        )
    return entries


def bbox_xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in box[:4]]
    return x, y, x + w, y + h


def bbox_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area2 = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = max(area1 + area2 - inter, 1e-6)
    return float(inter / union)


def center_error(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return float(np.hypot(cx1 - cx2, cy1 - cy2))


def is_present(box: list[float] | tuple[float, ...]) -> bool:
    return bool(box) and len(box) >= 4 and float(box[2]) > 0 and float(box[3]) > 0


def build_nanotrack(args: argparse.Namespace):
    return solutions.build_tracker(
        "nanotrack",
        nanotrack_root=args.nanotrack_root or None,
        config_path=args.config,
        snapshot_path=args.snapshot,
        device=args.device,
        score_threshold=args.score_threshold,
    )


def evaluate_sequence(entry: dict, tracker, args: argparse.Namespace) -> dict:
    video_path = Path(entry["video"]).expanduser().resolve()
    label_path = Path(entry["label"]).expanduser().resolve()
    boxes = read_boxes(label_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    metrics = {
        "sequence": entry["name"],
        "frames_evaluated": 0,
        "gt_present_frames": 0,
        "predicted_frames": 0,
        "tp_frames": 0,
        "fp_frames": 0,
        "fn_frames": 0,
        "matched_frames": 0,
        "center_hits": 0,
        "absent_fp_frames": 0,
        "iou_sum": 0.0,
        "started_from_frame": -1,
    }
    per_frame = []

    frame_index = 0
    initialized = False
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            gt_box = boxes[frame_index] if frame_index < len(boxes) else []
            gt_present = is_present(gt_box)
            gt_xyxy = bbox_xywh_to_xyxy(gt_box) if gt_present else None

            if not initialized:
                if gt_present:
                    tracker.reset()
                    tracker.init(frame, gt_xyxy)
                    initialized = True
                    metrics["started_from_frame"] = frame_index
                frame_index += 1
                continue

            pred_present, pred_bbox, score = tracker.update(frame)
            metrics["frames_evaluated"] += 1
            metrics["gt_present_frames"] += int(gt_present)
            metrics["predicted_frames"] += int(pred_present)

            iou = 0.0
            error = None
            matched = False
            if gt_present and pred_present and pred_bbox is not None:
                iou = bbox_iou(pred_bbox, gt_xyxy)
                error = center_error(pred_bbox, gt_xyxy)
                if iou >= args.iou_threshold:
                    matched = True
                    metrics["tp_frames"] += 1
                    metrics["matched_frames"] += 1
                    metrics["iou_sum"] += iou
                else:
                    metrics["fp_frames"] += 1
                    metrics["fn_frames"] += 1
                if error is not None and error <= args.center_threshold:
                    metrics["center_hits"] += 1
            elif gt_present:
                metrics["fn_frames"] += 1
            elif pred_present:
                metrics["fp_frames"] += 1
                metrics["absent_fp_frames"] += 1

            per_frame.append(
                {
                    "frame_index": frame_index,
                    "gt_present": gt_present,
                    "pred_present": bool(pred_present),
                    "score": float(score),
                    "iou": float(iou),
                    "center_error": None if error is None else float(error),
                    "matched": matched,
                }
            )
            frame_index += 1
    finally:
        cap.release()

    gt_present = metrics["gt_present_frames"]
    gt_absent = max(metrics["frames_evaluated"] - gt_present, 0)
    metrics["precision"] = metrics["tp_frames"] / max(metrics["tp_frames"] + metrics["fp_frames"], 1)
    metrics["recall"] = metrics["tp_frames"] / max(metrics["tp_frames"] + metrics["fn_frames"], 1)
    metrics["avg_iou"] = metrics["iou_sum"] / max(metrics["matched_frames"], 1)
    metrics["success_rate"] = metrics["tp_frames"] / max(gt_present, 1)
    metrics["center_precision"] = metrics["center_hits"] / max(gt_present, 1)
    metrics["absent_fp_rate"] = metrics["absent_fp_frames"] / max(gt_absent, 1)
    metrics["composite"] = (
        metrics["success_rate"] * 0.6
        + metrics["center_precision"] * 0.2
        + (1.0 - metrics["absent_fp_rate"]) * 0.2
    )
    metrics["frames"] = per_frame
    return metrics


def aggregate_results(sequence_results: list[dict]) -> dict:
    totals = {
        "sequence_count": len(sequence_results),
        "frames_evaluated": 0,
        "gt_present_frames": 0,
        "predicted_frames": 0,
        "tp_frames": 0,
        "fp_frames": 0,
        "fn_frames": 0,
        "matched_frames": 0,
        "center_hits": 0,
        "absent_fp_frames": 0,
        "iou_sum": 0.0,
    }
    for result in sequence_results:
        for key in totals:
            if key == "sequence_count":
                continue
            totals[key] += result.get(key, 0)

    totals["precision"] = totals["tp_frames"] / max(totals["tp_frames"] + totals["fp_frames"], 1)
    totals["recall"] = totals["tp_frames"] / max(totals["tp_frames"] + totals["fn_frames"], 1)
    totals["avg_iou"] = totals["iou_sum"] / max(totals["matched_frames"], 1)
    totals["success_rate"] = totals["tp_frames"] / max(totals["gt_present_frames"], 1)
    totals["center_precision"] = totals["center_hits"] / max(totals["gt_present_frames"], 1)
    absent = max(totals["frames_evaluated"] - totals["gt_present_frames"], 1)
    totals["absent_fp_rate"] = totals["absent_fp_frames"] / absent
    totals["composite"] = (
        totals["success_rate"] * 0.6
        + totals["center_precision"] * 0.2
        + (1.0 - totals["absent_fp_rate"]) * 0.2
    )
    return totals


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    converted_root = args.converted_root.expanduser().resolve()
    entries = resolve_split_sequences(source_root, converted_root, args.modality, args.split, args.val_ratio)
    if args.max_sequences:
        entries = entries[: args.max_sequences]

    manifest = {
        "source_root": str(source_root),
        "converted_root": str(converted_root),
        "modality": args.modality,
        "split": args.split,
        "sequence_count": len(entries),
        "sequence_names": [entry["name"] for entry in entries],
        "config": "" if args.config is None else str(args.config.expanduser().resolve()),
        "snapshot": "" if args.snapshot is None else str(args.snapshot.expanduser().resolve()),
    }
    if args.dry_run:
        if args.output_json:
            write_json(args.output_json.expanduser().resolve(), manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    if args.config is None or args.snapshot is None:
        raise ValueError("--config and --snapshot are required unless --dry-run is used")

    tracker = build_nanotrack(args)
    results = []
    for entry in entries:
        tracker.reset()
        summary = evaluate_sequence(entry, tracker, args)
        results.append(summary)
        if args.per_sequence_dir:
            write_json(args.per_sequence_dir.expanduser().resolve() / entry["name"] / "summary.json", {k: v for k, v in summary.items() if k != "frames"})
            write_json(args.per_sequence_dir.expanduser().resolve() / entry["name"] / "frames.json", {"frames": summary["frames"]})

    payload = {
        **manifest,
        "aggregate": aggregate_results(results),
        "sequence_results": [{k: v for k, v in item.items() if k != "frames"} for item in results],
    }
    if args.output_json:
        write_json(args.output_json.expanduser().resolve(), payload)
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
