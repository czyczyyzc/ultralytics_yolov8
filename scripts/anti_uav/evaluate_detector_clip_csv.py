#!/usr/bin/env python3
"""Evaluate source-frame-indexed detector CSV rows against YOLO frame labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True, help="Exclusive source frame index")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def load_detections(path: Path) -> dict[int, list[dict]]:
    detections: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x, y = float(row["x"]), float(row["y"])
            detections[int(row["source_frame"])].append(
                {
                    "box": (x, y, x + float(row["width"]), y + float(row["height"])),
                    "confidence": float(row["confidence"]),
                }
            )
    return detections


def load_ground_truth(path: Path, width: int, height: int) -> list[tuple[float, ...]]:
    if not path.is_file() or not path.read_text().strip():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, center_x, center_y, box_width, box_height = map(float, fields[:5])
        boxes.append(
            (
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            )
        )
    return boxes


def match_frame(ground_truth: list[tuple[float, ...]], detections: list[dict], threshold: float) -> tuple[int, int, int]:
    pairs = sorted(
        (
            (box_iou(gt_box, detection["box"]), gt_index, detection_index)
            for gt_index, gt_box in enumerate(ground_truth)
            for detection_index, detection in enumerate(detections)
        ),
        reverse=True,
    )
    matched_gt: set[int] = set()
    matched_detections: set[int] = set()
    for overlap, gt_index, detection_index in pairs:
        if overlap < threshold:
            break
        if gt_index in matched_gt or detection_index in matched_detections:
            continue
        matched_gt.add(gt_index)
        matched_detections.add(detection_index)
    true_positive = len(matched_gt)
    return true_positive, len(detections) - true_positive, len(ground_truth) - true_positive


def main() -> None:
    args = parse_args()
    if args.end_frame <= args.start_frame or args.width <= 0 or args.height <= 0:
        raise ValueError("Invalid frame range or image dimensions")
    detections = load_detections(args.detections)
    totals = {"tp": 0, "fp": 0, "fn": 0, "positive_frames": 0, "absent_frames": 0}
    per_frame = []
    for source_frame in range(args.start_frame, args.end_frame):
        labels = args.labels_dir / f"{source_frame:06d}.txt"
        ground_truth = load_ground_truth(labels, args.width, args.height)
        rows = detections.get(source_frame, [])
        tp, fp, fn = match_frame(ground_truth, rows, args.iou)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["positive_frames" if ground_truth else "absent_frames"] += 1
        per_frame.append({"source_frame": source_frame, "tp": tp, "fp": fp, "fn": fn})

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    payload = {
        "detections": str(args.detections.resolve()),
        "labels_dir": str(args.labels_dir.resolve()),
        "source_frame_range": [args.start_frame, args.end_frame],
        "image_size": [args.width, args.height],
        "iou_threshold": args.iou,
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "per_frame": per_frame,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
