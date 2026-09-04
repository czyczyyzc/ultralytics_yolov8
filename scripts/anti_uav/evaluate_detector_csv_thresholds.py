#!/usr/bin/env python3
"""Sweep confidence thresholds for native detector CSV output against Anti-UAV GT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


Box = tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            0.01,
            0.02,
            0.03,
            0.05,
            0.08,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.70,
        ],
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    predictions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x = float(row["x"])
            y = float(row["y"])
            predictions[int(row["frame"])].append(
                {
                    "box": (x, y, x + float(row["width"]), y + float(row["height"])),
                    "confidence": float(row["confidence"]),
                }
            )
    return predictions


def box_iou(lhs: Box, rhs: Box) -> float:
    x1 = max(lhs[0], rhs[0])
    y1 = max(lhs[1], rhs[1])
    x2 = min(lhs[2], rhs[2])
    y2 = min(lhs[3], rhs[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    lhs_area = max(0.0, lhs[2] - lhs[0]) * max(0.0, lhs[3] - lhs[1])
    rhs_area = max(0.0, rhs[2] - rhs[0]) * max(0.0, rhs[3] - rhs[1])
    return intersection / max(lhs_area + rhs_area - intersection, 1e-12)


def ground_truth_box(annotation: dict[str, Any], frame: int) -> Box | None:
    if not annotation["exist"][frame]:
        return None
    x, y, width, height = (float(value) for value in annotation["gt_rect"][frame])
    return x, y, x + width, y + height


def evaluate_threshold(
    predictions: dict[int, list[dict[str, Any]]],
    annotation: dict[str, Any],
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    selected_detections = 0
    detection_frames = 0
    absent_frames = 0
    absent_detection_frames = 0
    matched_ious: list[float] = []

    for frame in range(len(annotation["exist"])):
        selected = [
            row for row in predictions.get(frame, []) if row["confidence"] >= confidence_threshold
        ]
        selected_detections += len(selected)
        detection_frames += bool(selected)
        ground_truth = ground_truth_box(annotation, frame)
        if ground_truth is None:
            absent_frames += 1
            absent_detection_frames += bool(selected)
            false_positive += len(selected)
            continue

        overlaps = [box_iou(row["box"], ground_truth) for row in selected]
        best_iou = max(overlaps, default=0.0)
        if best_iou >= iou_threshold:
            true_positive += 1
            false_positive += len(selected) - 1
            matched_ious.append(best_iou)
        else:
            false_negative += 1
            false_positive += len(selected)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "confidence_threshold": confidence_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "selected_detections": selected_detections,
        "detection_frames": detection_frames,
        "absent_detection_frames": absent_detection_frames,
        "absent_false_output_rate": absent_detection_frames / max(absent_frames, 1),
        "mean_matched_iou": mean(matched_ious) if matched_ious else 0.0,
    }


def main() -> None:
    args = parse_args()
    annotation = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    if len(annotation["exist"]) != len(annotation["gt_rect"]):
        raise ValueError("Ground-truth exist and gt_rect lengths differ")

    predictions = load_predictions(args.predictions)
    rows = [
        evaluate_threshold(predictions, annotation, threshold, args.iou)
        for threshold in sorted(set(args.thresholds))
    ]
    best = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    payload = {
        "schema_version": "anti_uav.native_detector_threshold_sweep.v1",
        "predictions_csv": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "ground_truth": str(args.ground_truth.resolve()),
        "ground_truth_sha256": sha256_file(args.ground_truth),
        "protocol": {
            "frame_count": len(annotation["exist"]),
            "visible_frame_count": sum(bool(value) for value in annotation["exist"]),
            "absent_frame_count": sum(not bool(value) for value in annotation["exist"]),
            "iou_threshold": args.iou,
            "matching": "single_gt_best_iou; extra_detections_are_false_positives",
        },
        "best_f1": best,
        "thresholds": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
