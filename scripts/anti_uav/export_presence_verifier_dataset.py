#!/usr/bin/env python3
"""Export lightweight presence-verifier samples from replay state logs and ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.replay_eval import load_ground_truth
from ultralytics import solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-jsonl", type=Path, required=True, help="Replay state log produced by AntiUAVSystem.")
    parser.add_argument("--annotations", type=Path, required=True, help="Ground-truth annotation file.")
    parser.add_argument(
        "--dataset-format",
        default="anti-uav-json",
        choices=("anti-uav-json", "drone-vs-bird-txt", "jsonl-bbox"),
        help="Ground-truth annotation format.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL of labeled feature vectors.")
    parser.add_argument("--sequence-name", default="", help="Optional sequence name stored in each sample.")
    parser.add_argument("--positive-iou", type=float, default=0.5, help="IoU threshold for positive labels.")
    parser.add_argument("--negative-iou", type=float, default=0.2, help="IoU threshold for negative labels.")
    return parser.parse_args()


def bbox_iou(box1: list[float], box2: tuple[float, float, float, float]) -> float:
    """Compute IoU between logged xyxy boxes and GT xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(value) for value in box1[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in box2[:4]]
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


def iter_state_rows(path: Path):
    """Yield parsed JSON rows from a replay state log."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                yield json.loads(raw)


def main() -> None:
    args = parse_args()
    sequence_name = args.sequence_name or args.states_jsonl.stem
    ground_truth = load_ground_truth(args.annotations.expanduser().resolve(), args.dataset_format)
    feature_names = tuple(solutions.HeuristicPresenceVerifier.feature_names)

    args.output_jsonl = args.output_jsonl.expanduser().resolve()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    counts = {"written": 0, "positive": 0, "negative": 0, "skipped": 0}
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in iter_state_rows(args.states_jsonl.expanduser().resolve()):
            bbox = row.get("bbox")
            features = row.get("presence_features")
            frame_index = int(row.get("frame_index", 0))
            if not bbox or not features:
                counts["skipped"] += 1
                continue

            gt_bbox = ground_truth.get(frame_index)
            label = None
            iou = 0.0
            if gt_bbox is None:
                label = 0
            else:
                iou = bbox_iou(bbox, gt_bbox)
                if iou >= args.positive_iou:
                    label = 1
                elif iou <= args.negative_iou:
                    label = 0

            if label is None:
                counts["skipped"] += 1
                continue

            payload = {
                "sequence": sequence_name,
                "frame_index": frame_index,
                "label": int(label),
                "iou": float(iou),
                "track_score": float(row.get("track_score", 0.0)),
                "presence_score": float(row.get("presence_score", 0.0)),
                "features": {name: float(features.get(name, 0.0)) for name in feature_names},
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts["written"] += 1
            counts["positive" if label == 1 else "negative"] += 1

    print(json.dumps({"sequence": sequence_name, **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
