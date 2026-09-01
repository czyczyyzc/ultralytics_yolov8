#!/usr/bin/env python3
"""Measure detector coverage and continuity on an unlabeled fixed gray video."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10, 0.25])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.thresholds or min(args.thresholds) <= 0.0:
        raise ValueError("Thresholds must be positive")
    from ultralytics import YOLO

    thresholds = sorted(set(args.thresholds))
    counts = {threshold: 0 for threshold in thresholds}
    longest_gaps = {threshold: 0 for threshold in thresholds}
    current_gaps = {threshold: 0 for threshold in thresholds}
    max_confidences: list[float] = []
    total_detections = 0
    start = time.time()
    model = YOLO(str(args.model))
    for result in model.predict(
        source=str(args.video),
        imgsz=args.imgsz,
        conf=min(thresholds),
        iou=0.45,
        max_det=20,
        device=args.device,
        stream=True,
        verbose=False,
    ):
        confidences = [] if result.boxes is None else result.boxes.conf.cpu().tolist()
        total_detections += len(confidences)
        maximum = max(confidences, default=0.0)
        max_confidences.append(maximum)
        for threshold in thresholds:
            if maximum >= threshold:
                counts[threshold] += 1
                current_gaps[threshold] = 0
            else:
                current_gaps[threshold] += 1
                longest_gaps[threshold] = max(longest_gaps[threshold], current_gaps[threshold])

    elapsed = time.time() - start
    frames = len(max_confidences)
    output = {
        "schema_version": "anti_uav.unlabeled_gray_video_coverage.v1",
        "warning": (
            "Proxy metrics only; this video has no human ground truth and these values are not "
            "mAP/precision/recall."
        ),
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "video": str(args.video),
        "video_sha256": sha256_file(args.video),
        "input_height_width": args.imgsz,
        "frames": frames,
        "elapsed_sec": elapsed,
        "pipeline_fps": frames / max(elapsed, 1e-12),
        "total_detections_at_minimum_threshold": total_detections,
        "mean_top_confidence": sum(max_confidences) / max(frames, 1),
        "thresholds": {
            f"{threshold:.2f}": {
                "detection_frames": counts[threshold],
                "frame_coverage": counts[threshold] / max(frames, 1),
                "longest_gap_frames": longest_gaps[threshold],
            }
            for threshold in thresholds
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
