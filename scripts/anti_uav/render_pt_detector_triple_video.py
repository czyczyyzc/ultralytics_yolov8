#!/usr/bin/env python3
"""Render a synchronized, ground-truth-aware comparison of three YOLO models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.render_pt_detector_pair_video import (
    infer,
    load_gt,
    make_panel,
    metrics,
    read_manifest,
    sha256,
    source_fps,
    text_with_shadow,
)


MODEL_COLORS = ((40, 150, 255), (255, 210, 40), (70, 225, 225))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Ordered image manifest")
    parser.add_argument("--source-video", type=Path, help="Original video, used only for output FPS")
    parser.add_argument("--models", type=Path, nargs=3, required=True)
    parser.add_argument(
        "--labels",
        nargs=3,
        default=("P3 (standard)", "P2 (full retrain)", "Frozen-P3 + Add-on P2"),
    )
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, nargs=2, default=(544, 960), metavar=("H", "W"))
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fps", type=float, help="Override output FPS")
    parser.add_argument("--panel-size", type=int, nargs=2, default=(960, 540), metavar=("W", "H"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = read_manifest(args.images)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    inference_seconds = []
    for label, model in zip(args.labels, args.models):
        print(f"Inferring {label} on {len(images)} frames: {model}", flush=True)
        predictions, seconds = infer(model, args.images, args)
        all_predictions.append(predictions)
        inference_seconds.append(seconds)

    fps = source_fps(args.source_video, args.fps)
    panel_size = tuple(args.panel_size)
    output_size = (panel_size[0] * 3, panel_size[1])
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.output_video}")

    totals = [{"tp": 0, "fp": 0, "fn": 0} for _ in args.models]
    absent_fp_frames = [0 for _ in args.models]
    started = time.perf_counter()
    for index, image_path in enumerate(images):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read {image_path}")
        gt = load_gt(image_path, frame.shape[1], frame.shape[0])
        panels = []
        for model_index, (label, color, predictions) in enumerate(
            zip(args.labels, MODEL_COLORS, all_predictions)
        ):
            pred = predictions.get(str(image_path), np.empty((0, 5), dtype=np.float32))
            panel, counts = make_panel(frame, gt, pred, label, color, panel_size, args.match_iou)
            for key, value in zip(("tp", "fp", "fn"), counts):
                totals[model_index][key] += value
            if len(gt) == 0 and counts[1] > 0:
                absent_fp_frames[model_index] += 1
            panels.append(panel)

        combined = np.concatenate(panels, axis=1)
        for divider in (panel_size[0], panel_size[0] * 2):
            cv2.line(combined, (divider, 0), (divider, panel_size[1]), (255, 255, 255), 2)
        footer = (
            f"Video00004  |  frame {index + 1:04d}/{len(images)}  |  {index / fps:06.2f}s"
            f"  |  input {args.imgsz[1]}x{args.imgsz[0]}  conf {args.conf:.2f}"
        )
        text_with_shadow(combined, footer, (950, panel_size[1] - 14), 0.52, (245, 245, 245), 1)
        writer.write(combined)
        if (index + 1) % 250 == 0 or index + 1 == len(images):
            print(f"Rendered {index + 1}/{len(images)} frames", flush=True)
    writer.release()

    models = []
    for index, (label, model) in enumerate(zip(args.labels, args.models)):
        models.append(
            {
                "label": label,
                "model": str(model.resolve()),
                "sha256": sha256(model),
                "inference_seconds": inference_seconds[index],
                "metrics": metrics(totals[index]),
                "absent_fp_frames": absent_fp_frames[index],
            }
        )
    summary = {
        "images_manifest": str(args.images.resolve()),
        "source_video": str(args.source_video.resolve()) if args.source_video else None,
        "frame_count": len(images),
        "fps": fps,
        "output_resolution": list(output_size),
        "input_resolution": [args.imgsz[1], args.imgsz[0]],
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "visualization_style": "clean_native_crop_corner_boxes_v2",
        "models": models,
        "render_seconds": time.perf_counter() - started,
        "output_video": str(args.output_video.resolve()),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
