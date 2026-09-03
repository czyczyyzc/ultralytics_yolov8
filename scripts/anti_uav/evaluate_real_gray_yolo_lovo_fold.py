#!/usr/bin/env python3
"""Evaluate one YOLO model on a complete gray holdout and Anti-UAV300 RGB validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--rgb-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.10, 0.25, 0.45])
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--skip-rgb", action="store_true")
    return parser.parse_args()


def standard_metrics(model: YOLO, data: Path, args: argparse.Namespace) -> dict[str, float]:
    result = model.val(
        data=str(data),
        imgsz=[544, 960],
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        save_json=False,
        verbose=False,
        rect=False,
    )
    return {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
    }


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_gt(image_path: Path, width: int, height: int) -> np.ndarray | None:
    label = label_path(image_path)
    text = label.read_text().strip()
    if not text:
        return None
    fields = [float(value) for value in text.splitlines()[0].split()]
    _, center_x, center_y, box_width, box_height = fields
    x1 = (center_x - box_width * 0.5) * width
    y1 = (center_y - box_height * 0.5) * height
    x2 = (center_x + box_width * 0.5) * width
    y2 = (center_y + box_height * 0.5) * height
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_one_to_many(gt: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    intersection_x1 = np.maximum(gt[0], boxes[:, 0])
    intersection_y1 = np.maximum(gt[1], boxes[:, 1])
    intersection_x2 = np.minimum(gt[2], boxes[:, 2])
    intersection_y2 = np.minimum(gt[3], boxes[:, 3])
    intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
        0.0, intersection_y2 - intersection_y1
    )
    gt_area = max(0.0, float(gt[2] - gt[0])) * max(0.0, float(gt[3] - gt[1]))
    box_areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(gt_area + box_areas - intersection, 1e-12)


def fixed_threshold_metrics(model: YOLO, image_list: Path, args: argparse.Namespace) -> dict[str, dict]:
    images = [Path(line.strip()) for line in image_list.read_text().splitlines() if line.strip()]
    minimum_confidence = min(args.thresholds)
    counters = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "positive_frames": 0,
            "absent_frames": 0,
            "false_positive_absent_frames": 0,
            "detection_frames": 0,
            "matched_ious": [],
        }
        for threshold in args.thresholds
    }
    predictions = model.predict(
        source=str(image_list),
        imgsz=[544, 960],
        conf=minimum_confidence,
        iou=0.45,
        max_det=100,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
        rect=False,
    )
    seen = 0
    for result in predictions:
        image_path = Path(result.path)
        height, width = result.orig_shape
        gt = load_gt(image_path, width, height)
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4), dtype=np.float32)
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty((0,), dtype=np.float32)
        for threshold, counter in counters.items():
            selected = boxes[confidences >= threshold]
            if len(selected):
                counter["detection_frames"] += 1
            if gt is None:
                counter["absent_frames"] += 1
                counter["fp"] += len(selected)
                if len(selected):
                    counter["false_positive_absent_frames"] += 1
                continue
            counter["positive_frames"] += 1
            if not len(selected):
                counter["fn"] += 1
                continue
            overlaps = iou_one_to_many(gt, selected)
            best = float(overlaps.max())
            if best >= args.iou:
                counter["tp"] += 1
                counter["fp"] += len(selected) - 1
                counter["matched_ious"].append(best)
            else:
                counter["fn"] += 1
                counter["fp"] += len(selected)
        seen += 1
    if seen != len(images):
        raise RuntimeError(f"Predicted {seen} images, expected {len(images)}")

    output: dict[str, dict] = {}
    for threshold, counter in counters.items():
        tp, fp, fn = counter["tp"], counter["fp"], counter["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        absent_frames = counter["absent_frames"]
        output[f"{threshold:.2f}"] = {
            **{key: value for key, value in counter.items() if key != "matched_ious"},
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "absent_frame_false_positive_rate": counter["false_positive_absent_frames"] / max(absent_frames, 1),
            "mean_matched_iou": float(np.mean(counter["matched_ious"])) if counter["matched_ious"] else 0.0,
        }
    return output


def main() -> None:
    args = parse_args()
    holdout_data = args.fold_dir / "holdout_all.yaml"
    holdout_images = args.fold_dir / "holdout_all_frames.txt"
    model = YOLO(str(args.model))
    output = {
        "model": str(args.model),
        "fold": args.fold_dir.name,
        "input_height_width": [544, 960],
        "gray_holdout_standard": standard_metrics(model, holdout_data, args),
        "gray_holdout_fixed_thresholds": fixed_threshold_metrics(model, holdout_images, args),
    }
    if not args.skip_rgb:
        output["anti_uav300_rgb_standard"] = standard_metrics(model, args.rgb_data, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
