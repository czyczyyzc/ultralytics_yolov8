#!/usr/bin/env python3
"""Mine missed, poorly localized, and weak positive training images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--conf", type=float, default=0.05, help="PT mining confidence calibrated to RKNN conf=0.01.")
    parser.add_argument("--weak-conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--gray-only",
        action="store_true",
        help="Mine only gray positives while leaving RGB images available to the rehearsal mix.",
    )
    return parser.parse_args()


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as error:
        raise ValueError(f"Image path has no images component: {image_path}") from error
    return Path(*parts).with_suffix(".txt")


def read_normalized_boxes(image_path: Path) -> np.ndarray:
    label = label_path(image_path)
    rows: list[list[float]] = []
    for line in label.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"Malformed YOLO label in {label}: {line}")
        _, center_x, center_y, width, height = map(float, fields[:5])
        rows.append(
            [
                center_x - width * 0.5,
                center_y - height * 0.5,
                center_x + width * 0.5,
                center_y + height * 0.5,
            ]
        )
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def pairwise_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=np.float32)
    intersection_min = np.maximum(first[:, None, :2], second[None, :, :2])
    intersection_max = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = np.maximum(0.0, intersection_max - intersection_min)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    first_area = np.prod(np.maximum(0.0, first[:, 2:] - first[:, :2]), axis=1)
    second_area = np.prod(np.maximum(0.0, second[:, 2:] - second[:, :2]), axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def greedy_matches(ious: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    candidates = [
        (float(ious[gt_index, pred_index]), gt_index, pred_index)
        for gt_index in range(ious.shape[0])
        for pred_index in range(ious.shape[1])
        if ious[gt_index, pred_index] >= threshold
    ]
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for overlap, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, overlap))
    return matches


def classify_difficulty(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    confidences: np.ndarray,
    iou_threshold: float,
    weak_confidence: float,
) -> dict:
    if not len(gt_boxes):
        raise ValueError("Hard-positive mining requires at least one ground-truth box")
    if not len(pred_boxes):
        return {
            "category": "missed",
            "extra_repeats": 3,
            "matched_gt": 0,
            "best_iou": 0.0,
            "weakest_matched_confidence": 0.0,
        }

    ious = pairwise_iou(gt_boxes, pred_boxes)
    matches = greedy_matches(ious, iou_threshold)
    best_iou = float(ious.max())
    if len(matches) < len(gt_boxes):
        return {
            "category": "localization" if not matches else "partial",
            "extra_repeats": 3 if not matches else 2,
            "matched_gt": len(matches),
            "best_iou": best_iou,
            "weakest_matched_confidence": min((float(confidences[pred]) for _, pred, _ in matches), default=0.0),
        }

    weakest_confidence = min(float(confidences[pred]) for _, pred, _ in matches)
    if weakest_confidence < weak_confidence:
        return {
            "category": "weak",
            "extra_repeats": 1,
            "matched_gt": len(matches),
            "best_iou": best_iou,
            "weakest_matched_confidence": weakest_confidence,
        }
    return {
        "category": "easy",
        "extra_repeats": 0,
        "matched_gt": len(matches),
        "best_iou": best_iou,
        "weakest_matched_confidence": weakest_confidence,
    }


def source_group(path: Path) -> str:
    parts = path.parts
    if "gray" in parts:
        index = parts.index("gray")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "rgb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.conf < args.weak_conf <= 1.0:
        raise ValueError("Require 0 < conf < weak-conf <= 1")
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be in (0, 1]")

    source_paths = [Path(line.strip()) for line in args.train_list.read_text().splitlines() if line.strip()]
    unique_paths = list(dict.fromkeys(source_paths))
    all_positives = [path for path in unique_paths if len(read_normalized_boxes(path))]
    positives = [path for path in all_positives if not args.gray_only or source_group(path) != "rgb"]
    args.output.mkdir(parents=True, exist_ok=True)
    positive_list = args.output / "positive_unique.txt"
    positive_list.write_text("".join(f"{path}\n" for path in positives))

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    predictions = model.predict(
        source=str(positive_list),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.45,
        max_det=args.max_det,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    )

    hard_records: list[dict] = []
    categories: Counter[str] = Counter()
    seen = 0
    for result in predictions:
        image_path = Path(result.path)
        gt_boxes = read_normalized_boxes(image_path)
        height, width = result.orig_shape
        if result.boxes is None:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
        else:
            pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_boxes[:, [0, 2]] /= width
            pred_boxes[:, [1, 3]] /= height
            confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
        difficulty = classify_difficulty(gt_boxes, pred_boxes, confidences, args.iou, args.weak_conf)
        categories[difficulty["category"]] += 1
        if difficulty["category"] != "easy":
            hard_records.append(
                {
                    "image": str(image_path),
                    "group": source_group(image_path),
                    "gt_count": len(gt_boxes),
                    "prediction_count": len(pred_boxes),
                    **difficulty,
                }
            )
        seen += 1
    if seen != len(positives):
        raise RuntimeError(f"Predicted {seen} unique positives, expected {len(positives)}")

    jsonl = args.output / "hard_positives.jsonl"
    jsonl.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in hard_records))
    (args.output / "hard_positive_images.txt").write_text("".join(f"{record['image']}\n" for record in hard_records))
    summary = {
        "schema_version": "anti_uav.hard_positive_mining.v1",
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "train_list": str(args.train_list),
        "input_height_width": args.imgsz,
        "mining_confidence": args.conf,
        "weak_confidence": args.weak_conf,
        "iou_threshold": args.iou,
        "unique_training_images": len(unique_paths),
        "unique_positive_images_before_scope_filter": len(all_positives),
        "unique_positive_images": len(positives),
        "gray_only": args.gray_only,
        "hard_positive_images": len(hard_records),
        "categories": dict(sorted(categories.items())),
        "requested_extra_rehearsal_slots": sum(record["extra_repeats"] for record in hard_records),
        "hard_positive_jsonl": str(jsonl),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
