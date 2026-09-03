#!/usr/bin/env python3
"""Analyze positive-frame detector misses by size, localization, and image quality."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--positive-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--probe-conf", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--montage-count", type=int, default=16)
    return parser.parse_args()


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_gt(image_path: Path, width: int, height: int) -> np.ndarray:
    fields = [float(value) for value in label_path(image_path).read_text().splitlines()[0].split()]
    _, center_x, center_y, box_width, box_height = fields
    return np.array(
        [
            (center_x - box_width * 0.5) * width,
            (center_y - box_height * 0.5) * height,
            (center_x + box_width * 0.5) * width,
            (center_y + box_height * 0.5) * height,
        ],
        dtype=np.float32,
    )


def iou_one_to_many(gt: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty((0,), dtype=np.float32)
    top_left = np.maximum(gt[:2], boxes[:, :2])
    bottom_right = np.minimum(gt[2:], boxes[:, 2:])
    intersection_wh = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    gt_area = max(float(gt[2] - gt[0]), 0.0) * max(float(gt[3] - gt[1]), 0.0)
    box_areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    return intersection / np.maximum(gt_area + box_areas - intersection, 1e-12)


def image_quality(image_path: Path, gt: np.ndarray) -> dict[str, float]:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Unable to read {image_path}")
    height, width = gray.shape
    x1, y1, x2, y2 = gt
    ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    ix2, iy2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    target = gray[iy1:iy2, ix1:ix2]
    box_width, box_height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    margin_x, margin_y = max(12, int(round(box_width * 3))), max(12, int(round(box_height * 3)))
    rx1, ry1 = max(0, ix1 - margin_x), max(0, iy1 - margin_y)
    rx2, ry2 = min(width, ix2 + margin_x), min(height, iy2 + margin_y)
    local = gray[ry1:ry2, rx1:rx2]
    mask = np.ones(local.shape, dtype=bool)
    mask[iy1 - ry1 : iy2 - ry1, ix1 - rx1 : ix2 - rx1] = False
    ring = local[mask]
    target_mean = float(target.mean()) if target.size else 0.0
    ring_mean = float(ring.mean()) if ring.size else target_mean
    ring_std = float(ring.std()) if ring.size else 0.0
    return {
        "target_mean": target_mean,
        "background_mean": ring_mean,
        "absolute_contrast": abs(target_mean - ring_mean),
        "local_snr": abs(target_mean - ring_mean) / max(ring_std, 1e-6),
        "local_std": ring_std,
        "local_laplacian_variance": float(cv2.Laplacian(local, cv2.CV_64F).var()),
    }


def classify_miss(
    confidences: np.ndarray,
    overlaps: np.ndarray,
    threshold: float,
    match_iou: float,
) -> str:
    matched = overlaps >= match_iou
    if np.any(matched & (confidences >= threshold)):
        return "hit"
    if np.any(matched):
        return "low_confidence"
    if len(overlaps) and float(overlaps.max()) >= 0.10:
        return "localization"
    return "no_target_candidate"


def contiguous_runs(rows: list[dict]) -> list[dict]:
    misses = sorted((row for row in rows if row["outcome"] != "hit"), key=lambda row: row["frame"])
    if not misses:
        return []
    runs: list[list[dict]] = [[misses[0]]]
    for row in misses[1:]:
        if row["frame"] == runs[-1][-1]["frame"] + 1:
            runs[-1].append(row)
        else:
            runs.append([row])
    return [
        {
            "start": run[0]["frame"],
            "end": run[-1]["frame"],
            "length": len(run),
            "outcomes": dict(Counter(row["outcome"] for row in run)),
            "median_input_max_side": float(np.median([row["input_max_side"] for row in run])),
        }
        for run in sorted(runs, key=len, reverse=True)
    ]


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
    }


def size_recall(rows: list[dict]) -> list[dict]:
    edges = [(0.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0), (12.0, float("inf"))]
    output = []
    for lower, upper in edges:
        selected = [row for row in rows if lower <= row["input_max_side"] < upper]
        if not selected:
            continue
        hits = sum(row["outcome"] == "hit" for row in selected)
        output.append(
            {
                "input_max_side_range": f"[{lower:g},{upper:g})",
                "frames": len(selected),
                "hits": hits,
                "recall": hits / len(selected),
            }
        )
    return output


def draw_montage(rows: list[dict], output: Path, count: int) -> list[int]:
    misses = [row for row in rows if row["outcome"] != "hit"]
    runs = contiguous_runs(rows)
    selected: list[dict] = []
    by_frame = {row["frame"]: row for row in misses}
    for run in runs[:6]:
        middle = (run["start"] + run["end"]) // 2
        nearest = min(range(run["start"], run["end"] + 1), key=lambda frame: abs(frame - middle))
        selected.append(by_frame[nearest])
    for key in ("input_max_side", "absolute_contrast", "best_overlap_confidence"):
        selected.extend(sorted(misses, key=lambda row: row[key])[:6])
    unique = []
    seen = set()
    for row in selected:
        if row["frame"] in seen:
            continue
        seen.add(row["frame"])
        unique.append(row)
        if len(unique) == count:
            break

    panel_width, panel_height = 640, 400
    columns = 4
    rows_count = max(1, int(np.ceil(len(unique) / columns)))
    montage = np.full((rows_count * panel_height, columns * panel_width, 3), 24, dtype=np.uint8)
    for index, row in enumerate(unique):
        image = cv2.imread(row["image"])
        if image is None:
            continue
        original_height, original_width = image.shape[:2]
        panel = cv2.resize(image, (panel_width, 360), interpolation=cv2.INTER_AREA)
        scale_x, scale_y = panel_width / original_width, 360 / original_height
        gt = np.asarray(row["gt_xyxy"], dtype=float)
        gt_panel = np.rint(gt * [scale_x, scale_y, scale_x, scale_y]).astype(int)
        cv2.rectangle(panel, tuple(gt_panel[:2]), tuple(gt_panel[2:]), (0, 0, 255), 2)

        prediction = row["best_prediction_xyxy"]
        if prediction is not None:
            pred = np.asarray(prediction, dtype=float)
            pred_panel = np.rint(pred * [scale_x, scale_y, scale_x, scale_y]).astype(int)
            cv2.rectangle(panel, tuple(pred_panel[:2]), tuple(pred_panel[2:]), (255, 180, 0), 2)

        center_x, center_y = (gt[0] + gt[2]) * 0.5, (gt[1] + gt[3]) * 0.5
        crop_size = max(96, int(round(max(gt[2] - gt[0], gt[3] - gt[1]) * 12)))
        cx1 = max(0, min(original_width - crop_size, int(round(center_x - crop_size * 0.5))))
        cy1 = max(0, min(original_height - crop_size, int(round(center_y - crop_size * 0.5))))
        cx2, cy2 = min(original_width, cx1 + crop_size), min(original_height, cy1 + crop_size)
        crop = image[cy1:cy2, cx1:cx2].copy()
        if crop.size:
            zoom = cv2.resize(crop, (180, 180), interpolation=cv2.INTER_NEAREST)
            zoom_scale_x, zoom_scale_y = 180 / (cx2 - cx1), 180 / (cy2 - cy1)
            zoom_gt = np.rint(
                (gt - [cx1, cy1, cx1, cy1]) * [zoom_scale_x, zoom_scale_y, zoom_scale_x, zoom_scale_y]
            ).astype(int)
            cv2.rectangle(zoom, tuple(zoom_gt[:2]), tuple(zoom_gt[2:]), (0, 0, 255), 2)
            panel[6:186, panel_width - 186 : panel_width - 6] = zoom

        canvas = np.full((panel_height, panel_width, 3), 24, dtype=np.uint8)
        canvas[:360] = panel
        label = (
            f"f={row['frame']} {row['outcome']} size={row['input_width']:.1f}x"
            f"{row['input_height']:.1f} conf={row['best_overlap_confidence']:.3f} "
            f"IoU={row['best_iou']:.2f}"
        )
        cv2.putText(canvas, label, (8, 387), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
        y, x = divmod(index, columns)
        montage[y * panel_height : (y + 1) * panel_height, x * panel_width : (x + 1) * panel_width] = canvas
    cv2.imwrite(str(output), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return [row["frame"] for row in unique]


def main() -> None:
    args = parse_args()
    if not 0.0 < args.probe_conf < args.threshold:
        raise ValueError("Require 0 < --probe-conf < --threshold")
    image_paths = [Path(line.strip()) for line in args.positive_list.read_text().splitlines() if line.strip()]
    model = YOLO(str(args.model))
    predictions = model.predict(
        source=str(args.positive_list),
        imgsz=args.imgsz,
        conf=args.probe_conf,
        iou=0.45,
        max_det=args.max_det,
        device=args.device,
        batch=args.batch,
        rect=False,
        stream=True,
        verbose=False,
    )
    rows: list[dict] = []
    for image_path, result in zip(image_paths, predictions):
        height, width = result.orig_shape
        gt = load_gt(image_path, width, height)
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty((0,))
        overlaps = iou_one_to_many(gt, boxes)
        outcome = classify_miss(confidences, overlaps, args.threshold, args.match_iou)
        best_index = int(overlaps.argmax()) if len(overlaps) else -1
        matching_confidences = confidences[overlaps >= args.match_iou]
        scale = min(args.imgsz[1] / width, args.imgsz[0] / height)
        gt_width, gt_height = float(gt[2] - gt[0]), float(gt[3] - gt[1])
        row = {
            "frame": int(image_path.stem),
            "image": str(image_path),
            "outcome": outcome,
            "gt_xyxy": [float(value) for value in gt],
            "source_width": gt_width,
            "source_height": gt_height,
            "input_width": gt_width * scale,
            "input_height": gt_height * scale,
            "input_center_x": float((gt[0] + gt[2]) * 0.5 * scale),
            "input_center_y": float((gt[1] + gt[3]) * 0.5 * scale),
            "input_max_side": max(gt_width, gt_height) * scale,
            "input_min_side": min(gt_width, gt_height) * scale,
            "input_area": gt_width * gt_height * scale * scale,
            "input_boundary_distance": float(
                min(gt[0], gt[1], width - gt[2], height - gt[3]) * scale
            ),
            "detections_at_threshold": int(np.sum(confidences >= args.threshold)),
            "best_iou": float(overlaps[best_index]) if best_index >= 0 else 0.0,
            "best_iou_confidence": float(confidences[best_index]) if best_index >= 0 else 0.0,
            "best_overlap_confidence": float(matching_confidences.max()) if len(matching_confidences) else 0.0,
            "best_prediction_xyxy": (
                [float(value) for value in boxes[best_index]]
                if best_index >= 0
                else None
            ),
            **image_quality(image_path, gt),
        }
        rows.append(row)
    if len(rows) != len(image_paths):
        raise RuntimeError(f"Analyzed {len(rows)} frames, expected {len(image_paths)}")

    by_frame = {row["frame"]: row for row in rows}
    for row in rows:
        velocities = []
        for neighbor in (row["frame"] - 1, row["frame"] + 1):
            if neighbor not in by_frame:
                continue
            first = np.asarray([row["input_center_x"], row["input_center_y"]])
            second = np.asarray(
                [by_frame[neighbor]["input_center_x"], by_frame[neighbor]["input_center_y"]]
            )
            velocities.append(float(np.linalg.norm(first - second)))
        row["input_center_motion_per_frame"] = float(np.mean(velocities)) if velocities else 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scalar_fields = [key for key, value in rows[0].items() if not isinstance(value, list)]
    with (args.output_dir / "positive_frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in rows)
    missed = [row for row in rows if row["outcome"] != "hit"]
    with (args.output_dir / "missed_frames.jsonl").open("w") as handle:
        for row in missed:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    summary = {
        "model": str(args.model),
        "positive_list": str(args.positive_list),
        "threshold": args.threshold,
        "probe_confidence": args.probe_conf,
        "match_iou": args.match_iou,
        "positive_frames": len(rows),
        "hits": len(rows) - len(missed),
        "misses": len(missed),
        "recall": (len(rows) - len(missed)) / len(rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "size_recall": size_recall(rows),
        "hit_vs_miss": {
            group: {
                field: summarize([row[field] for row in selected])
                for field in (
                    "input_width",
                    "input_height",
                    "input_max_side",
                    "input_min_side",
                    "input_area",
                    "input_boundary_distance",
                    "absolute_contrast",
                    "local_snr",
                    "local_laplacian_variance",
                    "input_center_motion_per_frame",
                )
            }
            for group, selected in (
                ("hit", [row for row in rows if row["outcome"] == "hit"]),
                ("miss", missed),
            )
        },
        "longest_miss_runs": contiguous_runs(rows)[:20],
    }
    summary["montage_frames"] = draw_montage(
        rows, args.output_dir / "representative_misses.jpg", args.montage_count
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
