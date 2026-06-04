#!/usr/bin/env python3
"""Evaluate ONNX/RKNN detector outputs on YOLO-label image manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import cv2
import numpy as np


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.anti_uav_rk3588 import YoloBoardBackend, parse_hw  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Detector .onnx or .rknn path.")
    parser.add_argument("--images", type=Path, required=True, help="Image directory or txt manifest.")
    parser.add_argument("--labels", type=Path, default=None, help="Optional label directory. Defaults to image path inference.")
    parser.add_argument("--input-size", default="640,640", help="Detector input size as H,W.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold used before metric accumulation.")
    parser.add_argument("--metric-conf", type=float, default=0.25, help="Confidence threshold for point precision/recall.")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--pre-nms-topk", type=int, default=2000, help="Maximum candidates kept per image before NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections kept per image after NMS.")
    parser.add_argument("--limit", type=int, default=0, help="Optional image cap.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output summary JSON.")
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        root = path.parent
        images = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            item = Path(raw)
            images.append(item if item.is_absolute() else (root / item).resolve())
        return images
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in suffixes)


def infer_label_path(image_path: Path, labels_root: Path | None) -> Path:
    if labels_root is not None:
        return labels_root.expanduser().resolve() / f"{image_path.stem}.txt"
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_yolo_labels(label_path: Path, image_shape: tuple[int, int]) -> list[dict]:
    if not label_path.exists():
        return []
    height, width = image_shape
    labels = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, box_w, box_h = (float(value) for value in parts[1:5])
        x1 = (cx - box_w / 2.0) * width
        y1 = (cy - box_h / 2.0) * height
        x2 = (cx + box_w / 2.0) * width
        y2 = (cy + box_h / 2.0) * height
        if x2 <= x1 or y2 <= y1:
            continue
        labels.append({"class_id": class_id, "bbox": np.array([x1, y1, x2, y2], dtype=np.float32)})
    return labels


def bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0.0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        ious = bbox_iou_matrix(boxes[index : index + 1], boxes[order[1:]]).reshape(-1)
        order = order[1:][ious <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def apply_classwise_nms(boxes: np.ndarray, class_ids: np.ndarray, scores: np.ndarray, iou_thresh: float):
    kept = []
    for class_id in sorted(set(class_ids.tolist())):
        indices = np.where(class_ids == class_id)[0]
        nms_indices = nms_boxes(boxes[indices], scores[indices], iou_thresh)
        kept.extend(indices[nms_indices].tolist())
    kept = np.array(sorted(kept, key=lambda idx: float(scores[idx]), reverse=True), dtype=np.int64)
    return boxes[kept], class_ids[kept], scores[kept]


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    recall_grid = np.linspace(0.0, 1.0, 101)
    precision_interp = np.zeros_like(recall_grid)
    for index, recall_value in enumerate(recall_grid):
        mask = recalls >= recall_value
        precision_interp[index] = np.max(precisions[mask]) if np.any(mask) else 0.0
    return float(np.mean(precision_interp))


def evaluate_predictions(predictions: list[dict], gt_by_image: dict[int, list[dict]], iou_thresh: float) -> dict:
    gt_count = sum(len(items) for items in gt_by_image.values())
    matched: dict[int, set[int]] = {image_index: set() for image_index in gt_by_image}
    tp = np.zeros(len(predictions), dtype=np.float32)
    fp = np.zeros(len(predictions), dtype=np.float32)

    for pred_index, pred in enumerate(predictions):
        gt_items = gt_by_image.get(pred["image_index"], [])
        candidates = [(idx, item) for idx, item in enumerate(gt_items) if item["class_id"] == pred["class_id"]]
        if not candidates:
            fp[pred_index] = 1.0
            continue
        gt_boxes = np.stack([item["bbox"] for _, item in candidates], axis=0)
        ious = bbox_iou_matrix(pred["bbox"][None], gt_boxes).reshape(-1)
        best_pos = int(np.argmax(ious))
        best_gt_index = candidates[best_pos][0]
        if ious[best_pos] >= iou_thresh and best_gt_index not in matched[pred["image_index"]]:
            tp[pred_index] = 1.0
            matched[pred["image_index"]].add(best_gt_index)
        else:
            fp[pred_index] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(gt_count, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1.0)
    return {
        "ap": compute_ap(recalls, precisions),
        "tp": int(cum_tp[-1]) if cum_tp.size else 0,
        "fp": int(cum_fp[-1]) if cum_fp.size else 0,
        "fn": int(max(gt_count - (cum_tp[-1] if cum_tp.size else 0), 0)),
    }


def point_metrics(predictions: list[dict], gt_by_image: dict[int, list[dict]], metric_conf: float) -> dict:
    filtered = [pred for pred in predictions if pred["score"] >= metric_conf]
    result = evaluate_predictions(filtered, gt_by_image, 0.5)
    tp, fp, fn = result["tp"], result["fp"], result["fn"]
    return {
        "conf": metric_conf,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    images = iter_images(args.images)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images found from: {args.images}")

    backend = YoloBoardBackend(
        args.model.expanduser().resolve(),
        input_hw=parse_hw(args.input_size),
        conf_thresh=args.conf,
        postprocess_backend="python",
    )
    predictions: list[dict] = []
    gt_by_image: dict[int, list[dict]] = {}
    failures: list[dict] = []
    try:
        for image_index, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                failures.append({"image": str(image_path), "error": "imread_failed"})
                continue
            labels = read_yolo_labels(infer_label_path(image_path, args.labels), image.shape[:2])
            gt_by_image[image_index] = labels
            boxes, class_ids, scores = backend.infer(image)
            if args.pre_nms_topk > 0 and scores.shape[0] > args.pre_nms_topk:
                keep = scores.argsort()[::-1][: args.pre_nms_topk]
                boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]
            boxes, class_ids, scores = apply_classwise_nms(boxes, class_ids, scores, args.nms_iou)
            if args.max_det > 0 and scores.shape[0] > args.max_det:
                keep = scores.argsort()[::-1][: args.max_det]
                boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]
            for box, class_id, score in zip(boxes, class_ids, scores):
                predictions.append(
                    {
                        "image_index": image_index,
                        "class_id": int(class_id),
                        "score": float(score),
                        "bbox": box.astype(np.float32),
                    }
                )
    finally:
        backend.release()

    predictions.sort(key=lambda item: item["score"], reverse=True)
    ap50 = evaluate_predictions(predictions, gt_by_image, 0.5)
    ap_by_thresh = {f"{thr:.2f}": evaluate_predictions(predictions, gt_by_image, float(thr))["ap"] for thr in np.arange(0.5, 1.0, 0.05)}
    summary = {
        "model": str(args.model.expanduser().resolve()),
        "images": str(args.images.expanduser().resolve()),
        "image_count": len(images),
        "gt_count": sum(len(items) for items in gt_by_image.values()),
        "prediction_count": len(predictions),
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "ap50": ap50["ap"],
        "map50_95": float(np.mean(list(ap_by_thresh.values()))) if ap_by_thresh else 0.0,
        "ap_by_iou": ap_by_thresh,
        "point_metrics": point_metrics(predictions, gt_by_image, args.metric_conf),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
