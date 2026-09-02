#!/usr/bin/env python3
"""Render a synchronized, ground-truth-aware comparison of two YOLO models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


COLORS = {
    "gt": (80, 220, 80),
    "left": (40, 150, 255),
    "right": (255, 210, 40),
    "ok": (80, 220, 80),
    "bad": (40, 40, 240),
    "clear": (210, 210, 210),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Ordered image manifest")
    parser.add_argument("--source-video", type=Path, help="Original video, used only for output FPS")
    parser.add_argument("--left-model", type=Path, required=True)
    parser.add_argument("--right-model", type=Path, required=True)
    parser.add_argument("--left-label", default="P3 (standard YOLOv8n)")
    parser.add_argument("--right-label", default="P2 (four-scale YOLOv8n)")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[Path]:
    images = [Path(line.strip()).resolve() for line in path.read_text().splitlines() if line.strip()]
    if not images:
        raise ValueError(f"No images found in {path}")
    missing = [image for image in images if not image.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image(s), first: {missing[0]}")
    return images


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as exc:
        raise ValueError(f"Image path has no images component: {image_path}") from exc
    return Path(*parts).with_suffix(".txt")


def load_gt(image_path: Path, width: int, height: int) -> np.ndarray:
    path = label_path(image_path)
    boxes = []
    if path.is_file():
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            _, cx, cy, bw, bh = map(float, fields[:5])
            boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                          (cx + bw / 2) * width, (cy + bh / 2) * height))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def infer(model_path: Path, images_txt: Path, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], float]:
    model = YOLO(str(model_path))
    predictions: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    results = model.predict(
        source=str(images_txt),
        imgsz=list(args.imgsz),
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        batch=args.batch,
        device=args.device,
        stream=True,
        verbose=False,
        rect=False,
    )
    for result in results:
        key = str(Path(result.path).resolve())
        if result.boxes is None or len(result.boxes) == 0:
            predictions[key] = np.empty((0, 5), dtype=np.float32)
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()[:, None]
            predictions[key] = np.concatenate((boxes, scores), axis=1).astype(np.float32)
    return predictions, time.perf_counter() - started


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.empty((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    first_area = np.prod(np.clip(first[:, 2:] - first[:, :2], 0, None), axis=1)
    second_area = np.prod(np.clip(second[:, 2:] - second[:, :2], 0, None), axis=1)
    return intersection / np.clip(first_area[:, None] + second_area[None, :] - intersection, 1e-9, None)


def match_counts(gt: np.ndarray, pred: np.ndarray, threshold: float) -> tuple[int, int, int]:
    if len(gt) == 0:
        return 0, len(pred), 0
    if len(pred) == 0:
        return 0, 0, len(gt)
    pairs = []
    ious = box_iou(gt, pred[:, :4])
    for gt_index, pred_index in zip(*np.where(ious >= threshold)):
        pairs.append((float(ious[gt_index, pred_index]), int(gt_index), int(pred_index)))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gt_index, pred_index in sorted(pairs, reverse=True):
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    tp = len(matched_gt)
    return tp, len(pred) - tp, len(gt) - tp


def scale_boxes(boxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = boxes.copy()
    if len(scaled):
        scaled[:, [0, 2]] *= sx
        scaled[:, [1, 3]] *= sy
    return scaled


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    x1, y1, x2, y2 = np.rint(box[:4]).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.drawMarker(image, (cx, cy), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    cv2.putText(image, label, (max(2, x1), max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                color, 1, cv2.LINE_AA)


def text_with_shadow(image: np.ndarray, text: str, position: tuple[int, int], scale: float,
                     color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def add_zoom(panel: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    if len(gt) == 0:
        return
    height, width = panel.shape[:2]
    x1, y1, x2, y2 = gt[0]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    crop_w, crop_h = 160, 100
    left = int(np.clip(cx - crop_w // 2, 0, max(0, width - crop_w)))
    top = int(np.clip(cy - crop_h // 2, 0, max(0, height - crop_h)))
    crop = panel[top:top + crop_h, left:left + crop_w]
    if crop.shape[:2] != (crop_h, crop_w):
        return
    zoom = cv2.resize(crop, (320, 200), interpolation=cv2.INTER_NEAREST)
    zx, zy = width - 330, 48
    cv2.rectangle(panel, (zx - 3, zy - 3), (zx + 323, zy + 203), (255, 255, 255), 2)
    panel[zy:zy + 200, zx:zx + 320] = zoom
    cv2.rectangle(panel, (left, top), (left + crop_w, top + crop_h), (255, 255, 255), 1)
    text_with_shadow(panel, "2x TARGET VIEW", (zx + 8, zy + 20), 0.48, (255, 255, 255), 1)


def make_panel(frame: np.ndarray, gt_raw: np.ndarray, pred_raw: np.ndarray, model_label: str,
               pred_color: tuple[int, int, int], panel_size: tuple[int, int], match_iou: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    panel_w, panel_h = panel_size
    source_h, source_w = frame.shape[:2]
    panel = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    gt = scale_boxes(gt_raw, panel_w / source_w, panel_h / source_h)
    pred = pred_raw.copy()
    if len(pred):
        pred[:, [0, 2]] *= panel_w / source_w
        pred[:, [1, 3]] *= panel_h / source_h
    counts = match_counts(gt_raw, pred_raw, match_iou)
    for box in gt:
        draw_box(panel, box, COLORS["gt"], "GT", 2)
    for box in pred:
        draw_box(panel, box, pred_color, f"P {box[4]:.2f}", 2)
    add_zoom(panel, gt, pred)
    tp, fp, fn = counts
    status = "CLEAR" if len(gt) == 0 and fp == 0 else "TP" if tp > 0 and fp == 0 and fn == 0 else f"TP {tp}  FP {fp}  MISS {fn}"
    status_color = COLORS["clear"] if status == "CLEAR" else COLORS["ok"] if fp == 0 and fn == 0 else COLORS["bad"]
    cv2.rectangle(panel, (0, 0), (panel_w, 40), (18, 18, 18), -1)
    text_with_shadow(panel, model_label, (14, 28), 0.72, (245, 245, 245), 2)
    text_with_shadow(panel, status, (panel_w - 260, 28), 0.62, status_color, 2)
    return panel, counts


def source_fps(video_path: Path | None, override: float | None) -> float:
    if override:
        return override
    if video_path and video_path.is_file():
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if fps > 0:
            return fps
    return 50.0


def metrics(stats: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {**stats, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def main() -> None:
    args = parse_args()
    images = read_manifest(args.images)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    print(f"Inferring left model on {len(images)} frames: {args.left_model}", flush=True)
    left_predictions, left_seconds = infer(args.left_model, args.images, args)
    print(f"Inferring right model: {args.right_model}", flush=True)
    right_predictions, right_seconds = infer(args.right_model, args.images, args)

    fps = source_fps(args.source_video, args.fps)
    panel_size = tuple(args.panel_size)
    output_size = (panel_size[0] * 2, panel_size[1])
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.output_video}")

    totals = {"left": {"tp": 0, "fp": 0, "fn": 0}, "right": {"tp": 0, "fp": 0, "fn": 0}}
    absent_fp_frames = {"left": 0, "right": 0}
    started = time.perf_counter()
    for index, image_path in enumerate(images):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read {image_path}")
        gt = load_gt(image_path, frame.shape[1], frame.shape[0])
        left_pred = left_predictions.get(str(image_path), np.empty((0, 5), dtype=np.float32))
        right_pred = right_predictions.get(str(image_path), np.empty((0, 5), dtype=np.float32))
        left_panel, left_counts = make_panel(frame, gt, left_pred, args.left_label, COLORS["left"], panel_size, args.match_iou)
        right_panel, right_counts = make_panel(frame, gt, right_pred, args.right_label, COLORS["right"], panel_size, args.match_iou)
        for side, counts in (("left", left_counts), ("right", right_counts)):
            for key, value in zip(("tp", "fp", "fn"), counts):
                totals[side][key] += value
            if len(gt) == 0 and counts[1] > 0:
                absent_fp_frames[side] += 1
        combined = np.concatenate((left_panel, right_panel), axis=1)
        cv2.line(combined, (panel_size[0], 0), (panel_size[0], panel_size[1]), (255, 255, 255), 2)
        footer = f"Video00004  |  frame {index + 1:04d}/{len(images)}  |  {index / fps:06.2f}s  |  input {args.imgsz[1]}x{args.imgsz[0]}  conf {args.conf:.2f}"
        text_with_shadow(combined, footer, (500, panel_size[1] - 14), 0.52, (245, 245, 245), 1)
        writer.write(combined)
        if (index + 1) % 250 == 0 or index + 1 == len(images):
            print(f"Rendered {index + 1}/{len(images)} frames", flush=True)
    writer.release()
    render_seconds = time.perf_counter() - started

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
        "left": {"label": args.left_label, "model": str(args.left_model.resolve()), "sha256": sha256(args.left_model),
                 "inference_seconds": left_seconds, "metrics": metrics(totals["left"]),
                 "absent_fp_frames": absent_fp_frames["left"]},
        "right": {"label": args.right_label, "model": str(args.right_model.resolve()), "sha256": sha256(args.right_model),
                  "inference_seconds": right_seconds, "metrics": metrics(totals["right"]),
                  "absent_fp_frames": absent_fp_frames["right"]},
        "render_seconds": render_seconds,
        "output_video": str(args.output_video.resolve()),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
