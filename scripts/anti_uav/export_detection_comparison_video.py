#!/usr/bin/env python3
"""Export side-by-side detector failure slideshow videos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.anti_uav.replay_eval import bbox_iou, parse_imgsz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Image txt manifest or image directory.")
    parser.add_argument("--left-model", type=Path, required=True)
    parser.add_argument("--right-model", type=Path, required=True)
    parser.add_argument("--left-label", default="960x960")
    parser.add_argument("--right-label", default="640x640")
    parser.add_argument("--left-imgsz", default="960")
    parser.add_argument("--right-imgsz", default="640")
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--repeat-frames", type=int, default=6, help="Repeat each failure image this many video frames.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)


def label_path_for_image(image_path: Path) -> Path:
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def load_gt(image_path: Path, shape: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    h, w = shape
    boxes = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((x1, y1, x2, y2))
    return boxes


def predict_boxes(model: YOLO, image_path: Path, imgsz, conf: float, device: str) -> list[tuple[tuple[float, float, float, float], float]]:
    result = model.predict(str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    boxes = []
    if result.boxes is None:
        return boxes
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for box, score in zip(xyxy, confs):
        boxes.append((tuple(float(v) for v in box[:4]), float(score)))
    return boxes


def best_match(gt_boxes: list[Sequence[float]], pred_boxes: list[tuple[Sequence[float], float]]) -> tuple[bool, float, Optional[Sequence[float]], float]:
    best_iou = 0.0
    best_box = None
    best_conf = 0.0
    for gt in gt_boxes:
        for pred, conf in pred_boxes:
            iou = bbox_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_box = pred
                best_conf = conf
    return best_iou > 0, best_iou, best_box, best_conf


def draw_box(frame: np.ndarray, bbox: Optional[Sequence[float]], color: tuple[int, int, int], label: str) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def put_text(frame: np.ndarray, text: str, y: int, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)


def resize_panel(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] == width:
        return frame
    h = int(round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)


def annotate(frame: np.ndarray, label: str, gt: list[Sequence[float]], preds, match_iou: float, best_box, best_conf: float, ok: bool) -> np.ndarray:
    panel = frame.copy()
    for box in gt:
        draw_box(panel, box, (0, 255, 0), "GT")
    color = (255, 180, 0) if ok else (0, 0, 255)
    for box, conf in preds[:5]:
        draw_box(panel, box, color, f"{conf:.2f}")
    put_text(panel, f"{label} {'OK' if ok else 'FAIL'} best_iou={match_iou:.2f} conf={best_conf:.2f}", 24, color)
    return panel


def open_writer(path: Path, frame: np.ndarray, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer: {path}")
    return writer


def main() -> None:
    args = parse_args()
    images = iter_images(args.images)
    if args.limit:
        images = images[: args.limit]
    left = YOLO(str(args.left_model))
    right = YOLO(str(args.right_model))
    left_imgsz = parse_imgsz(args.left_imgsz)
    right_imgsz = parse_imgsz(args.right_imgsz)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    failures = 0
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "gt_count", "left_ok", "left_best_iou", "left_conf", "right_ok", "right_best_iou", "right_conf", "reason"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        for index, image_path in enumerate(images, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            gt = load_gt(image_path, frame.shape[:2])
            left_preds = predict_boxes(left, image_path, left_imgsz, args.conf, args.device)
            right_preds = predict_boxes(right, image_path, right_imgsz, args.conf, args.device)
            _, left_iou, left_box, left_conf = best_match(gt, left_preds)
            _, right_iou, right_box, right_conf = best_match(gt, right_preds)
            left_ok = left_iou >= args.iou_thresh
            right_ok = right_iou >= args.iou_thresh
            if left_ok and right_ok:
                continue
            if left_ok and not right_ok:
                reason = "left_covers_better"
            elif right_ok and not left_ok:
                reason = "right_covers_better"
            else:
                reason = "both_fail"
            failures += 1
            csv_writer.writerow(
                {
                    "image": str(image_path),
                    "gt_count": len(gt),
                    "left_ok": left_ok,
                    "left_best_iou": f"{left_iou:.4f}",
                    "left_conf": f"{left_conf:.4f}",
                    "right_ok": right_ok,
                    "right_best_iou": f"{right_iou:.4f}",
                    "right_conf": f"{right_conf:.4f}",
                    "reason": reason,
                }
            )
            left_panel = annotate(frame, args.left_label, gt, left_preds, left_iou, left_box, left_conf, left_ok)
            right_panel = annotate(frame, args.right_label, gt, right_preds, right_iou, right_box, right_conf, right_ok)
            left_panel = resize_panel(left_panel, args.panel_width)
            right_panel = resize_panel(right_panel, args.panel_width)
            if left_panel.shape[0] != right_panel.shape[0]:
                h = min(left_panel.shape[0], right_panel.shape[0])
                left_panel = cv2.resize(left_panel, (left_panel.shape[1], h), interpolation=cv2.INTER_AREA)
                right_panel = cv2.resize(right_panel, (right_panel.shape[1], h), interpolation=cv2.INTER_AREA)
            combined = np.concatenate([left_panel, right_panel], axis=1)
            if writer is None:
                writer = open_writer(args.output_video, combined, args.fps)
            for _ in range(max(args.repeat_frames, 1)):
                writer.write(combined)
            if index % 100 == 0:
                print(f"processed={index} failures={failures}", flush=True)
    if writer is not None:
        writer.release()
    print(f"processed={len(images)} failures={failures} output={args.output_video}")


if __name__ == "__main__":
    main()
