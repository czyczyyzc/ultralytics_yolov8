#!/usr/bin/env python3
"""Run a PT detector on a video and export CSV plus a detector-only visualization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--model-label", default="YOLOv8n neg15")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (245, 248, 250),
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (5, 9, 12), 5, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def panel(image: np.ndarray, y1: int, y2: int) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y1), (image.shape[1], y2), (13, 21, 26), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)


def draw_detection(image: np.ndarray, box: np.ndarray, confidence: float) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box.round().astype(int)
    x1, x2 = np.clip([x1, x2], 0, width - 1)
    y1, y2 = np.clip([y1, y2], 0, height - 1)
    color = (32, 166, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    label = f"DRONE {confidence:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    label_y = max(65, y1 - text_height - baseline - 10)
    label_x = min(x1, max(0, width - text_width - 14))
    cv2.rectangle(
        image,
        (label_x, label_y),
        (label_x + text_width + 14, label_y + text_height + baseline + 10),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (label_x + 7, label_y + text_height + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (7, 12, 16),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.detections.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video: {args.output_video}")

    detection_frames = 0
    detection_count = 0
    frame_count = 0
    start = time.time()
    model = YOLO(str(args.model))
    with args.detections.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(("frame", "x", "y", "width", "height", "confidence", "class_id"))
        results = model.predict(
            source=str(args.video),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            stream=True,
            verbose=False,
        )
        try:
            for frame_index, result in enumerate(results):
                image = result.orig_img.copy()
                boxes = (
                    np.empty((0, 4), dtype=np.float32)
                    if result.boxes is None
                    else result.boxes.xyxy.cpu().numpy()
                )
                confidences = (
                    np.empty((0,), dtype=np.float32)
                    if result.boxes is None
                    else result.boxes.conf.cpu().numpy()
                )
                classes = (
                    np.empty((0,), dtype=np.int64)
                    if result.boxes is None
                    else result.boxes.cls.cpu().numpy().astype(np.int64)
                )
                if len(boxes):
                    detection_frames += 1
                    detection_count += len(boxes)
                for box, confidence, class_id in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = map(float, box)
                    csv_writer.writerow(
                        (
                            frame_index,
                            f"{x1:.6f}",
                            f"{y1:.6f}",
                            f"{x2 - x1:.6f}",
                            f"{y2 - y1:.6f}",
                            f"{float(confidence):.6f}",
                            int(class_id),
                        )
                    )
                    draw_detection(image, box, float(confidence))

                panel(image, 0, 64)
                state = f"{len(boxes)} DETECTION(S)" if len(boxes) else "NO DETECTION"
                state_color = (95, 242, 173) if len(boxes) else (180, 190, 198)
                put_text(image, f"PT REFERENCE | {args.model_label} | conf={args.conf:.2f}", (20, 27), 0.64)
                put_text(image, state, (20, 54), 0.51, state_color)
                time_text = f"FRAME {frame_index + 1}/{expected_frames} | {frame_index / fps:06.2f}s"
                (time_width, _), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
                put_text(image, time_text, (width - time_width - 20, 38), 0.54)
                writer.write(image)
                frame_count += 1
        finally:
            writer.release()

    elapsed = time.time() - start
    summary = {
        "schema_version": "anti_uav.pt_detector_video.v1",
        "runtime": "server_pt_reference",
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "video": str(args.video),
        "video_sha256": sha256_file(args.video),
        "input_height_width": args.imgsz,
        "confidence": args.conf,
        "nms_iou": args.iou,
        "source_fps": fps,
        "source_width": width,
        "source_height": height,
        "total_frames": frame_count,
        "detection_frames": detection_frames,
        "detection_count": detection_count,
        "elapsed_sec": elapsed,
        "pipeline_fps": frame_count / max(elapsed, 1e-12),
        "detections_csv": str(args.detections),
        "visualization": str(args.output_video),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
