#!/usr/bin/env python3
"""Export the highest-confidence YOLO detection for every video frame to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", default="640", help="Input size as N or H,W.")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def parse_imgsz(value: str) -> int | tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) == 1 and parts[0] > 0:
        return parts[0]
    if len(parts) == 2 and all(part > 0 for part in parts):
        return parts[0], parts[1]
    raise ValueError(f"Expected positive N or H,W image size, got: {value}")


def main() -> None:
    args = parse_args()
    imgsz = parse_imgsz(args.imgsz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model.expanduser().resolve()))
    results = model.predict(
        source=str(args.source.expanduser().resolve()),
        imgsz=imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        verbose=False,
    )

    detected_frames = 0
    total_frames = 0
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("frame", "x", "y", "width", "height", "confidence", "class_id"))
        for frame_index, result in enumerate(results):
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
            total_frames += 1
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            best_index = int(boxes.conf.argmax().item())
            x1, y1, x2, y2 = boxes.xyxy[best_index].detach().cpu().tolist()
            confidence = float(boxes.conf[best_index].item())
            class_id = int(boxes.cls[best_index].item())
            writer.writerow(
                (
                    frame_index,
                    f"{x1:.6f}",
                    f"{y1:.6f}",
                    f"{x2 - x1:.6f}",
                    f"{y2 - y1:.6f}",
                    f"{confidence:.6f}",
                    class_id,
                )
            )
            detected_frames += 1

    print(f"frames={total_frames} detected={detected_frames} output={args.output}")


if __name__ == "__main__":
    main()
