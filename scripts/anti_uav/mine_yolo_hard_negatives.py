#!/usr/bin/env python3
"""Mine empty-label images that a YOLO detector mistakes for targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("H", "W"))
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = [Path(line.strip()) for line in args.images.read_text().splitlines() if line.strip()]
    model = YOLO(str(args.model))
    predictions = model.predict(
        source=str(args.images),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        stream=True,
        verbose=False,
    )

    mined: list[tuple[float, int, Path]] = []
    seen = 0
    for result in predictions:
        seen += 1
        if result.boxes is None or len(result.boxes) == 0:
            continue
        confidences = result.boxes.conf.cpu().tolist()
        mined.append((max(confidences), len(confidences), Path(result.path)))
    if seen != len(images):
        raise RuntimeError(f"Predicted {seen} images, expected {len(images)}")

    mined.sort(key=lambda item: (-item[0], str(item[2])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{path}\n" for _, _, path in mined))
    summary = {
        "model": str(args.model),
        "images": str(args.images),
        "input_height_width": args.imgsz,
        "confidence": args.conf,
        "iou": args.iou,
        "scanned_images": seen,
        "hard_negative_images": len(mined),
        "hard_negative_rate": len(mined) / max(seen, 1),
        "maximum_confidence": mined[0][0] if mined else 0.0,
        "total_detections": sum(count for _, count, _ in mined),
        "output": str(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
