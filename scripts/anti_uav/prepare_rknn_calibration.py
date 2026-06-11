#!/usr/bin/env python3
"""Sample frames from a video into an RKNN calibration image set and dataset.txt manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source video path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for sampled calibration images.")
    parser.add_argument("--dataset-txt", type=Path, required=True, help="Output dataset manifest consumed by RKNN build().")
    parser.add_argument("--max-frames", type=int, default=64, help="Maximum number of calibration frames to save.")
    parser.add_argument("--frame-step", type=int, default=30, help="Save one frame every N decoded frames.")
    parser.add_argument("--width", type=int, default=960, help="Optional resize width, 0 keeps original.")
    parser.add_argument("--height", type=int, default=960, help="Optional resize height, 0 keeps original.")
    parser.add_argument("--prefix", default="calib", help="Image filename prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    dataset_txt = args.dataset_txt.expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_txt.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {source}")

    saved = []
    frame_index = 0
    sample_index = 0
    try:
        while capture.isOpened() and sample_index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % args.frame_step != 0:
                continue
            if args.width > 0 and args.height > 0:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            image_path = output_dir / f"{args.prefix}_{sample_index + 1:04d}.jpg"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Failed to write calibration frame: {image_path}")
            saved.append(str(image_path))
            sample_index += 1
    finally:
        capture.release()

    if not saved:
        raise RuntimeError("No calibration frames were saved")

    dataset_txt.write_text("\n".join(saved) + "\n", encoding="utf-8")
    print(f"Saved {len(saved)} calibration frames to: {output_dir}")
    print(f"Wrote dataset manifest to: {dataset_txt}")


if __name__ == "__main__":
    main()
