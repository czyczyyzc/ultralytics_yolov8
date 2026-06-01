#!/usr/bin/env python3
"""Convert tracker_sequences style image lists into a NanoTrack crop511 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.convert_anti_uav300_nanotrack import crop_like_nanotrack


DEFAULT_SOURCE_ROOT = Path("/mnt/hanlue/hanlue_multirotor_assets_phase1_20260525/tracker_sequences")
DEFAULT_IMAGE_ROOT = Path("/mnt/hanlue/hanlue_multirotor_assets_phase1_20260525/images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="tracker_sequences root containing train/val sequence folders.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Fallback image root used when frames.txt absolute paths are not valid.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output NanoTrack dataset root.")
    parser.add_argument("--crop-size", type=int, default=511)
    parser.add_argument("--exemplar-size", type=int, default=127)
    parser.add_argument("--context-amount", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def remap_frame_path(raw_path: str, image_root: Path, split: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    fallback = image_root / split / candidate.name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Frame path not found: {raw_path} (fallback: {fallback})")


def load_groundtruth(path: Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        values = [float(part) for part in raw.replace(",", " ").split()]
        if len(values) < 4:
            raise ValueError(f"Invalid groundtruth row in {path}: {raw}")
        boxes.append(values[:4])
    return boxes


def convert_split(split_dir: Path, split: str, image_root: Path, output_root: Path, crop_size: int, exemplar_size: int, context_amount: float, overwrite: bool) -> tuple[dict, int]:
    crop_root = output_root / "rgb" / "crop511"
    crop_root.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict[str, dict[str, list[float]]]] = {}
    positive_frames = 0
    for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        frames_path = sequence_dir / "frames.txt"
        groundtruth_path = sequence_dir / "groundtruth.txt"
        if not frames_path.exists() or not groundtruth_path.exists():
            continue
        frame_paths = [remap_frame_path(line.strip(), image_root, split) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        boxes = load_groundtruth(groundtruth_path)
        if not frame_paths or not boxes:
            continue
        if len(frame_paths) != len(boxes):
            raise ValueError(f"Frame/groundtruth count mismatch in {sequence_dir}: {len(frame_paths)} vs {len(boxes)}")

        video_name = sequence_dir.name
        target_dir = crop_root / video_name
        target_dir.mkdir(parents=True, exist_ok=True)
        track_key = "00"
        track_frames: dict[str, list[float]] = {}
        for frame_index, (frame_path, bbox) in enumerate(zip(frame_paths, boxes)):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Unable to read frame: {frame_path}")
            x, y, w, h = bbox
            crop = crop_like_nanotrack(
                frame,
                [x, y, w, h],
                crop_size=crop_size,
                exemplar_size=exemplar_size,
                context_amount=context_amount,
            )
            frame_key = f"{frame_index:06d}"
            crop_path = target_dir / f"{frame_key}.{track_key}.x.jpg"
            if overwrite or not crop_path.exists():
                cv2.imwrite(str(crop_path), crop)
            track_frames[frame_key] = [0.0, 0.0, float(w), float(h)]
            positive_frames += 1
        meta[video_name] = {track_key: track_frames}
    return meta, positive_frames


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"tracker_sequences root does not exist: {source_root}")
    if not image_root.exists():
        raise FileNotFoundError(f"image root does not exist: {image_root}")

    summary = {"source_root": str(source_root), "image_root": str(image_root), "output_root": str(output_root), "modalities": {"rgb": {}}}
    modality_root = output_root / "rgb"
    modality_root.mkdir(parents=True, exist_ok=True)
    train_meta, train_frames = convert_split(source_root / "train", "train", image_root, output_root, args.crop_size, args.exemplar_size, args.context_amount, args.overwrite)
    val_meta, val_frames = convert_split(source_root / "val", "val", image_root, output_root, args.crop_size, args.exemplar_size, args.context_amount, args.overwrite)

    (modality_root / "train.json").write_text(json.dumps(train_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (modality_root / "val.json").write_text(json.dumps(val_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["modalities"]["rgb"] = {
        "crop_root": str((modality_root / "crop511").resolve()),
        "train_json": str((modality_root / "train.json").resolve()),
        "val_json": str((modality_root / "val.json").resolve()),
        "train_sequences": len(train_meta),
        "val_sequences": len(val_meta),
        "train_positive_frames": train_frames,
        "val_positive_frames": val_frames,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
