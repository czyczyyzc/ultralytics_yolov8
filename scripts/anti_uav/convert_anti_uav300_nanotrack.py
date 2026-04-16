#!/usr/bin/env python3
"""Convert Anti-UAV300 into a NanoTrack-style crop511 dataset and JSON annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SOURCE_ROOT = Path("/mnt/chenziye/datasets/anti_uav/Anti-UAV300")
DEFAULT_OUTPUT_ROOT = Path("/mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack")
ALLOWED_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".mkv")
MODALITIES = ("rgb", "ir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help=f"Extracted Anti-UAV300 root. Default: {DEFAULT_SOURCE_ROOT}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help=f"Output NanoTrack dataset root. Default: {DEFAULT_OUTPUT_ROOT}")
    parser.add_argument("--modalities", nargs="+", choices=list(MODALITIES), default=list(MODALITIES), help="Modalities to export.")
    parser.add_argument("--frame-step", type=int, default=1, help="Keep every Nth positive frame.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when no explicit split file is provided.")
    parser.add_argument("--crop-size", type=int, default=511, help="Square crop size expected by NanoTrack preprocessing.")
    parser.add_argument("--exemplar-size", type=int, default=127, help="NanoTrack exemplar size used to derive crop scale.")
    parser.add_argument("--context-amount", type=float, default=0.5, help="Context padding around the target during crop export.")
    parser.add_argument("--min-box-size", type=float, default=2.0, help="Skip boxes smaller than this size in pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crops and JSON files.")
    return parser.parse_args()


def discover_sequences(source_root: Path) -> list[dict]:
    """Find Anti-UAV sequence directories with RGB/IR videos and labels."""
    sequences = {}
    for json_path in source_root.rglob("*_label.json"):
        sequence_dir = json_path.parent
        key = str(sequence_dir.relative_to(source_root))
        entry = sequences.setdefault(key, {"name": key.replace("/", "_"), "dir": sequence_dir, "modalities": {}})
        stem = json_path.stem.replace("_label", "")
        modality = stem.lower()
        if modality not in MODALITIES:
            continue
        video_path = find_video_for_modality(sequence_dir, stem)
        if video_path is None:
            continue
        entry["modalities"][modality] = {"video": video_path, "label": json_path}
    return [item for _, item in sorted(sequences.items()) if item["modalities"]]


def find_video_for_modality(sequence_dir: Path, stem: str) -> Path | None:
    """Find the matching video file for a modality."""
    for suffix in ALLOWED_VIDEO_SUFFIXES:
        candidate = sequence_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def split_sequences(sequence_names: list[str], val_ratio: float) -> tuple[set[str], set[str]]:
    """Deterministically split sequence names into train and val groups."""
    scored = sorted(((stable_score(name), name) for name in sequence_names), key=lambda item: (item[0], item[1]))
    val_count = max(1, int(round(len(scored) * val_ratio))) if len(scored) > 1 else 0
    val_names = {name for _, name in scored[:val_count]}
    train_names = {name for _, name in scored[val_count:]}
    if not train_names and val_names:
        moved = sorted(val_names)[-1]
        val_names.remove(moved)
        train_names.add(moved)
    return train_names, val_names


def stable_score(text: str) -> int:
    """Stable hash for deterministic sequence splits."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def read_boxes(label_path: Path) -> list[list[float]]:
    """Load gt_rect boxes from an Anti-UAV label JSON file."""
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "gt_rect" in payload:
        return payload["gt_rect"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported annotation schema in {label_path}")


def crop_like_nanotrack(
    frame: np.ndarray,
    bbox_xywh: list[float],
    *,
    crop_size: int,
    exemplar_size: int,
    context_amount: float,
) -> np.ndarray:
    """Export a centered search-style crop compatible with NanoTrack crop511 datasets."""
    x, y, w, h = [float(v) for v in bbox_xywh[:4]]
    cx = x + (w - 1.0) / 2.0
    cy = y + (h - 1.0) / 2.0
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = max(np.sqrt(max(wc_z * hc_z, 1.0)), 1.0)
    s_x = s_z * (crop_size / float(exemplar_size))
    return get_subwindow_numpy(frame, (cx, cy), crop_size, int(round(s_x)))


def get_subwindow_numpy(frame: np.ndarray, center_xy: tuple[float, float], crop_size: int, original_size: int) -> np.ndarray:
    """Numpy/OpenCV version of the Siamese tracker crop helper."""
    if crop_size <= 0 or original_size <= 0:
        raise ValueError("crop_size and original_size must be positive")

    cx, cy = center_xy
    avg_chans = np.mean(frame, axis=(0, 1))
    im_h, im_w = frame.shape[:2]
    c = (original_size + 1) / 2.0
    xmin = np.floor(cx - c + 0.5)
    xmax = xmin + original_size - 1
    ymin = np.floor(cy - c + 0.5)
    ymax = ymin + original_size - 1

    left_pad = int(max(0.0, -xmin))
    top_pad = int(max(0.0, -ymin))
    right_pad = int(max(0.0, xmax - im_w + 1))
    bottom_pad = int(max(0.0, ymax - im_h + 1))

    xmin += left_pad
    xmax += left_pad
    ymin += top_pad
    ymax += top_pad

    padded = frame
    if any((top_pad, bottom_pad, left_pad, right_pad)):
        padded = np.zeros((im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, frame.shape[2]), dtype=np.uint8)
        padded[top_pad:top_pad + im_h, left_pad:left_pad + im_w, :] = frame
        if top_pad:
            padded[:top_pad, left_pad:left_pad + im_w, :] = avg_chans
        if bottom_pad:
            padded[im_h + top_pad:, left_pad:left_pad + im_w, :] = avg_chans
        if left_pad:
            padded[:, :left_pad, :] = avg_chans
        if right_pad:
            padded[:, im_w + left_pad:, :] = avg_chans

    crop = padded[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size))
    return crop


def export_sequence(
    sequence: dict,
    modality: str,
    output_root: Path,
    frame_step: int,
    crop_size: int,
    exemplar_size: int,
    context_amount: float,
    min_box_size: float,
    overwrite: bool,
) -> dict:
    """Export one sequence/modality to NanoTrack crops plus JSON metadata."""
    meta = sequence["modalities"][modality]
    boxes = read_boxes(meta["label"])
    video_path = meta["video"]
    crop_root = output_root / modality / "crop511"
    sequence_dir = crop_root / sequence["name"]
    sequence_dir.mkdir(parents=True, exist_ok=True)

    track_key = "00"
    track_meta = {}
    exported_frames = 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_index = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            bbox = boxes[frame_index] if frame_index < len(boxes) else []
            if not bbox or len(bbox) < 4:
                frame_index += 1
                continue

            x, y, w, h = [float(value) for value in bbox[:4]]
            if frame_index % frame_step != 0 or w < min_box_size or h < min_box_size:
                frame_index += 1
                continue

            crop = crop_like_nanotrack(
                frame,
                [x, y, w, h],
                crop_size=crop_size,
                exemplar_size=exemplar_size,
                context_amount=context_amount,
            )
            frame_key = f"{frame_index:06d}"
            crop_path = sequence_dir / f"{frame_key}.{track_key}.x.jpg"
            if overwrite or not crop_path.exists():
                cv2.imwrite(str(crop_path), crop)
            track_meta[frame_key] = [0.0, 0.0, float(w), float(h)]
            exported_frames += 1
            frame_index += 1
    finally:
        cap.release()

    return {"meta": {sequence["name"]: {track_key: track_meta}} if track_meta else {}, "frames": exported_frames}


def merge_meta(target: dict, payload: dict) -> None:
    """Merge nested video->track->frame NanoTrack metadata."""
    for video_name, tracks in payload.items():
        video_entry = target.setdefault(video_name, {})
        for track_name, frames in tracks.items():
            video_entry.setdefault(track_name, {}).update(frames)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    sequences = discover_sequences(source_root)
    if not sequences:
        raise RuntimeError(f"No Anti-UAV sequences found under {source_root}")

    train_sequences, val_sequences = split_sequences([sequence["name"] for sequence in sequences], args.val_ratio)
    summary = {"source_root": str(source_root), "output_root": str(output_root), "sequence_count": len(sequences), "modalities": {}}

    for modality in args.modalities:
        train_meta = {}
        val_meta = {}
        train_frames = 0
        val_frames = 0
        for sequence in sequences:
            if modality not in sequence["modalities"]:
                continue
            result = export_sequence(
                sequence=sequence,
                modality=modality,
                output_root=output_root,
                frame_step=max(1, args.frame_step),
                crop_size=args.crop_size,
                exemplar_size=args.exemplar_size,
                context_amount=args.context_amount,
                min_box_size=args.min_box_size,
                overwrite=args.overwrite,
            )
            if sequence["name"] in train_sequences:
                merge_meta(train_meta, result["meta"])
                train_frames += result["frames"]
            else:
                merge_meta(val_meta, result["meta"])
                val_frames += result["frames"]

        modality_root = output_root / modality
        modality_root.mkdir(parents=True, exist_ok=True)
        train_path = modality_root / "train.json"
        val_path = modality_root / "val.json"
        if args.overwrite or not train_path.exists():
            train_path.write_text(json.dumps(train_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.overwrite or not val_path.exists():
            val_path.write_text(json.dumps(val_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        summary["modalities"][modality] = {
            "crop_root": str((modality_root / "crop511").resolve()),
            "train_json": str(train_path.resolve()),
            "val_json": str(val_path.resolve()),
            "train_sequences": sum(1 for name in train_sequences if name in train_meta),
            "val_sequences": sum(1 for name in val_sequences if name in val_meta),
            "train_frames": train_frames,
            "val_frames": val_frames,
        }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
