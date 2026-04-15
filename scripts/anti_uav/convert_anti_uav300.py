#!/usr/bin/env python3
"""Convert Anti-UAV300 videos and JSON annotations into a YOLO detection dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


DEFAULT_SOURCE_ROOT = Path("/mnt/chenziye/datasets/anti_uav/Anti-UAV-RGBT")
DEFAULT_OUTPUT_ROOT = Path("/mnt/chenziye/datasets/anti_uav/anti_uav300_yolo")
ALLOWED_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mpg", ".mpeg")
MODALITIES = ("rgb", "ir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help=f"Extracted Anti-UAV300 root. Default: {DEFAULT_SOURCE_ROOT}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help=f"Output YOLO dataset root. Default: {DEFAULT_OUTPUT_ROOT}")
    parser.add_argument("--modalities", nargs="+", choices=list(MODALITIES), default=list(MODALITIES), help="Modalities to export.")
    parser.add_argument("--frame-step", type=int, default=2, help="Keep every Nth frame.")
    parser.add_argument("--negative-frame-step", type=int, default=8, help="Keep every Nth empty frame. Use 0 to drop empty frames.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when no explicit split file is provided.")
    parser.add_argument("--min-box-size", type=float, default=2.0, help="Skip boxes smaller than this size in pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported images, labels, and split txt files.")
    return parser.parse_args()


def discover_sequences(source_root: Path) -> list[dict]:
    """Find Anti-UAV sequence directories with RGB/IR videos and labels."""
    sequences = {}
    for json_path in source_root.rglob("*_label.json"):
        sequence_dir = json_path.parent
        key = str(sequence_dir.relative_to(source_root))
        entry = sequences.setdefault(
            key,
            {
                "name": key.replace("/", "_"),
                "dir": sequence_dir,
                "modalities": {},
            },
        )
        stem = json_path.stem.replace("_label", "")
        modality = stem.lower()
        if modality not in MODALITIES:
            continue
        video_path = find_video_for_modality(sequence_dir, stem)
        if video_path is None:
            continue
        entry["modalities"][modality] = {
            "video": video_path,
            "label": json_path,
        }
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
    scored = sorted(
        ((stable_score(name), name) for name in sequence_names),
        key=lambda item: (item[0], item[1]),
    )
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


def should_keep_frame(frame_index: int, has_box: bool, frame_step: int, negative_frame_step: int) -> bool:
    """Decide whether a frame should be exported."""
    if has_box:
        return frame_index % frame_step == 0
    if negative_frame_step <= 0:
        return False
    return frame_index % negative_frame_step == 0


def box_to_yolo(size: tuple[int, int], box: list[float]) -> tuple[float, float, float, float] | None:
    """Convert an xywh bbox to YOLO normalized cxcywh."""
    width, height = size
    x, y, w, h = [float(value) for value in box[:4]]
    if w <= 0 or h <= 0:
        return None
    return (
        (x + w / 2.0) / width,
        (y + h / 2.0) / height,
        w / width,
        h / height,
    )


def export_sequence(
    sequence: dict,
    modality: str,
    output_root: Path,
    frame_step: int,
    negative_frame_step: int,
    min_box_size: float,
    overwrite: bool,
) -> list[Path]:
    """Export one sequence/modality to images and YOLO labels."""
    meta = sequence["modalities"][modality]
    boxes = read_boxes(meta["label"])
    video_path = meta["video"]
    image_dir = output_root / "images" / modality
    label_dir = output_root / "labels" / modality
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    exported = []
    frame_index = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            bbox = boxes[frame_index] if frame_index < len(boxes) else []
            has_box = bool(bbox and len(bbox) >= 4 and float(bbox[2]) >= min_box_size and float(bbox[3]) >= min_box_size)
            if not should_keep_frame(frame_index, has_box, frame_step, negative_frame_step):
                frame_index += 1
                continue

            image_name = f"{sequence['name']}_{modality}_{frame_index:06d}.jpg"
            label_name = f"{sequence['name']}_{modality}_{frame_index:06d}.txt"
            image_path = image_dir / image_name
            label_path = label_dir / label_name

            if overwrite or not image_path.exists():
                cv2.imwrite(str(image_path), frame)

            if has_box:
                yolo_box = box_to_yolo((frame.shape[1], frame.shape[0]), bbox)
                label_text = f"0 {' '.join(f'{value:.6f}' for value in yolo_box)}\n" if yolo_box else ""
            else:
                label_text = ""

            if overwrite or not label_path.exists():
                label_path.write_text(label_text, encoding="utf-8")
            exported.append(image_path)
            frame_index += 1
    finally:
        cap.release()

    return exported


def write_split(path: Path, items: list[Path], overwrite: bool) -> None:
    """Write a YOLO txt split file."""
    if path.exists() and not overwrite:
        return
    path.write_text("\n".join(str(item) for item in items) + ("\n" if items else ""), encoding="utf-8")


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

    split_paths = {
        "rgb": {"train": [], "val": []},
        "ir": {"train": [], "val": []},
        "full": {"train": [], "val": []},
    }

    for sequence in sequences:
        split_name = "train" if sequence["name"] in train_sequences else "val"
        for modality in args.modalities:
            if modality not in sequence["modalities"]:
                continue
            exported = export_sequence(
                sequence=sequence,
                modality=modality,
                output_root=output_root,
                frame_step=max(1, args.frame_step),
                negative_frame_step=args.negative_frame_step,
                min_box_size=args.min_box_size,
                overwrite=args.overwrite,
            )
            split_paths[modality][split_name].extend(exported)
            split_paths["full"][split_name].extend(exported)

    for modality in ("rgb", "ir", "full"):
        if not split_paths[modality]["train"] and not split_paths[modality]["val"]:
            continue
        write_split(output_root / f"train_{modality}.txt", split_paths[modality]["train"], args.overwrite)
        write_split(output_root / f"val_{modality}.txt", split_paths[modality]["val"], args.overwrite)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "sequence_count": len(sequences),
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "train_rgb_frames": len(split_paths["rgb"]["train"]),
        "val_rgb_frames": len(split_paths["rgb"]["val"]),
        "train_ir_frames": len(split_paths["ir"]["train"]),
        "val_ir_frames": len(split_paths["ir"]["val"]),
        "train_full_frames": len(split_paths["full"]["train"]),
        "val_full_frames": len(split_paths["full"]["val"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
