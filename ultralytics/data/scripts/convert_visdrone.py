#!/usr/bin/env python3
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convert VisDrone detection annotations to YOLO detection labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


DEFAULT_ROOT = Path("/mnt/chenziye/datasets/vis_drone")
DEFAULT_SPLITS = ("VisDrone2019-DET-train", "VisDrone2019-DET-val")
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"VisDrone dataset root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset split directories under --root to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files.",
    )
    return parser.parse_args()


def find_image(images_dir: Path, stem: str) -> Path:
    for suffix in IMG_SUFFIXES:
        image_path = images_dir / f"{stem}{suffix}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Image not found for '{stem}' in '{images_dir}'")


def convert_box(size: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    width, height = size
    x, y, w, h = box
    return (
        (x + w / 2) / width,
        (y + h / 2) / height,
        w / width,
        h / height,
    )


def convert_annotation_file(annotation_path: Path, images_dir: Path, labels_dir: Path, overwrite: bool = False) -> int:
    label_path = labels_dir / annotation_path.name
    if label_path.exists() and not overwrite:
        return -1

    image_path = find_image(images_dir, annotation_path.stem)
    with Image.open(image_path) as image:
        size = image.size

    output_lines = []
    for line_number, raw_line in enumerate(annotation_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw_line.strip():
            continue

        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 6:
            print(f"WARNING: skipping malformed line {annotation_path}:{line_number}: {raw_line}")
            continue

        x, y, w, h = parts[:4]
        score, category = parts[4], parts[5]
        if score == "0":
            continue

        cls = int(category) - 1
        if cls < 0 or cls >= len(CLASS_NAMES):
            continue

        box = convert_box(size, (int(x), int(y), int(w), int(h)))
        output_lines.append(f"{cls} {' '.join(f'{value:.6f}' for value in box)}")

    label_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""))
    return len(output_lines)


def convert_split(split_dir: Path, overwrite: bool = False) -> tuple[int, int, int]:
    annotations_dir = split_dir / "annotations"
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)

    converted_files = 0
    skipped_files = 0
    object_count = 0
    for annotation_path in sorted(annotations_dir.glob("*.txt")):
        result = convert_annotation_file(annotation_path, images_dir, labels_dir, overwrite=overwrite)
        if result < 0:
            skipped_files += 1
            continue
        converted_files += 1
        object_count += result

    return converted_files, skipped_files, object_count


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    print(f"Converting VisDrone dataset under: {root}")
    print(f"Classes: {len(CLASS_NAMES)} -> {', '.join(CLASS_NAMES)}")

    for split_name in args.splits:
        split_dir = root / split_name
        converted_files, skipped_files, object_count = convert_split(split_dir, overwrite=args.overwrite)
        print(
            f"[{split_name}] converted {converted_files} files, skipped {skipped_files} existing files, "
            f"wrote {object_count} objects"
        )


if __name__ == "__main__":
    main()
