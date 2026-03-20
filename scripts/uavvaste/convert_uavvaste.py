#!/usr/bin/env python3
"""Convert UAVVaste COCO annotations and split json to YOLO labels and txt splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path("/mnt/chenziye/datasets/uav_vaste")
DEFAULT_ANN = Path("annotations/annotations.json")
DEFAULT_SPLITS = Path("annotations/train_val_test_distribution_file.json")
DEFAULT_IMAGE_DIR = Path("images")
DEFAULT_LABEL_DIR = Path("labels")
DEFAULT_SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"Dataset root. Default: {DEFAULT_ROOT}")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANN,
        help=f"COCO annotation json relative to --root. Default: {DEFAULT_ANN}",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
        help=f"Split json relative to --root. Default: {DEFAULT_SPLITS}",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels and split txt files.")
    return parser.parse_args()


def coco_box_to_yolo(width: int, height: int, bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox[:4]
    return (
        (x + w / 2) / width,
        (y + h / 2) / height,
        w / width,
        h / height,
    )


def write_label_file(
    label_path: Path,
    image_info: dict,
    annotations: list[dict],
    category_map: dict[int, int],
    overwrite: bool = False,
) -> int:
    if label_path.exists() and not overwrite:
        return -1

    width = int(image_info["width"])
    height = int(image_info["height"])
    lines = []
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue

        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        bw = float(bbox[2])
        bh = float(bbox[3])
        if bw <= 0 or bh <= 0:
            continue

        cls = category_map[int(ann["category_id"])]
        box = coco_box_to_yolo(width, height, bbox)
        lines.append(f"{cls} {' '.join(f'{value:.6f}' for value in box)}")

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def ensure_empty_label(label_path: Path, overwrite: bool = False) -> None:
    if label_path.exists() and not overwrite:
        return
    label_path.write_text("", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    annotations_path = (root / args.annotations).resolve()
    split_path = (root / args.splits).resolve()
    images_dir = (root / DEFAULT_IMAGE_DIR).resolve()
    labels_dir = (root / DEFAULT_LABEL_DIR).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation json not found: {annotations_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split json not found: {split_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(annotations_path.read_text(encoding="utf-8"))
    split_data = json.loads(split_path.read_text(encoding="utf-8"))

    images = coco.get("images", [])
    categories = coco.get("categories", [])
    annotations = coco.get("annotations", [])

    category_ids = sorted(int(category["id"]) for category in categories)
    category_map = {category_id: idx for idx, category_id in enumerate(category_ids)}
    category_names = [category["name"] for category in sorted(categories, key=lambda item: int(item["id"]))]

    images_by_id = {int(image["id"]): image for image in images}
    images_by_name = {image["file_name"]: image for image in images}
    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[int(ann["image_id"])].append(ann)

    written_images = 0
    skipped_images = 0
    object_count = 0
    for image in images:
        label_path = labels_dir / Path(image["file_name"]).with_suffix(".txt").name
        result = write_label_file(
            label_path=label_path,
            image_info=image,
            annotations=annotations_by_image.get(int(image["id"]), []),
            category_map=category_map,
            overwrite=args.overwrite,
        )
        if result < 0:
            skipped_images += 1
            continue
        written_images += 1
        object_count += result

    print(f"Dataset root: {root}")
    print(f"Classes ({len(category_names)}): {', '.join(category_names)}")
    print(f"Labels written: {written_images}, skipped existing: {skipped_images}, objects: {object_count}")

    for split_name in DEFAULT_SPLIT_NAMES:
        filenames = split_data.get(split_name, [])
        split_lines = []
        missing_images = 0
        for filename in filenames:
            image_path = (images_dir / filename).resolve()
            if not image_path.exists():
                print(f"WARNING: image listed in split but missing on disk: {image_path}")
                missing_images += 1
                continue

            image_info = images_by_name.get(filename)
            if image_info is None:
                ensure_empty_label(labels_dir / Path(filename).with_suffix(".txt").name, overwrite=args.overwrite)
            split_lines.append(str(image_path))

        split_file = root / f"{split_name}.txt"
        if split_file.exists() and not args.overwrite:
            print(f"[{split_name}] keeping existing split file: {split_file}")
        else:
            split_file.write_text("\n".join(split_lines) + ("\n" if split_lines else ""), encoding="utf-8")
            print(f"[{split_name}] wrote {len(split_lines)} image paths to {split_file} (missing: {missing_images})")


if __name__ == "__main__":
    main()
