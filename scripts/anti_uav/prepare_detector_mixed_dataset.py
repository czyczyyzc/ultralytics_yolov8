#!/usr/bin/env python3
"""Prepare merged YOLO train/val manifests from Anti-UAV300 and extra YOLO datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antiuav-root", type=Path, required=True, help="Converted Anti-UAV YOLO root containing train_rgb.txt/val_rgb.txt.")
    parser.add_argument(
        "--extra-yolo-root",
        type=Path,
        required=True,
        action="append",
        nargs="+",
        help=(
            "Extra YOLO dataset root(s). Supports images/{train,val}+labels/{train,val} "
            "or flat images/labels with train_*.jpg and val_*.jpg names. Can be repeated."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for merged txt manifests and dataset yaml.")
    parser.add_argument("--modality", choices=("rgb",), default="rgb", help="Merged modality. Only rgb is supported for the extra dataset.")
    parser.add_argument("--class-name", default="drone", help="Single-class name written into the merged dataset yaml.")
    return parser.parse_args()


def read_manifest(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [Path(line.strip()).expanduser().resolve() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flatten_extra_roots(raw_roots: list[list[Path]]) -> list[Path]:
    roots: list[Path] = []
    for group in raw_roots:
        roots.extend(group)
    return roots


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_file())


def collect_extra_split(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.exists():
        image_dir = root / "images"
        label_dir = root / "labels"
        prefix = f"{split}_"
        image_paths = [path for path in image_files(image_dir) if path.stem.startswith(prefix)]
    else:
        image_paths = image_files(image_dir)
    selected: list[Path] = []
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            selected.append(image_path.resolve())
    return selected


def write_manifest(path: Path, items: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(item) for item in items]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yaml(path: Path, dataset_root: Path, modality: str, class_name: str) -> None:
    payload = (
        "# Ultralytics YOLO mixed Anti-UAV300 + Hanlue dataset\n"
        f"path: {dataset_root}\n"
        f"train: train_{modality}.txt\n"
        f"val: val_{modality}.txt\n\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    args = parse_args()
    anti_root = args.antiuav_root.expanduser().resolve()
    extra_roots = [root.expanduser().resolve() for root in flatten_extra_roots(args.extra_yolo_root)]
    output_root = args.output_root.expanduser().resolve()

    if not anti_root.exists():
        raise FileNotFoundError(f"Anti-UAV YOLO root does not exist: {anti_root}")
    for extra_root in extra_roots:
        if not extra_root.exists():
            raise FileNotFoundError(f"Extra YOLO root does not exist: {extra_root}")

    train_items = read_manifest(anti_root / f"train_{args.modality}.txt")
    val_items = read_manifest(anti_root / f"val_{args.modality}.txt")
    extra_counts = []
    for extra_root in extra_roots:
        extra_train = collect_extra_split(extra_root, "train")
        extra_val = collect_extra_split(extra_root, "val")
        train_items.extend(extra_train)
        val_items.extend(extra_val)
        extra_counts.append({"root": str(extra_root), "train_items": len(extra_train), "val_items": len(extra_val)})

    train_items = sorted(dict.fromkeys(train_items))
    val_items = sorted(dict.fromkeys(val_items))

    write_manifest(output_root / f"train_{args.modality}.txt", train_items)
    write_manifest(output_root / f"val_{args.modality}.txt", val_items)
    yaml_path = output_root / f"AntiUAV300PlusHanlue{args.modality.upper()}.yaml"
    write_yaml(yaml_path, output_root, args.modality, args.class_name)

    summary = {
        "antiuav_root": str(anti_root),
        "extra_yolo_root": str(extra_roots[0]) if len(extra_roots) == 1 else None,
        "extra_yolo_roots": [str(root) for root in extra_roots],
        "extra_counts": extra_counts,
        "output_root": str(output_root),
        "yaml": str(yaml_path),
        "train_items": len(train_items),
        "val_items": len(val_items),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
