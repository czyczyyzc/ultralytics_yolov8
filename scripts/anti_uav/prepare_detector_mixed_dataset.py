#!/usr/bin/env python3
"""Prepare merged YOLO train/val manifests from Anti-UAV300 and an extra YOLO dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antiuav-root", type=Path, required=True, help="Converted Anti-UAV YOLO root containing train_rgb.txt/val_rgb.txt.")
    parser.add_argument("--extra-yolo-root", type=Path, required=True, help="Extra YOLO dataset root containing images/{train,val} and labels/{train,val}.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for merged txt manifests and dataset yaml.")
    parser.add_argument("--modality", choices=("rgb",), default="rgb", help="Merged modality. Only rgb is supported for the extra dataset.")
    parser.add_argument("--class-name", default="drone", help="Single-class name written into the merged dataset yaml.")
    return parser.parse_args()


def read_manifest(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [Path(line.strip()).expanduser().resolve() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_extra_split(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.exists():
        return []
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
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
    extra_root = args.extra_yolo_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not anti_root.exists():
        raise FileNotFoundError(f"Anti-UAV YOLO root does not exist: {anti_root}")
    if not extra_root.exists():
        raise FileNotFoundError(f"Extra YOLO root does not exist: {extra_root}")

    train_items = read_manifest(anti_root / f"train_{args.modality}.txt")
    val_items = read_manifest(anti_root / f"val_{args.modality}.txt")
    train_items.extend(collect_extra_split(extra_root, "train"))
    val_items.extend(collect_extra_split(extra_root, "val"))

    train_items = sorted(dict.fromkeys(train_items))
    val_items = sorted(dict.fromkeys(val_items))

    write_manifest(output_root / f"train_{args.modality}.txt", train_items)
    write_manifest(output_root / f"val_{args.modality}.txt", val_items)
    yaml_path = output_root / f"AntiUAV300PlusHanlue{args.modality.upper()}.yaml"
    write_yaml(yaml_path, output_root, args.modality, args.class_name)

    summary = {
        "antiuav_root": str(anti_root),
        "extra_yolo_root": str(extra_root),
        "output_root": str(output_root),
        "yaml": str(yaml_path),
        "train_items": len(train_items),
        "val_items": len(val_items),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
