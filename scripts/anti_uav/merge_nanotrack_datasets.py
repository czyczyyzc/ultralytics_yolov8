#!/usr/bin/env python3
"""Merge multiple NanoTrack dataset roots into one symlinked crop511 workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", type=Path, nargs="+", required=True, help="One or more modality-specific NanoTrack roots containing crop511/train.json/val.json.")
    parser.add_argument("--output-root", type=Path, required=True, help="Merged modality-specific NanoTrack root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing symlinks and JSON outputs.")
    return parser.parse_args()


def load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def merge_dict(target: dict, payload: dict) -> None:
    for video_name, tracks in payload.items():
        if video_name in target:
            raise ValueError(f"Duplicate NanoTrack video name during merge: {video_name}")
        target[video_name] = tracks


def ensure_link(link_path: Path, source: Path, overwrite: bool) -> None:
    if link_path.exists() or link_path.is_symlink():
        if not overwrite:
            return
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            raise IsADirectoryError(f"Refusing to overwrite non-link directory: {link_path}")
    link_path.symlink_to(source)


def main() -> None:
    args = parse_args()
    input_roots = [path.expanduser().resolve() for path in args.input_roots]
    output_root = args.output_root.expanduser().resolve()
    crop_root = output_root / "crop511"
    crop_root.mkdir(parents=True, exist_ok=True)

    train_meta: dict = {}
    val_meta: dict = {}
    split_manifest = {"train": [], "val": []}

    for root in input_roots:
        if not root.exists():
            raise FileNotFoundError(f"NanoTrack root does not exist: {root}")
        root_crop = root / "crop511"
        if not root_crop.exists():
            raise FileNotFoundError(f"crop511 directory not found in {root}")
        merge_dict(train_meta, load_meta(root / "train.json"))
        merge_dict(val_meta, load_meta(root / "val.json"))
        manifest_path = root / "split_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            split_manifest["train"].extend(manifest.get("train", []))
            split_manifest["val"].extend(manifest.get("val", []))
        for sequence_dir in sorted(path for path in root_crop.iterdir() if path.is_dir()):
            ensure_link(crop_root / sequence_dir.name, sequence_dir, overwrite=args.overwrite)

    (output_root / "train.json").write_text(json.dumps(train_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "val.json").write_text(json.dumps(val_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "output_root": str(output_root),
        "input_roots": [str(path) for path in input_roots],
        "train_sequences": len(train_meta),
        "val_sequences": len(val_meta),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
