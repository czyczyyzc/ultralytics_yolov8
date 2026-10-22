#!/usr/bin/env python3
"""Attach audited large-target online crops to an existing native gray dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.anti_uav.online_random_scale import eligible_donors  # noqa: E402
from ultralytics.data.utils import img2label_paths  # noqa: E402


def read_yolo_row(image_path: str) -> dict:
    """Read one missing cache record without rescanning the full inherited dataset."""
    label_path = Path(img2label_paths([image_path])[0])
    rows = []
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            values = [float(value) for value in line.split()]
            if len(values) != 5 or int(values[0]) != 0:
                raise ValueError(f"Unexpected detection label: {label_path}")
            rows.append(values)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    return dict(
        im_file=image_path,
        shape=image.shape[:2],
        cls=np.asarray([[row[0]] for row in rows], dtype=np.float32),
        bboxes=np.asarray([row[1:] for row in rows], dtype=np.float32).reshape(-1, 4),
        segments=[],
        keypoints=None,
        normalized=True,
        bbox_format="xywh",
    )


def donor_row(row: dict) -> dict:
    return {
        "im_file": row["im_file"],
        "shape": list(row["shape"]),
        "bboxes": np.asarray(row["bboxes"]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--inherited-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-augmented-positive-fraction", type=float, default=.25)
    parser.add_argument("--max-views-per-donor", type=int, default=64)
    parser.add_argument("--partial-probability", type=float, default=.20)
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.inherited_cache = args.inherited_cache.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.target_augmented_positive_fraction < .5:
        raise ValueError("Keep augmented positives below half of positive exposure")
    if args.max_views_per_donor < 1 or not 0 <= args.partial_probability <= .5:
        raise ValueError("Invalid view or partial-target setting")

    config = yaml.safe_load(args.data.read_text())
    if not config.get("online_replacement") or not config.get("label_sampling"):
        raise ValueError("Expected online replacement and an audited negative pool")
    if config.get("online_scale"):
        raise ValueError("Dataset already has online scale augmentation")
    train_paths = Path(config["train"]).read_text().splitlines()
    validation = set(Path(config["val"]).read_text().splitlines())
    if set(train_paths) & validation:
        raise ValueError("Train/validation overlap")

    cache = np.load(str(args.inherited_cache), allow_pickle=True).item()
    inherited = {row["im_file"]: row for row in cache["labels"]}
    unique_paths = sorted(set(train_paths))
    records = []
    missing = []
    for index, path in enumerate(unique_paths):
        row = inherited.get(path)
        if row is None:
            row = read_yolo_row(path)
            missing.append(path)
        records.append(row)
        if index and index % 1000 == 0:
            print(json.dumps({"scanned": index, "unique": len(unique_paths), "cache_misses": len(missing)}), flush=True)
    donors = eligible_donors(records)
    if not donors:
        raise ValueError("No safe large-target donors")

    dataset_manifest = json.loads((args.data.parent / "manifest.json").read_text())
    native_positive = int(dataset_manifest["epoch_positive"])
    desired_slots = round(native_positive * args.target_augmented_positive_fraction /
                          (1 - args.target_augmented_positive_fraction))
    views = min(args.max_views_per_donor, max(1, math.ceil(desired_slots / len(donors))))
    additional_slots = len(donors) * views
    realized_fraction = additional_slots / (native_positive + additional_slots)

    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_bytes((args.data.parent / "manifest.json").read_bytes())
    donor_manifest = args.output / "online_scale_donors.json"
    donor_manifest.write_text(json.dumps({"donors": [donor_row(row) for row in donors]}, indent=2) + "\n")
    result = dict(config)
    result["online_scale"] = dict(
        donor_manifest=str(donor_manifest),
        views_per_donor=views,
        max_upscale=4.0,
        partial_probability=args.partial_probability,
        target_long_edge_bins=[[.15, .30, 1.], [.30, .60, 3.], [.60, .80, 1.]],
        motion_blur_probability=.12,
        local_glare_probability=.12,
    )
    result["label_sampling"] = dict(result["label_sampling"])
    result["label_sampling"]["anchor_slots"] += additional_slots
    output_yaml = args.output / "train_50videos_multiscale.yaml"
    output_yaml.write_text(yaml.safe_dump(result, sort_keys=False))
    audit = dict(
        schema="online_large_target_scale.v2",
        source_data=str(args.data),
        output_data=str(output_yaml),
        inherited_cache=str(args.inherited_cache),
        unique_training_images=len(unique_paths),
        inherited_cache_records=len(inherited),
        newly_parsed_images=len(missing),
        eligible_donors=len(donors),
        native_positive_slots=native_positive,
        views_per_donor=views,
        online_additional_slots=additional_slots,
        target_augmented_positive_fraction=args.target_augmented_positive_fraction,
        realized_augmented_positive_fraction=realized_fraction,
        all_native_slots_preserved=True,
        original_boxes_filtered=False,
        target_long_edge_fraction=[.15, .80],
        emphasized_long_edge_fraction=[.30, .60],
        partial_probability=args.partial_probability,
    )
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
