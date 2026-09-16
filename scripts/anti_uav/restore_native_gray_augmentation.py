#!/usr/bin/env python3
"""Prepare a native-only baseline without modifying archived scale experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def native_schedule(train, val):
    """Remove extra zoom files only; preserve order and all native repetitions."""
    result = [p for p in train if "zoom_train" not in Path(p).parts]
    if not result or set(result) & set(val):
        raise ValueError("Empty training schedule or validation overlap")
    for p in result:
        if {"gray_val", "zoom_val", "Video00004"} & set(Path(p).parts):
            raise ValueError(f"Held-out frame in training: {p}")
    if any("Video00004" in Path(p).parts for p in val):
        raise ValueError("Video00004 must remain test-only")
    return result


def restore(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load((source / "train_hardneg_gray_monitor.yaml").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    train = Path(config["train"]).read_text().splitlines()
    val_text = Path(config["val"]).read_text()
    native = native_schedule(train, val_text.splitlines())
    if len(native) != manifest["append_only_entries"]:
        raise ValueError("Native schedule count differs from the audited source")
    val_hash = manifest["validation_video"]["sha256"][:12]
    if any(val_hash in p or "stationary_video00009" in p for p in native):
        raise ValueError("Validation video leaked into training")
    output.mkdir(parents=True)
    native_text = "\n".join(native) + "\n"
    (output / "train_hardneg.txt").write_text(native_text)
    (output / "val_monitor.txt").write_text(val_text)
    restored = dict(path=str(output), train=str(output / "train_hardneg.txt"),
                    val=str(output / "val_monitor.txt"), names=config["names"])
    (output / "train_hardneg_gray_monitor.yaml").write_text(yaml.safe_dump(restored, sort_keys=False))
    manifest.update(schema="native_augmentation_restored.v1", source_dataset=str(source),
                    final_entries=len(native), final_negative_fraction=manifest["negative"] / len(native),
                    zoom_training_samples=0, online_donors=0, online_views_per_donor=0,
                    online_additional_slots=0, all_native_positive_and_negative_exposure_preserved=True,
                    augmentation_type="Original YOLO transforms only; no extra static or dynamic scale views",
                    training_started=False,
                    native_schedule_sha256=hashlib.sha256(native_text.encode()).hexdigest(),
                    validation_schedule_sha256=hashlib.sha256(val_text.encode()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.source, args.output), indent=2))
