#!/usr/bin/env python3
"""Append a frozen approved-video snapshot, preserving every native baseline slot."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import yaml

from scripts.anti_uav.build_approved_gray_rehearsal import extract_task
from scripts.anti_uav.build_manual_gray_video_rehearsal import sha256_file


def append_samples(base, additions, validation):
    if len(set(additions)) != len(additions) or set(base) & set(additions):
        raise ValueError("New samples must be unique and absent from the baseline")
    result = list(base) + sorted(additions)
    if set(result) & set(validation):
        raise ValueError("Train/validation overlap")
    for path in result:
        if {"Video00004", "gray_val", "zoom_val", "zoom_train"} & set(Path(path).parts):
            raise ValueError(f"Forbidden training source: {path}")
    assert Counter(result[:len(base)]) == Counter(base)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--old-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--positive-stride", type=int, default=3)
    p.add_argument("--negative-stride", type=int, default=20)
    a = p.parse_args()
    if a.output.exists() or min(a.positive_stride, a.negative_stride) < 1:
        raise ValueError("Use a fresh output and positive sample strides")
    cv2.setNumThreads(1)
    config = yaml.safe_load((a.source / "train_hardneg_gray_monitor.yaml").read_text())
    source = json.loads((a.source / "manifest.json").read_text())
    if config.get("online_scale") or source["zoom_training_samples"] or source["online_additional_slots"]:
        raise ValueError("Baseline must use native data only")
    base = Path(config["train"]).read_text().splitlines()
    val_text = Path(config["val"]).read_text()
    old = json.loads((a.old_root / "manifest.json").read_text())
    test_hash = next(r["video_sha256"] for r in old["videos"] if Path(r["video_name"]).stem == "Video00004")
    val_hash = source["validation_video"]["sha256"]
    blocked = set(source["train_video_hashes"]) | {test_hash, val_hash}
    snapshot = json.loads(a.snapshot.read_text())
    rows = snapshot["new_rows"]
    if not rows or len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate snapshot")
    if any(r["sha256"] in blocked for r in rows):
        raise ValueError("Snapshot contains baseline/held-out video")
    append_samples(base, [], val_text.splitlines())
    a.output.mkdir(parents=True)
    (a.output / "snapshot.json").write_bytes(a.snapshot.read_bytes())
    records, additions, positive_count, negative_count = [], [], 0, 0
    for row in rows:
        task = Path(row["task"])
        before = sha256_file(task / "manifest.json")
        live = json.loads((task / "manifest.json").read_text())
        if live["video"]["sha256"] != row["sha256"] or live["video"]["name"] != row["name"]:
            raise ValueError(f"Snapshot source changed: {task}")
        pos, neg, record = extract_task(task, a.video_root, a.output, blocked,
                                        a.positive_stride, a.negative_stride)
        if before != record["manifest_sha256"]:
            raise ValueError(f"Manifest changed during extraction: {task}")
        if not pos:
            raise ValueError(f"No positive sample from new video: {task}")
        additions.extend(map(str, pos + neg))
        positive_count += len(pos)
        negative_count += len(neg)
        records.append(record)
        (a.output / "preparation_progress.json").write_text(json.dumps(records, indent=2) + "\n")
    schedule = append_samples(base, additions, val_text.splitlines())
    train_file = a.output / "train_hardneg.txt"
    train_file.write_text("\n".join(schedule) + "\n")
    (a.output / "val_monitor.txt").write_text(val_text)
    config = dict(path=str(a.output), train=str(train_file),
                  val=str(a.output / "val_monitor.txt"), names=config["names"])
    (a.output / "train_hardneg_gray_monitor.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    negative = source["negative"] + negative_count
    manifest = dict(source, schema="native_approved_expansion.v1", source_dataset=str(a.source),
                    source_train_sha256=sha256_file(Path(yaml.safe_load((a.source / "train_hardneg_gray_monitor.yaml").read_text())["train"])),
                    snapshot_sha256=sha256_file(a.snapshot), appended_videos=records,
                    original_gray_training_videos=len(source["train_video_hashes"]) + len(records),
                    train_video_hashes=sorted(set(source["train_video_hashes"]) | {r["sha256"] for r in records}),
                    append_only_entries=len(schedule), final_entries=len(schedule),
                    append_only_positive=source["append_only_positive"] + positive_count,
                    negative=negative, append_only_negative_fraction=negative / len(schedule),
                    final_negative_fraction=negative / len(schedule), added_positive_samples=positive_count,
                    added_negative_samples=negative_count, positive_stride=a.positive_stride,
                    negative_stride=a.negative_stride, baseline_prefix_exactly_preserved=True,
                    native_schedule_sha256=sha256_file(train_file), training_started=False,
                    test_sha256=test_hash, validation_sha256=val_hash,
                    limitation="Whole-video SHA256 isolation; same-session or reencoded overlap is not ruled out.")
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("original_gray_training_videos", "final_entries",
          "added_positive_samples", "added_negative_samples", "final_negative_fraction")}), flush=True)


if __name__ == "__main__":
    main()
