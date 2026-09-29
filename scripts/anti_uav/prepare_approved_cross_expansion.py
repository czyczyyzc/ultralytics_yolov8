#!/usr/bin/env python3
"""Append one reviewed cross-target video to the 40-video native schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.build_manual_gray_video_rehearsal import sha256_file


def verify_sources(native: dict, online: dict, task: dict, cross_sha256: str) -> None:
    training_hashes = set(native["train_video_hashes"])
    if cross_sha256 not in training_hashes:
        raise ValueError("The existing _x video must remain in the baseline, not be duplicated")
    new_hash = task["video"]["sha256"]
    if new_hash in training_hashes or new_hash in {
        native["test_sha256"], native["validation_sha256"]
    }:
        raise ValueError("New video overlaps a training, validation or test video")
    if training_hashes != set(online["train_video_hashes"]):
        raise ValueError("Online view is not based on the same 40 training videos")
    if online["online_replacement"]["replacement_probability"] != 0.5:
        raise ValueError("Expected the existing 50% online replacement policy")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-native", type=Path, required=True)
    p.add_argument("--source-online", type=Path, required=True)
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--old-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    for key in ("source_native", "source_online", "task", "video_root", "old_root", "output"):
        setattr(a, key, getattr(a, key).resolve())
    snapshot_path = a.output.with_suffix(".snapshot.json")
    if a.output.exists() or snapshot_path.exists():
        raise FileExistsError("Use a fresh output directory and snapshot path")

    native = json.loads((a.source_native / "manifest.json").read_text())
    online = json.loads((a.source_online / "manifest.json").read_text())
    task = json.loads((a.task / "manifest.json").read_text())
    cross_sha256 = "13797ecf054926747a668ed52847f870a07d16a2f929f2bf5b1b317e6f1b3aed"
    verify_sources(native, online, task, cross_sha256)
    row = dict(task=str(a.task), name=task["video"]["name"],
               sha256=task["video"]["sha256"],
               manifest_sha256=sha256_file(a.task / "manifest.json"))
    snapshot_path.write_text(json.dumps(dict(new_rows=[row]), indent=2) + "\n")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/anti_uav/append_approved_gray_native.py"),
        "--source", str(a.source_native), "--snapshot", str(snapshot_path),
        "--video-root", str(a.video_root), "--old-root", str(a.old_root),
        "--output", str(a.output), "--positive-stride", "1", "--negative-stride", "1",
        "--negative-fraction", "0.15",
    ], check=True)

    config = yaml.safe_load((a.output / "train_hardneg_gray_monitor.yaml").read_text())
    old_config = yaml.safe_load((a.source_online / "train_online_gray_monitor.yaml").read_text())
    config["online_replacement"] = old_config["online_replacement"]
    (a.output / "train_online_gray_monitor.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False))
    result = json.loads((a.output / "manifest.json").read_text())
    result.update(cross_expansion=dict(
        already_present_cross_video_sha256=cross_sha256,
        newly_added_video_sha256=row["sha256"],
        baseline_online_cache=config["online_replacement"]["cache"],
        new_video_online_replacement=False,
        note="All reviewed new frames are added once. Existing 40-video cache remains active; "
             "new video uses its original pixels. Video00009 selects weights; Video00004 remains test-only.",
    ))
    (a.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "training_videos": len(result["train_video_hashes"]),
        "added_positive_samples": result["added_positive_samples"],
        "added_negative_samples": result["added_negative_samples"],
        "epoch_entries": result["final_entries"],
        "online_data": str(a.output / "train_online_gray_monitor.yaml"),
    }), flush=True)


if __name__ == "__main__":
    main()
