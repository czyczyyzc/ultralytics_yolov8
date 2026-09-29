#!/usr/bin/env python3
"""Attach a completed online cutout cache to an append-only approved schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--baseline-online", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads((a.dataset / "manifest.json").read_text())
    base = yaml.safe_load((a.baseline_online / "train_online_gray_monitor.yaml").read_text())
    config = yaml.safe_load((a.dataset / "train_hardneg_gray_monitor.yaml").read_text())
    old_train = Path(base["train"]).read_text().splitlines()
    new_train = Path(config["train"]).read_text().splitlines()
    if new_train[:len(old_train)] != old_train or len(manifest["train_video_hashes"]) != 44:
        raise ValueError("40-video schedule was not preserved or training video count is not 44")
    if Path(base["val"]).read_bytes() != Path(config["val"]).read_bytes():
        raise ValueError("Validation split changed")
    cache = Path(base["online_replacement"]["cache"])
    summary = json.loads((cache / "summary.json").read_text())
    if summary["stage"] != "complete" or summary["asset_count"] != 328:
        raise ValueError("Expected a complete 328-asset baseline cache")
    if manifest["validation_sha256"] in manifest["train_video_hashes"] or manifest["test_sha256"] in manifest["train_video_hashes"]:
        raise ValueError("Validation or test video appears in training")
    config["online_replacement"] = base["online_replacement"]
    a.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(dict(dataset=str(a.dataset), training_videos=44,
                          preserved_baseline_slots=len(old_train),
                          added_slots=len(new_train) - len(old_train),
                          cache=str(cache), assets=summary["asset_count"],
                          new_video_replacement="original_only_unless_indexed")))


if __name__ == "__main__":
    main()
