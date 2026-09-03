#!/usr/bin/env python3
"""Select a detector checkpoint using validation fitness without consulting the test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="Repeat for each run directory containing results.csv and weights/best.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-name", default="Anti-UAV300 RGB")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate(value: str) -> dict:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=RUN_DIR, received {value!r}")
    label, raw_path = value.split("=", 1)
    run_dir = Path(raw_path).expanduser().resolve()
    results_path = run_dir / "results.csv"
    checkpoint = run_dir / "weights/best.pt"
    if not results_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Incomplete training run: {run_dir}")
    rows = list(csv.DictReader(results_path.open(), skipinitialspace=True))
    if not rows:
        raise ValueError(f"No epochs found in {results_path}")
    for row in rows:
        row["fitness"] = 0.1 * float(row["metrics/mAP50(B)"]) + 0.9 * float(row["metrics/mAP50-95(B)"])
    best = max(rows, key=lambda row: row["fitness"])
    return {
        "stage": label,
        "epoch": int(float(best["epoch"])),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "fitness": best["fitness"],
        "checkpoint": str(checkpoint),
    }


def main() -> None:
    args = parse_args()
    candidates = [load_candidate(value) for value in args.stage]
    selected = max(candidates, key=lambda candidate: candidate["fitness"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied_checkpoint = args.output_dir / "best.pt"
    shutil.copy2(selected["checkpoint"], copied_checkpoint)
    selected = {
        **selected,
        "copied_checkpoint": str(copied_checkpoint.resolve()),
        "sha256": sha256(copied_checkpoint),
    }
    manifest = {
        "selection_policy": f"max(0.1*mAP50 + 0.9*mAP50-95) on {args.validation_name} validation only",
        "test_set_used_for_selection": False,
        "candidates": candidates,
        "selected": selected,
    }
    manifest_path = args.output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
