#!/usr/bin/env python3
"""Blend a proven calibration manifest with a newer stratified calibration audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.prepare_stratified_rknn_calibration import CATEGORIES, category_budgets


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_stratified(rows: list[dict], count: int, rng: np.random.Generator) -> list[dict]:
    budgets = category_budgets(count)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    selected = []
    for category in CATEGORIES:
        pool = grouped[category]
        selected.extend(np.asarray(pool, dtype=object)[rng.permutation(len(pool))[: budgets[category]]].tolist())
    if len(selected) != count:
        raise ValueError(f"Unable to select {count} stratified rows: {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--stratified-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--base-fraction", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refuse to overwrite non-empty output: {output}")
    if args.samples < len(CATEGORIES) * 2 or not 0 < args.base_fraction < 1:
        raise ValueError("Invalid sample count or base fraction")
    output.mkdir(parents=True, exist_ok=True)

    base_path = args.base.expanduser().resolve()
    audit_path = args.stratified_audit.expanduser().resolve()
    base = [line.strip() for line in base_path.read_text().splitlines() if line.strip()]
    rows = json.loads(audit_path.read_text())["rows"]
    if any(token in path for token in ("Video00004", "Video00009") for path in base):
        raise ValueError("Base calibration contains an evaluation video")
    if any(token in row["image"] for token in ("Video00004", "Video00009") for row in rows):
        raise ValueError("Stratified calibration contains an evaluation video")

    rng = np.random.default_rng(args.seed)
    base_count = round(args.samples * args.base_fraction)
    new_count = args.samples - base_count
    if len(base) < base_count:
        raise ValueError(f"Base calibration has only {len(base)} rows")
    base_selected = np.asarray(base, dtype=object)[rng.permutation(len(base))[:base_count]].tolist()
    new_rows = choose_stratified(rows, new_count, rng)
    manifest = base_selected + [row["calibration_image"] for row in new_rows]
    rng.shuffle(manifest)
    if len(manifest) != len(set(manifest)):
        raise ValueError("Duplicate calibration images in blended manifest")
    for path in manifest:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    dataset = output / "dataset.txt"
    dataset.write_text("\n".join(manifest) + "\n")
    audit = {
        "samples": len(manifest),
        "seed": args.seed,
        "base": str(base_path),
        "base_sha256": sha256(base_path),
        "base_samples": base_count,
        "stratified_audit": str(audit_path),
        "stratified_audit_sha256": sha256(audit_path),
        "stratified_samples": new_count,
        "stratified_category_counts": {
            category: sum(row["category"] == category for row in new_rows) for category in CATEGORIES
        },
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
