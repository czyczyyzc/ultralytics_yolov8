#!/usr/bin/env python3
"""Rebalance duplicate positive slots toward hard examples while preserving neg15."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--hard-positives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-positive-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as error:
        raise ValueError(f"Image path has no images component: {image_path}") from error
    return Path(*parts).with_suffix(".txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def is_positive(path: Path) -> bool:
    return bool(label_path(path).read_text().strip())


def read_hard_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hard_rehearsal_schedule(records: list[dict], quota: int, seed: int) -> list[Path]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for record in records:
        path = Path(record["image"])
        for _ in range(int(record["extra_repeats"])):
            groups[str(record["group"])].append(path)
    for index, group in enumerate(sorted(groups)):
        random.Random(seed + index).shuffle(groups[group])

    schedule: list[Path] = []
    positions = {group: 0 for group in groups}
    while len(schedule) < quota:
        progressed = False
        for group in sorted(groups):
            position = positions[group]
            if position >= len(groups[group]):
                continue
            schedule.append(groups[group][position])
            positions[group] += 1
            progressed = True
            if len(schedule) == quota:
                break
        if not progressed:
            break
    return schedule


def build_rehearsal_mix(
    source_paths: list[Path],
    hard_records: list[dict],
    hard_positive_fraction: float,
    seed: int,
) -> tuple[list[Path], dict]:
    if not 0.0 <= hard_positive_fraction <= 1.0:
        raise ValueError("hard-positive-fraction must be in [0, 1]")

    positive_flags = [is_positive(path) for path in source_paths]
    positive_count = sum(positive_flags)
    negative_count = len(source_paths) - positive_count
    source_positive_paths = [path for path, positive in zip(source_paths, positive_flags) if positive]
    source_unique_positives = set(source_positive_paths)

    validated_records: list[dict] = []
    for record in hard_records:
        path = Path(record["image"])
        if path not in source_unique_positives:
            raise ValueError(f"Hard positive is absent from source training data: {path}")
        if not is_positive(path):
            raise ValueError(f"Hard-positive record has an empty label: {path}")
        validated_records.append(record)

    seen: Counter[Path] = Counter()
    replaceable_indices: list[int] = []
    for index, (path, positive) in enumerate(zip(source_paths, positive_flags)):
        if not positive:
            continue
        seen[path] += 1
        if seen[path] > 1:
            replaceable_indices.append(index)

    requested_quota = round(positive_count * hard_positive_fraction)
    schedule = hard_rehearsal_schedule(
        validated_records,
        min(requested_quota, len(replaceable_indices)),
        seed,
    )
    random.Random(seed).shuffle(replaceable_indices)
    output_paths = list(source_paths)
    for index, hard_path in zip(replaceable_indices, schedule):
        output_paths[index] = hard_path

    output_positive_flags = [is_positive(path) for path in output_paths]
    output_negatives = [path for path, positive in zip(output_paths, output_positive_flags) if not positive]
    source_negatives = [path for path, positive in zip(source_paths, positive_flags) if not positive]
    output_unique_positives = {path for path, positive in zip(output_paths, output_positive_flags) if positive}
    if output_negatives != source_negatives:
        raise RuntimeError("Negative sample identity or order changed")
    if output_unique_positives != source_unique_positives:
        raise RuntimeError("Unique positive coverage changed")

    categories = Counter(record["category"] for record in validated_records)
    manifest = {
        "source_training_samples": len(source_paths),
        "training_samples": len(output_paths),
        "positive_samples": positive_count,
        "negative_samples": negative_count,
        "negative_fraction": negative_count / max(len(output_paths), 1),
        "unique_positive_images": len(source_unique_positives),
        "hard_positive_candidates": len(validated_records),
        "hard_positive_candidate_categories": dict(sorted(categories.items())),
        "hard_positive_fraction_requested": hard_positive_fraction,
        "hard_positive_rehearsal_slots": len(schedule),
        "replaceable_duplicate_positive_slots": len(replaceable_indices),
        "negative_samples_unchanged": True,
        "unique_positive_coverage_unchanged": True,
    }
    return output_paths, manifest


def main() -> None:
    args = parse_args()
    source_data = yaml.safe_load(args.source_data.read_text())
    source_train = Path(source_data["train"])
    source_paths = [Path(line.strip()) for line in source_train.read_text().splitlines() if line.strip()]
    hard_records = read_hard_records(args.hard_positives)
    output_paths, manifest = build_rehearsal_mix(
        source_paths,
        hard_records,
        args.hard_positive_fraction,
        args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    train_list = args.output / "train_hard_positive_rehearsal.txt"
    train_list.write_text("".join(f"{path}\n" for path in output_paths))
    output_data = dict(source_data)
    output_data["path"] = str(args.output)
    output_data["train"] = str(train_list)
    data_yaml = args.output / "train_rgb_monitor.yaml"
    data_yaml.write_text(yaml.safe_dump(output_data, sort_keys=False))

    manifest.update(
        {
            "schema_version": "anti_uav.hard_positive_rehearsal_mix.v1",
            "seed": args.seed,
            "source_data": str(args.source_data),
            "source_train": str(source_train),
            "source_train_sha256": sha256_file(source_train),
            "hard_positives": str(args.hard_positives),
            "hard_positives_sha256": sha256_file(args.hard_positives),
            "train_list": str(train_list),
            "data_yaml": str(data_yaml),
        }
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
