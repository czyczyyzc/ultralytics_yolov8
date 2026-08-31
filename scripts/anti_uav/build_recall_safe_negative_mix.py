#!/usr/bin/env python3
"""Build a temporally diverse negative mix while protecting target recall."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-list", type=Path, required=True)
    parser.add_argument("--negative-pool", type=Path, required=True)
    parser.add_argument("--hard-negative-list", type=Path, default=None)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--rgb-val-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-fraction", type=float, required=True)
    parser.add_argument(
        "--guard-seconds",
        type=float,
        default=0.5,
        help="Exclude absent gray frames this close to a visible-target frame.",
    )
    parser.add_argument(
        "--temporal-sample-seconds",
        type=float,
        default=0.05,
        help="Keep one regular gray negative per interval; safe hard negatives bypass thinning.",
    )
    parser.add_argument(
        "--hard-negative-max-fraction",
        type=float,
        default=0.15,
        help="Maximum fraction of the selected negative quota reserved for ranked hard negatives.",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def read_paths(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def write_paths(path: Path, paths: list[Path]) -> None:
    path.write_text("".join(f"{item}\n" for item in paths))


def parse_gray_frame(path: Path) -> tuple[str, int] | None:
    parts = path.parts
    for index, part in enumerate(parts[:-2]):
        if part != "gray":
            continue
        sequence = parts[index + 1]
        if sequence.startswith("Video") and path.stem.isdigit():
            return sequence, int(path.stem)
    return None


def stable_phase(seed: int, sequence: str, stride: int) -> int:
    if stride <= 1:
        return 0
    digest = hashlib.sha256(f"{seed}:{sequence}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % stride


def nearest_distance(sorted_indices: list[int], frame: int) -> int:
    position = bisect.bisect_left(sorted_indices, frame)
    distances = []
    if position < len(sorted_indices):
        distances.append(sorted_indices[position] - frame)
    if position:
        distances.append(frame - sorted_indices[position - 1])
    return min(distances) if distances else 1 << 60


def filter_gray_negatives(
    candidates: list[Path],
    hard_negatives: list[Path],
    exists_by_sequence: dict[str, list[bool]],
    fps_by_sequence: dict[str, float],
    guard_seconds: float,
    temporal_sample_seconds: float,
    seed: int,
) -> tuple[list[Path], list[Path], dict]:
    """Remove transition-adjacent and near-duplicate gray negatives."""
    if guard_seconds < 0.0:
        raise ValueError("guard_seconds must be non-negative")
    if temporal_sample_seconds < 0.0:
        raise ValueError("temporal_sample_seconds must be non-negative")

    hard_set = set(hard_negatives)
    visible_by_sequence = {
        sequence: [index for index, present in enumerate(exists) if present]
        for sequence, exists in exists_by_sequence.items()
    }
    filtered: list[Path] = []
    safe_hard: list[Path] = []
    counts: dict[str, int | dict[str, dict[str, int]]] = {
        "input_candidates": 0,
        "rgb_candidates": 0,
        "gray_candidates": 0,
        "guard_excluded": 0,
        "temporal_thinning_excluded": 0,
        "safe_hard_negatives": 0,
        "per_sequence": {},
    }

    for path in dict.fromkeys(candidates):
        counts["input_candidates"] += 1
        reference = parse_gray_frame(path)
        if reference is None:
            counts["rgb_candidates"] += 1
            filtered.append(path)
            if path in hard_set:
                safe_hard.append(path)
            continue

        sequence, frame = reference
        if sequence not in exists_by_sequence or sequence not in fps_by_sequence:
            raise KeyError(f"Missing annotation or FPS metadata for {sequence}")
        exists = exists_by_sequence[sequence]
        if frame >= len(exists):
            raise IndexError(f"Frame {frame} is outside {sequence} annotation length {len(exists)}")
        if exists[frame]:
            raise ValueError(f"Negative pool contains a positive frame: {path}")

        counts["gray_candidates"] += 1
        sequence_counts = counts["per_sequence"].setdefault(
            sequence,
            {"input": 0, "guard_excluded": 0, "thinned": 0, "kept": 0, "safe_hard": 0},
        )
        sequence_counts["input"] += 1
        fps = fps_by_sequence[sequence]
        guard_frames = round(guard_seconds * fps)
        if nearest_distance(visible_by_sequence[sequence], frame) <= guard_frames:
            counts["guard_excluded"] += 1
            sequence_counts["guard_excluded"] += 1
            continue

        is_hard = path in hard_set
        stride = max(1, round(temporal_sample_seconds * fps))
        phase = stable_phase(seed, sequence, stride)
        if not is_hard and (frame - phase) % stride:
            counts["temporal_thinning_excluded"] += 1
            sequence_counts["thinned"] += 1
            continue

        filtered.append(path)
        sequence_counts["kept"] += 1
        if is_hard:
            safe_hard.append(path)
            counts["safe_hard_negatives"] += 1
            sequence_counts["safe_hard"] += 1

    counts["filtered_candidates"] = len(filtered)
    return filtered, safe_hard, counts


def candidate_group(path: Path) -> str:
    reference = parse_gray_frame(path)
    return reference[0] if reference else "rgb"


def select_stratified_negatives(
    candidates: list[Path],
    ranked_hard_negatives: list[Path],
    positive_count: int,
    negative_fraction: float,
    hard_negative_max_fraction: float,
    seed: int,
) -> tuple[list[Path], int, dict[str, int]]:
    """Select an exact quota with bounded hard negatives and balanced sequence coverage."""
    if not 0.0 <= negative_fraction < 1.0:
        raise ValueError("negative_fraction must be in [0, 1)")
    if not 0.0 <= hard_negative_max_fraction <= 1.0:
        raise ValueError("hard_negative_max_fraction must be in [0, 1]")
    if negative_fraction == 0.0:
        return [], 0, {}

    unique_candidates = list(dict.fromkeys(candidates))
    candidate_set = set(unique_candidates)
    target_count = round(positive_count * negative_fraction / (1.0 - negative_fraction))
    if target_count > len(unique_candidates):
        raise RuntimeError(
            f"Need {target_count} negatives for fraction {negative_fraction:.3f}, "
            f"but only {len(unique_candidates)} recall-safe candidates are available"
        )

    hard_limit = round(target_count * hard_negative_max_fraction)
    prioritized = [
        path for path in dict.fromkeys(ranked_hard_negatives) if path in candidate_set
    ][:hard_limit]
    selected = list(prioritized)
    selected_set = set(selected)

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in unique_candidates:
        if path not in selected_set:
            groups[candidate_group(path)].append(path)
    for index, key in enumerate(sorted(groups)):
        random.Random(seed + index).shuffle(groups[key])

    group_keys = sorted(groups)
    positions = {key: 0 for key in group_keys}
    while len(selected) < target_count:
        progressed = False
        for key in group_keys:
            position = positions[key]
            if position >= len(groups[key]):
                continue
            selected.append(groups[key][position])
            positions[key] += 1
            progressed = True
            if len(selected) == target_count:
                break
        if not progressed:
            raise RuntimeError("Recall-safe negative groups were exhausted before reaching the quota")

    per_group: dict[str, int] = defaultdict(int)
    for path in selected:
        per_group[candidate_group(path)] += 1
    return selected, len(prioritized), dict(sorted(per_group.items()))


def main() -> None:
    args = parse_args()
    source_manifest = json.loads(args.source_manifest.read_text())
    gray_videos = source_manifest["gray_videos"]
    exists_by_sequence: dict[str, list[bool]] = {}
    fps_by_sequence: dict[str, float] = {}
    for sequence, metadata in gray_videos.items():
        annotation = json.loads(Path(metadata["annotation"]).read_text())
        exists_by_sequence[sequence] = [bool(value) for value in annotation["exist"]]
        fps_by_sequence[sequence] = float(Fraction(metadata["fps"]))

    positives = read_paths(args.positive_list)
    negative_pool = read_paths(args.negative_pool)
    hard_negatives = read_paths(args.hard_negative_list) if args.hard_negative_list else []
    candidates, safe_hard, filter_counts = filter_gray_negatives(
        negative_pool,
        hard_negatives,
        exists_by_sequence,
        fps_by_sequence,
        args.guard_seconds,
        args.temporal_sample_seconds,
        args.seed,
    )
    selected, hard_count, per_group = select_stratified_negatives(
        candidates,
        safe_hard,
        len(positives),
        args.negative_fraction,
        args.hard_negative_max_fraction,
        args.seed,
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    mixed = positives + selected
    random.Random(args.seed).shuffle(mixed)
    positive_path = output / "train_positive.txt"
    negative_path = output / "train_negative_selected.txt"
    mixed_path = output / "train_mixed.txt"
    yaml_path = output / "train_rgb_monitor.yaml"
    write_paths(positive_path, positives)
    write_paths(negative_path, selected)
    write_paths(mixed_path, mixed)
    yaml_path.write_text(
        f"path: {output}\n"
        f"train: {mixed_path}\n"
        f"val: {args.rgb_val_list}\n\n"
        "names:\n"
        "  0: drone\n"
    )

    manifest = {
        "schema_version": "anti_uav.recall_safe_negative_mix.v1",
        "seed": args.seed,
        "positive_list": str(args.positive_list),
        "negative_pool": str(args.negative_pool),
        "hard_negative_list": str(args.hard_negative_list) if args.hard_negative_list else None,
        "source_manifest": str(args.source_manifest),
        "policy": {
            "negative_fraction": args.negative_fraction,
            "guard_seconds": args.guard_seconds,
            "temporal_sample_seconds": args.temporal_sample_seconds,
            "hard_negative_max_fraction": args.hard_negative_max_fraction,
            "selection": "ranked safe hard negatives, then round-robin across RGB and gray videos",
        },
        "positive_samples": len(positives),
        "selected_negative_samples": len(selected),
        "selected_hard_negative_samples": hard_count,
        "actual_negative_fraction": len(selected) / len(mixed),
        "training_samples": len(mixed),
        "selected_negatives_per_group": per_group,
        "filter": filter_counts,
        "train_list": str(mixed_path),
        "data_yaml": str(yaml_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
