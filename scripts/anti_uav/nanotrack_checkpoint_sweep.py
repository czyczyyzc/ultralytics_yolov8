#!/usr/bin/env python3
"""Evaluate multiple NanoTrack checkpoints on the Anti-UAV validation split and rank them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.nanotrack_val_eval import parse_args as _unused  # noqa: F401
from scripts.anti_uav.nanotrack_val_eval import resolve_split_sequences, write_json
from scripts.anti_uav.nanotrack_val_eval import build_nanotrack, evaluate_sequence, aggregate_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Directory containing NanoTrack checkpoints.")
    parser.add_argument("--config", type=Path, required=True, help="NanoTrack config yaml.")
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/chenziye/datasets/anti_uav/Anti-UAV300"), help="Original Anti-UAV300 root.")
    parser.add_argument("--converted-root", type=Path, default=Path("/mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack"), help="Converted NanoTrack dataset root.")
    parser.add_argument("--modality", choices=("rgb", "ir"), default="rgb", help="Validation modality.")
    parser.add_argument("--split", choices=("train", "val"), default="val", help="Validation split.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fallback split ratio when no split manifest exists.")
    parser.add_argument("--nanotrack-root", type=Path, default=None, help="Optional NanoTrack workspace root.")
    parser.add_argument("--device", default="cpu", help="Torch device for NanoTrack.")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Score threshold for valid predictions.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU success threshold.")
    parser.add_argument("--center-threshold", type=float, default=20.0, help="Center precision threshold in pixels.")
    parser.add_argument("--metric", choices=("composite", "success_rate", "avg_iou", "precision", "recall"), default="composite", help="Ranking metric.")
    parser.add_argument(
        "--hard-sequence-patterns",
        nargs="*",
        default=[],
        help="Optional sequence-name substrings for a hard subset used during final checkpoint selection.",
    )
    parser.add_argument(
        "--hard-metric",
        choices=("composite", "success_rate", "avg_iou", "precision", "recall"),
        default="composite",
        help="Metric used to rank checkpoints inside the hard-subset shortlist.",
    )
    parser.add_argument(
        "--hard-overall-tolerance",
        type=float,
        default=0.02,
        help="Only checkpoints within this absolute gap of the best overall metric enter the hard-subset shortlist.",
    )
    parser.add_argument(
        "--hard-min-sequences",
        type=int,
        default=1,
        help="Require at least this many matched hard-subset sequences before using the two-stage selector.",
    )
    parser.add_argument("--max-sequences", type=int, default=0, help="Optional cap on the number of validation sequences.")
    parser.add_argument("--pattern", default="epoch_*.pth", help="Glob pattern for checkpoint sweep.")
    parser.add_argument("--include-best", action="store_true", help="Include best.pth when present.")
    parser.add_argument("--include-last", action="store_true", help="Include last.pth when present.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON.")
    parser.add_argument("--dry-run", action="store_true", help="List discovered checkpoints and sequences without evaluating.")
    return parser.parse_args()


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    checkpoints = sorted(snapshot_dir.glob(args.pattern))
    if args.include_best:
        best_path = snapshot_dir / "best.pth"
        if best_path.exists():
            checkpoints.append(best_path)
    if args.include_last:
        last_path = snapshot_dir / "last.pth"
        if last_path.exists():
            checkpoints.append(last_path)
    unique = []
    seen = set()
    for path in checkpoints:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _matches_hard_sequence(name: str, patterns: list[str]) -> bool:
    """Return True when a sequence name matches any requested hard-subset pattern."""
    if not patterns:
        return False
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns if pattern)


def select_best_checkpoint(
    results: list[dict],
    *,
    metric: str,
    hard_metric: str,
    hard_overall_tolerance: float,
    hard_min_sequences: int,
) -> tuple[dict | None, dict]:
    """Select the best checkpoint using overall ranking first, then a hard-subset tie-break shortlist."""
    if not results:
        return None, {"strategy": "no_results", "shortlist_count": 0}

    sorted_results = sorted(results, key=lambda item: item["aggregate"][metric], reverse=True)
    best_overall = sorted_results[0]
    selection = {
        "strategy": "overall_metric_only",
        "best_overall_metric": best_overall["aggregate"][metric],
        "shortlist_count": 0,
        "hard_metric": hard_metric,
        "hard_overall_tolerance": hard_overall_tolerance,
        "hard_min_sequences": hard_min_sequences,
    }

    hard_eligible = []
    for item in sorted_results:
        hard_aggregate = item.get("hard_aggregate")
        if not hard_aggregate:
            continue
        if hard_aggregate.get("sequence_count", 0) < hard_min_sequences:
            continue
        if best_overall["aggregate"][metric] - item["aggregate"][metric] > hard_overall_tolerance:
            continue
        hard_eligible.append(item)

    if not hard_eligible:
        return best_overall, selection

    hard_eligible.sort(
        key=lambda item: (
            item["hard_aggregate"][hard_metric],
            item["aggregate"][metric],
        ),
        reverse=True,
    )
    best_hard = hard_eligible[0]
    selection.update(
        {
            "strategy": "overall_then_hard_subset",
            "shortlist_count": len(hard_eligible),
            "shortlist_checkpoints": [item["checkpoint"] for item in hard_eligible],
            "selected_hard_metric": best_hard["hard_aggregate"][hard_metric],
            "selected_overall_metric": best_hard["aggregate"][metric],
        }
    )
    return best_hard, selection


def main() -> None:
    args = parse_args()
    entries = resolve_split_sequences(
        args.source_root.expanduser().resolve(),
        args.converted_root.expanduser().resolve(),
        args.modality,
        args.split,
        args.val_ratio,
    )
    if args.max_sequences:
        entries = entries[: args.max_sequences]
    checkpoints = discover_checkpoints(args)
    hard_sequence_names = [entry["name"] for entry in entries if _matches_hard_sequence(entry["name"], args.hard_sequence_patterns)]

    manifest = {
        "snapshot_dir": str(args.snapshot_dir.expanduser().resolve()),
        "config": str(args.config.expanduser().resolve()),
        "metric": args.metric,
        "hard_metric": args.hard_metric,
        "hard_sequence_patterns": args.hard_sequence_patterns,
        "hard_sequence_names": hard_sequence_names,
        "hard_overall_tolerance": args.hard_overall_tolerance,
        "hard_min_sequences": args.hard_min_sequences,
        "checkpoints": [str(path) for path in checkpoints],
        "sequence_names": [entry["name"] for entry in entries],
    }
    if args.dry_run:
        if args.output_json:
            write_json(args.output_json.expanduser().resolve(), manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    results = []
    for checkpoint in checkpoints:
        tracker_args = argparse.Namespace(
            config=args.config,
            snapshot=checkpoint,
            nanotrack_root=args.nanotrack_root,
            device=args.device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            center_threshold=args.center_threshold,
        )
        tracker = build_nanotrack(tracker_args)
        sequence_results = []
        for entry in entries:
            tracker.reset()
            sequence_results.append(evaluate_sequence(entry, tracker, tracker_args))
        aggregate = aggregate_results(sequence_results)
        hard_results = [result for result in sequence_results if _matches_hard_sequence(result["sequence"], args.hard_sequence_patterns)]
        hard_aggregate = aggregate_results(hard_results) if hard_results else None
        results.append(
            {
                "checkpoint": str(checkpoint),
                "aggregate": aggregate,
                "hard_aggregate": hard_aggregate,
            }
        )

    results.sort(key=lambda item: item["aggregate"][args.metric], reverse=True)
    best_checkpoint, selection = select_best_checkpoint(
        results,
        metric=args.metric,
        hard_metric=args.hard_metric,
        hard_overall_tolerance=args.hard_overall_tolerance,
        hard_min_sequences=args.hard_min_sequences,
    )
    payload = {**manifest, "results": results, "selection": selection, "best_checkpoint": best_checkpoint}
    if args.output_json:
        write_json(args.output_json.expanduser().resolve(), payload)
    print(json.dumps(payload["best_checkpoint"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
