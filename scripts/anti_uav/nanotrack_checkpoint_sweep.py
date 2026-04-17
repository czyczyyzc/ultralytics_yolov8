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

    manifest = {
        "snapshot_dir": str(args.snapshot_dir.expanduser().resolve()),
        "config": str(args.config.expanduser().resolve()),
        "metric": args.metric,
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
        results.append({"checkpoint": str(checkpoint), "aggregate": aggregate})

    results.sort(key=lambda item: item["aggregate"][args.metric], reverse=True)
    payload = {**manifest, "results": results, "best_checkpoint": results[0] if results else None}
    if args.output_json:
        write_json(args.output_json.expanduser().resolve(), payload)
    print(json.dumps(payload["best_checkpoint"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
