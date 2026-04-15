#!/usr/bin/env python3
"""Batch replay and aggregate evaluation for Anti-UAV sequences."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Detector weights used by replay_eval.py.")
    parser.add_argument(
        "--dataset-root",
        default="/mnt/chenziye/datasets/anti_uav/Anti-UAV300",
        help="Anti-UAV300 dataset root containing split directories such as test-dev.",
    )
    parser.add_argument("--split", default="test-dev", help="Dataset split directory to iterate.")
    parser.add_argument("--modality", default="rgb", choices=("rgb", "ir", "auto"), help="Sequence modality selector.")
    parser.add_argument("--output-root", required=True, help="Directory for per-sequence outputs and aggregate JSON.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python binary used to invoke replay_eval.py.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of sequences to process.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip sequences with an existing summary.json.")
    parser.add_argument("--save-video", action="store_true", help="Save one annotated replay video per sequence.")
    parser.add_argument("--verbose", action="store_true", help="Print each replay command before running it.")
    args, forward_args = parser.parse_known_args()
    return args, forward_args


def iter_sequence_dirs(split_root: Path) -> Iterable[Path]:
    for path in sorted(split_root.iterdir()):
        if path.is_dir():
            yield path


def aggregate_metrics(sequence_summaries: list[dict]) -> dict:
    totals = {
        "sequence_count": len(sequence_summaries),
        "total_frames": 0,
        "gt_present_frames": 0,
        "predicted_frames": 0,
        "tp_frames": 0,
        "fp_frames": 0,
        "fn_frames": 0,
        "matched_frames": 0,
        "iou_sum": 0.0,
        "alert_raised": 0,
        "alert_hit": 0,
    }
    for summary in sequence_summaries:
        for key in totals:
            if key == "sequence_count":
                continue
            totals[key] += summary.get(key, 0)

    tp = totals["tp_frames"]
    fp = totals["fp_frames"]
    fn = totals["fn_frames"]
    matched = totals["matched_frames"]
    alert_raised = totals["alert_raised"]
    totals["precision"] = tp / max(tp + fp, 1)
    totals["recall"] = tp / max(tp + fn, 1)
    totals["alert_precision"] = totals["alert_hit"] / max(alert_raised, 1)
    totals["avg_iou"] = totals["iou_sum"] / max(matched, 1)
    totals["mean_sequence_precision"] = sum(item["precision"] for item in sequence_summaries) / max(len(sequence_summaries), 1)
    totals["mean_sequence_recall"] = sum(item["recall"] for item in sequence_summaries) / max(len(sequence_summaries), 1)
    totals["mean_sequence_avg_iou"] = sum(item["avg_iou"] for item in sequence_summaries) / max(len(sequence_summaries), 1)
    return totals


def main() -> None:
    args, forward_args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_root = dataset_root / args.split
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not split_root.exists():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")

    replay_script = ROOT / "scripts" / "anti_uav" / "replay_eval.py"
    if not replay_script.exists():
        raise FileNotFoundError(f"Replay script not found: {replay_script}")

    processed = 0
    sequence_summaries: list[dict] = []
    failures: list[dict] = []

    for sequence_root in iter_sequence_dirs(split_root):
        if args.limit and processed >= args.limit:
            break

        sequence_name = sequence_root.name
        sequence_out = output_root / sequence_name
        summary_path = sequence_out / "summary.json"

        if args.skip_existing and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            sequence_summaries.append(summary)
            processed += 1
            continue

        sequence_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_bin,
            str(replay_script),
            "--model",
            args.model,
            "--sequence-root",
            str(sequence_root),
            "--dataset-format",
            "anti-uav-json",
            "--modality",
            args.modality,
            "--summary-json",
            str(summary_path),
            "--error-log",
            str(sequence_out / "errors.jsonl"),
            "--state-log",
            str(sequence_out / "states.jsonl"),
            "--alert-log",
            str(sequence_out / "alerts.jsonl"),
        ]
        if args.save_video:
            cmd.extend(["--save-video", str(sequence_out / "replay.mp4")])
        cmd.extend(forward_args)

        if args.verbose:
            print("Running:", " ".join(cmd), flush=True)

        completed = subprocess.run(cmd, text=True, capture_output=True)
        if completed.returncode != 0:
            failures.append(
                {
                    "sequence": sequence_name,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            (sequence_out / "failure.log").write_text(
                f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n",
                encoding="utf-8",
            )
            processed += 1
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["sequence"] = sequence_name
        sequence_summaries.append(summary)
        processed += 1

    aggregate = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "modality": args.modality,
        "model": args.model,
        "forward_args": forward_args,
        "aggregate": aggregate_metrics(sequence_summaries),
        "sequence_summaries": sequence_summaries,
        "failures": failures,
    }
    aggregate_path = output_root / "aggregate_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(aggregate["aggregate"], indent=2, ensure_ascii=False))
    if failures:
        print(f"{len(failures)} sequences failed. See {output_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
