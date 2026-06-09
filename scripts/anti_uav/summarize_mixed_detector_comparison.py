#!/usr/bin/env python3
"""Summarize mixed detector comparison eval roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = ("anti_uav", "hanlue_old", "hanlue_new")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in name=/path/to/eval_root form. Can be repeated.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pick(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def load_detection(eval_root: Path, dataset: str) -> dict[str, Any]:
    path = eval_root / "detection_only" / dataset / "metrics.json"
    data = read_json(path)
    if data is None:
        return {"status": "missing", "path": str(path)}
    results = data.get("results_dict", {})
    return {
        "status": "ok",
        "path": str(path),
        "precision": pick(results, "metrics/precision(B)", "precision"),
        "recall": pick(results, "metrics/recall(B)", "recall"),
        "map50": pick(results, "metrics/mAP50(B)", "map50"),
        "map50_95": pick(results, "metrics/mAP50-95(B)", "map50_95"),
        "fitness": data.get("fitness"),
        "speed": data.get("speed", {}),
    }


def load_tracking(eval_root: Path, dataset: str) -> dict[str, Any]:
    path = eval_root / "tracking" / dataset / "aggregate_summary.json"
    data = read_json(path)
    if data is None:
        return {"status": "missing", "path": str(path)}
    aggregate = data.get("aggregate", {})
    return {
        "status": "ok",
        "path": str(path),
        "sequence_count": aggregate.get("sequence_count"),
        "precision": aggregate.get("precision"),
        "recall": aggregate.get("recall"),
        "avg_iou": aggregate.get("avg_iou"),
        "fp_frames": aggregate.get("fp_frames"),
        "fn_frames": aggregate.get("fn_frames"),
        "alert_precision": aggregate.get("alert_precision"),
    }


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec, expected name=/path: {spec}")
    name, raw_path = spec.split("=", 1)
    if not name:
        raise ValueError(f"Missing run name in --run spec: {spec}")
    return name, Path(raw_path).expanduser().resolve()


def format_float(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def print_markdown(summary: dict[str, Any]) -> None:
    print("## Detection Only")
    print("| run | dataset | P | R | mAP50 | mAP50-95 |")
    print("|---|---|---:|---:|---:|---:|")
    for run_name, run_data in summary["runs"].items():
        for dataset, metrics in run_data["detection_only"].items():
            print(
                "| {run} | {dataset} | {p} | {r} | {m50} | {m95} |".format(
                    run=run_name,
                    dataset=dataset,
                    p=format_float(metrics.get("precision")),
                    r=format_float(metrics.get("recall")),
                    m50=format_float(metrics.get("map50")),
                    m95=format_float(metrics.get("map50_95")),
                )
            )
    print()
    print("## Detection + Tracking")
    print("| run | dataset | seq | P | R | avg IoU | FP frames | FN frames |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for run_name, run_data in summary["runs"].items():
        for dataset, metrics in run_data["tracking"].items():
            print(
                "| {run} | {dataset} | {seq} | {p} | {r} | {iou} | {fp} | {fn} |".format(
                    run=run_name,
                    dataset=dataset,
                    seq=format_float(metrics.get("sequence_count")),
                    p=format_float(metrics.get("precision")),
                    r=format_float(metrics.get("recall")),
                    iou=format_float(metrics.get("avg_iou")),
                    fp=format_float(metrics.get("fp_frames")),
                    fn=format_float(metrics.get("fn_frames")),
                )
            )


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {"runs": {}}
    for spec in args.run:
        run_name, eval_root = parse_run(spec)
        summary["runs"][run_name] = {
            "eval_root": str(eval_root),
            "detection_only": {dataset: load_detection(eval_root, dataset) for dataset in DATASETS},
            "tracking": {dataset: load_tracking(eval_root, dataset) for dataset in DATASETS},
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_markdown(summary)


if __name__ == "__main__":
    main()
