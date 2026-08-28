#!/usr/bin/env python3
"""Summarize before/after YOLO real-gray LOVO evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adapted-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_results(directory: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(directory.glob("holdout_*.json")):
        result = json.loads(path.read_text())
        results[result["fold"]] = result
    if not results:
        raise FileNotFoundError(f"No holdout JSON files found in {directory}")
    return results


def macro_standard(results: dict[str, dict]) -> dict[str, float]:
    keys = ("precision", "recall", "map50", "map50_95")
    return {key: mean(result["gray_holdout_standard"][key] for result in results.values()) for key in keys}


def micro_fixed(results: dict[str, dict]) -> dict[str, dict[str, float | int]]:
    thresholds = next(iter(results.values()))["gray_holdout_fixed_thresholds"].keys()
    output = {}
    for threshold in thresholds:
        rows = [result["gray_holdout_fixed_thresholds"][threshold] for result in results.values()]
        tp = sum(row["tp"] for row in rows)
        fp = sum(row["fp"] for row in rows)
        fn = sum(row["fn"] for row in rows)
        absent_frames = sum(row["absent_frames"] for row in rows)
        false_positive_absent_frames = sum(row["false_positive_absent_frames"] for row in rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        output[threshold] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "positive_frames": sum(row["positive_frames"] for row in rows),
            "absent_frames": absent_frames,
            "false_positive_absent_frames": false_positive_absent_frames,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "absent_frame_false_positive_rate": false_positive_absent_frames / max(absent_frames, 1),
            "mean_matched_iou": sum(row["mean_matched_iou"] * row["tp"] for row in rows) / max(tp, 1),
        }
    return output


def rgb_standard(results: dict[str, dict]) -> dict[str, float] | None:
    rows = [result["anti_uav300_rgb_standard"] for result in results.values() if "anti_uav300_rgb_standard" in result]
    if not rows:
        return None
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before}


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    baseline = load_results(args.baseline_dir)
    adapted = load_results(args.adapted_dir)
    if baseline.keys() != adapted.keys():
        raise RuntimeError(f"Fold mismatch: baseline={sorted(baseline)}, adapted={sorted(adapted)}")

    baseline_standard = macro_standard(baseline)
    adapted_standard = macro_standard(adapted)
    baseline_fixed = micro_fixed(baseline)
    adapted_fixed = micro_fixed(adapted)
    baseline_rgb = rgb_standard(baseline)
    adapted_rgb = rgb_standard(adapted)
    per_fold = {}
    for fold in sorted(baseline):
        before = baseline[fold]
        after = adapted[fold]
        before_fixed = before["gray_holdout_fixed_thresholds"]["0.25"]
        after_fixed = after["gray_holdout_fixed_thresholds"]["0.25"]
        per_fold[fold] = {
            "baseline_map50": before["gray_holdout_standard"]["map50"],
            "adapted_map50": after["gray_holdout_standard"]["map50"],
            "map50_delta": after["gray_holdout_standard"]["map50"] - before["gray_holdout_standard"]["map50"],
            "baseline_conf025_f1": before_fixed["f1"],
            "adapted_conf025_f1": after_fixed["f1"],
            "conf025_f1_delta": after_fixed["f1"] - before_fixed["f1"],
            "baseline_conf025_absent_fpr": before_fixed["absent_frame_false_positive_rate"],
            "adapted_conf025_absent_fpr": after_fixed["absent_frame_false_positive_rate"],
        }

    summary = {
        "folds": len(baseline),
        "gray_standard_macro": {
            "baseline": baseline_standard,
            "adapted": adapted_standard,
            "delta": delta(adapted_standard, baseline_standard),
        },
        "gray_fixed_threshold_micro": {
            threshold: {
                "baseline": baseline_fixed[threshold],
                "adapted": adapted_fixed[threshold],
                "delta": {
                    key: adapted_fixed[threshold][key] - baseline_fixed[threshold][key]
                    for key in ("precision", "recall", "f1", "absent_frame_false_positive_rate")
                },
            }
            for threshold in baseline_fixed
        },
        "anti_uav300_rgb_standard": {
            "baseline": baseline_rgb,
            "adapted": adapted_rgb,
            "delta": delta(adapted_rgb, baseline_rgb) if baseline_rgb and adapted_rgb else None,
        },
        "per_fold": per_fold,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# YOLOv8n Anti-UAV300 + Real-gray LOVO Result",
        "",
        "## Aggregate",
        "",
        "| Metric | Baseline | Adapted | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("map50", "Gray macro mAP50"), ("map50_95", "Gray macro mAP50-95")):
        lines.append(
            f"| {label} | {pct(baseline_standard[key])} | {pct(adapted_standard[key])} | "
            f"{pct(adapted_standard[key] - baseline_standard[key])} |"
        )
    for threshold in ("0.10", "0.25", "0.45"):
        before = baseline_fixed[threshold]
        after = adapted_fixed[threshold]
        lines.append(
            f"| Gray conf={threshold} micro F1 | {pct(before['f1'])} | {pct(after['f1'])} | "
            f"{pct(after['f1'] - before['f1'])} |"
        )
        lines.append(
            f"| Gray conf={threshold} absent-frame FPR | {pct(before['absent_frame_false_positive_rate'])} | "
            f"{pct(after['absent_frame_false_positive_rate'])} | "
            f"{pct(after['absent_frame_false_positive_rate'] - before['absent_frame_false_positive_rate'])} |"
        )
    if baseline_rgb and adapted_rgb:
        lines.append(
            f"| Anti-UAV300 RGB mAP50-95 | {pct(baseline_rgb['map50_95'])} | {pct(adapted_rgb['map50_95'])} | "
            f"{pct(adapted_rgb['map50_95'] - baseline_rgb['map50_95'])} |"
        )

    lines.extend(["", "## Per Fold", "", "| Holdout | Baseline mAP50 | Adapted mAP50 | Delta |", "|---|---:|---:|---:|"])
    for fold, row in per_fold.items():
        lines.append(
            f"| {fold.removeprefix('holdout_')} | {pct(row['baseline_map50'])} | {pct(row['adapted_map50'])} | "
            f"{pct(row['map50_delta'])} |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
