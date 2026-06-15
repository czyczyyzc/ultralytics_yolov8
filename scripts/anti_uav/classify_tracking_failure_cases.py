#!/usr/bin/env python3
"""Classify side-by-side tracking failure videos into failure-case folders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = {
    "01_960x960_covers_640x640_misses": {
        "title": "960x960 可以 cover，但 640x640 miss / 跟丢",
        "cause": "小目标、低对比度或需要更高输入分辨率的帧；640 输入下 detector 初始化/重捕获不足，tracker 后续没有可靠目标。",
        "improve": [
            "保留高分辨率 detector 作为远距离/小目标模式，或在 640 模型上增加小目标样本和更强 crop/scale/mosaic 增强。",
            "加入 tile/ROI re-detect 或动态升分辨率策略：低置信/丢失时临时用 960 或局部高分辨率重检。",
            "针对 Hanlue new 的 low_to_high、tail_chase、edge_reacquire 增加 hard negative/positive replay 训练。",
        ],
    },
    "02_640x640_covers_960x960_misses": {
        "title": "640x640 可以 cover，但 960x960 miss / 跟丢",
        "cause": "高分辨率模型并非总是更稳，可能受训练快照、输入尺度分布、阈值、letterbox/尺度变化或 tracker 初始化框差异影响。",
        "improve": [
            "用相同增强策略重新训练 960，并检查 960 推理阈值是否需要单独校准。",
            "做多尺度一致性蒸馏/ensemble：640 与 960 互相验证，任一模型稳定命中时可触发校正。",
            "检查失败序列的首帧 detector 结果，区分 detector 漏检和 tracker 初始化框漂移。",
        ],
    },
    "03_short_sequence_sparse_miss": {
        "title": "短序列/稀疏帧漏检",
        "cause": "Hanlue old 里一些 val_xxxxxx 序列只有很少帧，任何一帧漏检都会让 recall 大幅下降，指标对单帧非常敏感。",
        "improve": [
            "对短序列单独看 detection 结果，不要过度解读 tracker 长时稳定性。",
            "评估时增加按帧数加权和按序列均值两套指标，避免短序列支配 failure 列表。",
            "训练中补充单帧极小目标、边缘目标样本，提高首帧命中率。",
        ],
    },
    "04_both_initial_acquisition_failure": {
        "title": "两种分辨率都初始捕获失败",
        "cause": "目标一开始就太小、低对比、贴边或运动模糊，detector 多帧没有给出足够高置信框，tracker 无法初始化。",
        "improve": [
            "降低首检阈值或使用双阈值：初始化阶段更低 conf，确认阶段再用 presence verifier 过滤。",
            "加入更多首帧 hard positives，尤其低空到高空、近远尺度突变、背景复杂样本。",
            "加入 temporal detector warmup：连续数帧弱检测聚合后再初始化 tracker。",
        ],
    },
    "05_localization_drift_or_bbox_mismatch": {
        "title": "定位漂移 / bbox 不准",
        "cause": "有预测框但 IoU 不达标，常见于 tracker 贴到背景纹理、框尺度不跟随目标、或 detector 校正框偏移。",
        "improve": [
            "提高 tracker 校正频率，缩短 detect_interval，或在 presence uncertainty 上升时强制 detector 校正。",
            "训练 detector 时增加尺度变化和局部裁剪样本，改善 bbox 尺寸回归。",
            "在 tracker 侧加入 bbox size/速度约束，异常扩大、快速漂移时触发重新检测。",
        ],
    },
    "06_false_positive_after_loss_or_absence": {
        "title": "目标丢失后仍输出框 / 背景误检",
        "cause": "目标 absent 或已离开视野后，tracker 仍粘在背景上，presence verifier 或置信门控没有及时终止轨迹。",
        "improve": [
            "加强 presence verifier 的 absent/背景负样本，尤其来自实际 FP/漂移帧。",
            "降低 max_lost 或对连续低 presence_score / 高 uncertainty 设置更严格 kill-switch。",
            "把这些 FP 片段导出为 hard negatives，回灌 detector 和 presence verifier。",
        ],
    },
    "07_long_term_target_loss_or_reacquire_failure": {
        "title": "长时间跟丢 / 重捕获失败",
        "cause": "目标中途发生尺度突变、快速机动、贴边或遮挡后，tracker 丢失；后续 full-frame/ROI re-detect 没能重新捕获。",
        "improve": [
            "丢失后扩大搜索区域并提高 re-detect 频率，必要时切换到 960 或 tile 模式。",
            "使用 motion prior 和多候选检测，而不是只依赖单一最高分检测框。",
            "对 edge_reacquire/evasive/tail_chase 场景单独补训 tracker 和 detector。",
        ],
    },
    "08_mixed_minor_tracking_errors": {
        "title": "混合型/轻微 tracking 错误",
        "cause": "错误分散，没有单一主导模式；可能包括少量 FN、少量定位误差、短暂漂移或阈值边界问题。",
        "improve": [
            "优先人工抽查，再决定是否纳入 hard example 训练。",
            "对阈值敏感样本做 conf/presence/iou sweep，找到更稳的 operating point。",
            "保留为 regression set，后续模型更新时防止小幅退化。",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-dir", default="failure_case_categories")
    parser.add_argument("--left-model", default="960x960")
    parser.add_argument("--right-model", default="640x640")
    parser.add_argument("--copy-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        counts[item.get("type", "unknown")] += 1
    return counts


def read_states_first_pred(path: Path) -> int | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("bbox") is not None:
            return int(item.get("frame_index", 0) or 0)
    return None


def seq_scenario(dataset: str, sequence: str) -> str:
    if dataset == "hanlue_new":
        for token in ("close_terminal", "edge_reacquire", "tail_chase", "evasive"):
            if token in sequence:
                return token
    if dataset == "anti_uav":
        return "real_anti_uav"
    if dataset == "hanlue_old":
        return "synthetic_single_or_short_clip"
    return "unknown"


def platform(sequence: str) -> str:
    if sequence.startswith("00_x500"):
        return "x500"
    if sequence.startswith("01_mavic"):
        return "mavic"
    if sequence.startswith("02_phantom"):
        return "phantom"
    return "unknown"


def classify(row: dict, features: dict) -> str:
    if row["reason"] == "left_covers_better":
        return "01_960x960_covers_640x640_misses"
    if row["reason"] == "right_covers_better":
        return "02_640x640_covers_960x960_misses"

    frames = int(float(row["frames"]))
    if frames <= 20:
        return "03_short_sequence_sparse_miss"

    left = features["left_summary"]
    right = features["right_summary"]
    left_recall = float(left.get("recall", 0.0))
    right_recall = float(right.get("recall", 0.0))
    left_precision = float(left.get("precision", 0.0))
    right_precision = float(right.get("precision", 0.0))
    left_iou = float(left.get("avg_iou", 0.0))
    right_iou = float(right.get("avg_iou", 0.0))
    gt_frames = max(int(left.get("gt_present_frames", right.get("gt_present_frames", frames)) or 0), 1)
    first_gt = int(left.get("first_gt_frame", right.get("first_gt_frame", 1)) or 1)
    left_first = features["left_first_pred"]
    right_first = features["right_first_pred"]
    left_delay = (left_first - first_gt) if left_first else gt_frames
    right_delay = (right_first - first_gt) if right_first else gt_frames

    combined = features["left_errors"] + features["right_errors"]
    loc = combined.get("localization_error", 0)
    fn = combined.get("false_negative", 0)
    fp = combined.get("false_positive", 0)
    total_error_lines = max(sum(combined.values()), 1)

    if (left_delay >= max(10, 0.15 * gt_frames) and right_delay >= max(10, 0.15 * gt_frames)) or (
        left_recall < 0.2 and right_recall < 0.2
    ):
        return "04_both_initial_acquisition_failure"
    if loc / total_error_lines >= 0.45 and min(left_recall, right_recall) >= 0.55:
        return "05_localization_drift_or_bbox_mismatch"
    if fp > fn * 1.5 and min(left_precision, right_precision) < 0.85:
        return "06_false_positive_after_loss_or_absence"
    if fn >= max(fp, loc) and min(left_recall, right_recall) < 0.8:
        return "07_long_term_target_loss_or_reacquire_failure"
    if min(left_iou, right_iou) < 0.68 and loc > 0:
        return "05_localization_drift_or_bbox_mismatch"
    return "08_mixed_minor_tracking_errors"


def safe_name(dataset: str, sequence: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{dataset}__{sequence}.mp4")


def place_video(source: Path, dest: Path, copy_mode: str) -> str:
    if dest.exists():
        return "existing"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, dest)
        return "copy"
    if copy_mode == "symlink":
        os.symlink(source, dest)
        return "symlink"
    try:
        os.link(source, dest)
        return "hardlink"
    except OSError:
        os.symlink(source, dest)
        return "symlink"


def write_manifest(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md_metric_table(rows: list[dict], columns: list[str], formats: dict[str, str] | None = None) -> list[str]:
    if not rows:
        return ["未找到对应结果文件。"]
    formats = formats or {}
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---:" if col not in {"resolution", "dataset"} else "---" for col in columns) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if col in formats and value != "":
                value = formats[col].format(float(value))
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def metric_deltas(rows: list[dict], baseline_resolution: str, compare_resolutions: list[str], metrics: list[str]) -> list[dict]:
    keyed = {(row["resolution"], row["dataset"]): row for row in rows}
    datasets = sorted({row["dataset"] for row in rows})
    deltas: list[dict] = []
    for dataset in datasets:
        base = keyed.get((baseline_resolution, dataset))
        if not base:
            continue
        for resolution in compare_resolutions:
            current = keyed.get((resolution, dataset))
            if not current:
                continue
            item = {"resolution": f"{resolution} - {baseline_resolution}", "dataset": dataset}
            for metric in metrics:
                item[metric] = float(current[metric]) - float(base[metric])
            deltas.append(item)
    return deltas


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    comparison_root = args.comparison_root.expanduser().resolve()
    out = comparison_root / args.output_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    track = eval_root / "tracking"
    with (comparison_root / "export_manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    classified: list[dict] = []
    for row in rows:
        dataset = row["dataset"]
        sequence = row["sequence"]
        left_dir = track / args.left_model / dataset / sequence
        right_dir = track / args.right_model / dataset / sequence
        features = {
            "left_summary": read_json(left_dir / "summary.json"),
            "right_summary": read_json(right_dir / "summary.json"),
            "left_errors": read_jsonl_counts(left_dir / "errors.jsonl"),
            "right_errors": read_jsonl_counts(right_dir / "errors.jsonl"),
            "left_first_pred": read_states_first_pred(left_dir / "states.jsonl"),
            "right_first_pred": read_states_first_pred(right_dir / "states.jsonl"),
        }
        category = classify(row, features)
        dest_video = out / category / safe_name(dataset, sequence)
        link_type = place_video(Path(row["video"]), dest_video, args.copy_mode)
        combined_errors = features["left_errors"] + features["right_errors"]
        classified.append(
            {
                **row,
                "category": category,
                "category_title": CATEGORIES[category]["title"],
                "classified_video": str(dest_video.resolve()),
                "link_type": link_type,
                "scenario": seq_scenario(dataset, sequence),
                "platform": platform(sequence),
                "left_fn_lines": features["left_errors"].get("false_negative", 0),
                "left_fp_lines": features["left_errors"].get("false_positive", 0),
                "left_loc_lines": features["left_errors"].get("localization_error", 0),
                "right_fn_lines": features["right_errors"].get("false_negative", 0),
                "right_fp_lines": features["right_errors"].get("false_positive", 0),
                "right_loc_lines": features["right_errors"].get("localization_error", 0),
                "combined_fn_lines": combined_errors.get("false_negative", 0),
                "combined_fp_lines": combined_errors.get("false_positive", 0),
                "combined_loc_lines": combined_errors.get("localization_error", 0),
                "left_first_pred": features["left_first_pred"] or "",
                "right_first_pred": features["right_first_pred"] or "",
            }
        )

    fieldnames = list(classified[0].keys())
    write_manifest(out / "classified_manifest.csv", classified, fieldnames)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for item in classified:
        by_cat[item["category"]].append(item)

    for cat, items in by_cat.items():
        cat_dir = out / cat
        write_manifest(cat_dir / "manifest.csv", items, fieldnames)
        dataset_counts = Counter(i["dataset"] for i in items)
        scenario_counts = Counter(i["scenario"] for i in items)
        platform_counts = Counter(i["platform"] for i in items if i["platform"] != "unknown")
        info = CATEGORIES[cat]
        lines = [
            f"# {info['title']}",
            "",
            f"- 数量: {len(items)}",
            f"- 数据集分布: {dict(dataset_counts)}",
            f"- 场景分布: {dict(scenario_counts)}",
        ]
        if platform_counts:
            lines.append(f"- Hanlue new 平台分布: {dict(platform_counts)}")
        lines += ["", "## 失败原因", info["cause"], "", "## 改进建议"]
        lines += [f"- {tip}" for tip in info["improve"]]
        lines += ["", "## 示例视频"]
        for item in items[:8]:
            lines.append(
                f"- `{Path(item['classified_video']).name}`: dataset={item['dataset']}, "
                f"sequence={item['sequence']}, reason={item['reason']}, recall_delta={float(item['recall_delta']):.3f}"
            )
        (cat_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cat_counts = Counter(i["category"] for i in classified)
    dataset_counts = Counter(i["dataset"] for i in classified)
    reason_counts = Counter(i["reason"] for i in classified)
    scenario_counts = Counter(i["scenario"] for i in classified)
    platform_counts = Counter(i["platform"] for i in classified if i["platform"] != "unknown")
    cat_dataset = {cat: dict(Counter(i["dataset"] for i in items)) for cat, items in sorted(by_cat.items())}
    detection_results = read_csv_rows(eval_root / "detection_results.csv")
    tracking_results = read_csv_rows(eval_root / "tracking_results.csv")
    metric_fmt = {
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "mAP50": "{:.4f}",
        "mAP50-95": "{:.4f}",
        "avg_iou": "{:.4f}",
        "mean_seq_precision": "{:.4f}",
        "mean_seq_recall": "{:.4f}",
        "mean_seq_avg_iou": "{:.4f}",
    }
    delta_fmt = {key: "{:+.4f}" for key in metric_fmt}

    report = [
        "# 960x960 vs 640x640 Tracking Failure Case 分类报告",
        "",
        f"- 源目录: `{comparison_root}`",
        f"- 分类目录: `{out}`",
        f"- 分类方式: 每个视频按主导 failure case 放入一个子文件夹，文件采用 `{args.copy_mode}`；默认不额外复制视频内容，不破坏原目录。",
        f"- 视频总数: {len(classified)}",
        "",
        "## 总览",
        "",
        "| 维度 | 分布 |",
        "| --- | --- |",
        f"| dataset | `{dict(dataset_counts)}` |",
        f"| 原始 reason | `{dict(reason_counts)}` |",
        f"| 场景 | `{dict(scenario_counts)}` |",
        f"| Hanlue new 平台 | `{dict(platform_counts)}` |",
        "",
        "## 相关精度结果",
        "",
        "### Detection",
        "",
        *md_metric_table(
            detection_results,
            ["resolution", "dataset", "precision", "recall", "mAP50", "mAP50-95"],
            metric_fmt,
        ),
        "",
        "### Detection 相对 640x640 的变化",
        "",
        *md_metric_table(
            metric_deltas(detection_results, "640x640", ["960x960", "960x540"], ["precision", "recall", "mAP50", "mAP50-95"]),
            ["resolution", "dataset", "precision", "recall", "mAP50", "mAP50-95"],
            delta_fmt,
        ),
        "",
        "### Detection + Tracking",
        "",
        *md_metric_table(
            tracking_results,
            [
                "resolution",
                "dataset",
                "sequences",
                "frames",
                "precision",
                "recall",
                "avg_iou",
                "mean_seq_precision",
                "mean_seq_recall",
                "mean_seq_avg_iou",
                "failures",
            ],
            metric_fmt,
        ),
        "",
        "### Detection + Tracking 相对 640x640 的变化",
        "",
        *md_metric_table(
            metric_deltas(
                tracking_results,
                "640x640",
                ["960x960", "960x540"],
                ["precision", "recall", "avg_iou", "mean_seq_precision", "mean_seq_recall", "mean_seq_avg_iou"],
            ),
            ["resolution", "dataset", "precision", "recall", "avg_iou", "mean_seq_precision", "mean_seq_recall", "mean_seq_avg_iou"],
            delta_fmt,
        ),
        "",
        "## Failure Case 分类",
        "",
        "| 类别目录 | 数量 | 数据集分布 | 主要含义 |",
        "| --- | ---: | --- | --- |",
    ]
    for cat in sorted(CATEGORIES):
        report.append(f"| `{cat}` | {cat_counts.get(cat, 0)} | `{cat_dataset.get(cat, {})}` | {CATEGORIES[cat]['title']} |")

    report += ["", "## 各类原因与改进建议"]
    for cat in sorted(CATEGORIES):
        items = by_cat.get(cat, [])
        if not items:
            continue
        info = CATEGORIES[cat]
        report += [
            "",
            f"### {cat}: {info['title']}",
            "",
            f"- 数量: {len(items)}",
            f"- 数据集分布: `{cat_dataset.get(cat, {})}`",
            f"- 场景 Top: `{dict(Counter(i['scenario'] for i in items).most_common(6))}`",
            "",
            "失败原因:",
            f"- {info['cause']}",
            "",
            "改进建议:",
        ]
        report += [f"- {tip}" for tip in info["improve"]]
        report += ["", "代表样例:"]
        examples = sorted(
            items,
            key=lambda x: (max(float(x["left_errors"]), float(x["right_errors"])), abs(float(x["recall_delta"]))),
            reverse=True,
        )[:5]
        for item in examples:
            report.append(
                f"- `{Path(item['classified_video']).name}` ({item['dataset']}/{item['sequence']}), "
                f"reason={item['reason']}, 960 R={float(item['left_recall']):.3f}, 640 R={float(item['right_recall']):.3f}"
            )

    report += [
        "",
        "## 文件说明",
        "",
        "- `classified_manifest.csv`: 全量分类索引，包含原视频、分类后视频、错误类型计数、首个预测帧等。",
        "- 每个类别子目录下的 `manifest.csv`: 该类失败的视频清单。",
        "- 每个类别子目录下的 `README.md`: 该类原因和改进建议。",
        "",
        "## 使用建议",
        "",
        "- 优先人工看 `01_960x960_covers_640x640_misses` 和 `02_640x640_covers_960x960_misses`，这两类最适合判断分辨率/尺度策略。",
        "- 训练改进优先看 `04/07`，它们通常对应 detector 初始捕获和重捕获能力。",
        "- verifier / 轨迹终止策略优先看 `06`。",
        "- bbox 回归和 tracker 校正优先看 `05`。",
    ]
    (out / "failure_case_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"classified {len(classified)} videos into {out}")
    print("category_counts", dict(cat_counts))
    print("dataset_counts", dict(dataset_counts))


if __name__ == "__main__":
    main()
