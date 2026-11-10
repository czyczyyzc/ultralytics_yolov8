#!/usr/bin/env python3
"""Replay a prior fixed-wing cohort with current frozen labels and a pinned Student."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.evaluate_imported_detector_videos import dump, sha256
from scripts.anti_uav.evaluate_detector_clip_csv import match_frame
from scripts.anti_uav.render_pt_detector_video import probe


def decoded_hashes(path):
    output = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                                      "-an", "-f", "framemd5", "-"], text=True)
    return [line.rsplit(",", 1)[-1].strip() for line in output.splitlines() if line and not line.startswith("#")]


def restore_video(item, source, output):
    target = output / (item["key"] + "_restored.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(source), "-map", "0:v:0",
                    "-an", "-c:v", "copy", "-movflags", "+faststart", str(target)], check=True)
    before, after = decoded_hashes(source), decoded_hashes(target)
    if not before or before != after:
        raise ValueError("Restored source changed decoded frames")
    if len(after) != item["frame_count"] and item["annotations"]:
        raise ValueError("Restored source cannot reuse labels from a different frame count")
    return target, dict(original_ts=str(source), original_ts_sha256=sha256(source),
                        remux_sha256=sha256(target), decoded_frames=len(after), decoded_frames_bit_exact=True,
                        reference_frame_count=item["frame_count"], timing_policy="uniform-average",
                        missing_reference_mp4_sha256=item["source_sha256"])


def snapshot(cohort, database, training_hashes):
    con = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("BEGIN")
    try:
        for item in cohort:
            rows = con.execute("""SELECT t.id,t.status,t.revision,t.approved_revision,t.is_published
                FROM tasks t JOIN videos v ON v.id=t.video_id WHERE v.source_sha256=?
                ORDER BY t.is_published DESC,t.updated_at DESC""", (item["source_sha256"],)).fetchall()
            if len([r for r in rows if r["is_published"]]) > 1:
                raise ValueError("Multiple published tasks for the same source")
            row = dict(rows[0]) if rows else dict(id=None, status="no_current_task", revision=None,
                                                 approved_revision=None, is_published=0)
            item.update(task_id=row["id"], task_status=row["status"], revision=row["revision"],
                        approved_revision=row["approved_revision"], is_published=row["is_published"],
                        in_training_source=item["source_sha256"] in training_hashes,
                        annotations=[dict(r) for r in con.execute("""SELECT frame_index,object_id,status,
                        bbox_x1,bbox_y1,bbox_x2,bbox_y2 FROM annotations WHERE task_id=?
                        ORDER BY frame_index,object_id""", (row["id"],))] if row["id"] else [])
        con.rollback()
    finally:
        con.close()


def frame_ground_truth(records, width, height):
    if not records:
        return None, "unlabelled"
    if any(row["status"] == "uncertain" for row in records):
        return None, "uncertain"
    if any(row["status"] not in ("confirmed", "absent") for row in records):
        return None, "invalid"
    boxes = []
    for row in records:
        if row["status"] != "confirmed":
            continue
        values = [row[k] for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        if any(v is None or not np.isfinite(v) for v in values):
            return None, "invalid"
        x1, y1, x2, y2 = values
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            return None, "invalid"
        boxes.append(values)
    return boxes, "positive" if boxes else "absent"


def ap_metrics(arrays):
    from ultralytics.utils.metrics import ap_per_class
    if not len(arrays["target_cls"]):
        return dict(map50=None, map50_95=None)
    if not len(arrays["conf"]):
        return dict(map50=0., map50_95=0.)
    ap = ap_per_class(**arrays, plot=False)[5]
    return dict(map50=float(ap[:, 0].mean()), map50_95=float(ap.mean()))


def score_cache(item, path, *, calculate_ap):
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou

    labels = defaultdict(list)
    for row in item["annotations"]:
        if not 0 <= row["frame_index"] < item["frame_count"]:
            raise ValueError("Annotation frame outside source")
        labels[row["frame_index"]].append(row)
    eligible = bool(item["is_published"] and item["task_status"] == "approved"
                    and item["revision"] == item["approved_revision"])
    totals = dict(tp=0, fp=0, fn=0, frames=0, gt_boxes=0)
    ignored = dict(unlabelled=0, uncertain=0, invalid=0, unreviewed=0)
    arrays = {k: [] for k in ("tp", "conf", "pred_cls", "target_cls")}
    state = SimpleNamespace(iouv=torch.linspace(.5, .95, 10))
    count = coverage = 0
    for line in path.open():
        record = json.loads(line)
        if record["frame_index"] != count:
            raise ValueError("Nonsequential or incomplete predictions")
        count += 1
        predictions = np.asarray(record["boxes_xyxy_score"], dtype=np.float32).reshape(-1, 5)
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite predictions")
        visible = predictions[predictions[:, 4] >= .03]
        coverage += bool(len(visible))
        if not eligible:
            ignored["unreviewed"] += 1
            continue
        gt, kind = frame_ground_truth(labels[count-1], item["frame_width"], item["frame_height"])
        if gt is None:
            ignored[kind] += 1
            continue
        tp, fp, fn = match_frame(gt, [dict(box=row[:4], confidence=row[4]) for row in visible], .5)
        for key, value in dict(tp=tp, fp=fp, fn=fn, frames=1, gt_boxes=len(gt)).items():
            totals[key] += value
        if calculate_ap:
            boxes = torch.as_tensor(predictions[:, :4])
            gt_boxes = torch.as_tensor(gt, dtype=torch.float32).reshape(-1, 4)
            correct = BaseValidator.match_predictions(state, torch.zeros(len(boxes)), torch.zeros(len(gt)),
                                                      box_iou(gt_boxes, boxes)).numpy()
            for key, value in dict(tp=correct, conf=predictions[:, 4], pred_cls=np.zeros(len(boxes)),
                                   target_cls=np.zeros(len(gt))).items():
                arrays[key].append(value)
    if count != item["frame_count"]:
        raise ValueError("Prediction/source frame coverage differs")
    totals.update(recall=totals["tp"]/totals["gt_boxes"] if totals["gt_boxes"] else None,
                  precision=totals["tp"]/(totals["tp"]+totals["fp"]) if totals["tp"]+totals["fp"] else 0.)
    result = dict(metrics=totals, ignored_frames=ignored, reviewed_labels=eligible,
                  frames=count, frames_with_predictions=coverage,
                  warning="Frames with predictions is not recall; no GT metrics for unreviewed/missing tasks")
    packed = {k: np.concatenate(v) if v else np.empty((0, 10) if k == "tp" else (0,)) for k, v in arrays.items()}
    if calculate_ap:
        result["metrics"].update(ap_metrics(packed))
    return result, packed


def pooled(results, arrays=None):
    output = {k: sum(r["metrics"][k] for r in results) for k in ("tp", "fp", "fn", "frames", "gt_boxes")}
    output.update(recall=output["tp"]/output["gt_boxes"] if output["gt_boxes"] else None,
                  precision=output["tp"]/(output["tp"]+output["fp"]) if output["tp"]+output["fp"] else 0.)
    if arrays:
        output.update(ap_metrics({k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}))
    return output


def write_report(output, protocol, results, aggregate):
    fmt = lambda value: "—" if value is None else f"{value:.2%}"
    lines = ["# 最新 P3 Student 自采固定翼检测复测", "",
        f"模型：74视频修复版 P3 Student FP32；SHA256 `{protocol['model_sha256']}`。",
        "输入为灰度三通道960×544，显示conf=0.03、NMS IoU=0.45、max_det=100；AP预测下限0.001。",
        "全部原视频逐帧推理，不加Tracker、不画GT/十字；1px预测四角框，放大窗从干净原图裁剪后再画框。",
        "这是47服务器GPU上的FP32结果，不是RKNN INT8或板端速度测试。",
        "", "## 评测范围", "",
        f"共{len(results)}段、{sum(r['new']['frames'] for r in results):,}帧。来源与训练重叠的影片仅作适配评测，不是独立泛化验证。",
        "GT采用本次只读数据库冻结的审核版本；uncertain、未标注、无效框帧和未审核任务均不计入精度。",
        "旧73版缓存按同一份当前GT重新计数，不沿用过去标签版本的指标；旧缓存只保留conf≥0.03，因此不计算旧版mAP。",
        "原先已删除的尾随1/00000从本机原始TS无重编码remux恢复，逐帧MD5验证一致；当前无审核任务，仅提供可视化。",
        "该TS实际1796帧，不沿用旧平台转码版1800帧、不人为补帧；可视化按平均帧率均匀播放，保留全部画面与总时长，但不保留原始VFR逐帧时间戳。",
        "", "## 合并指标", "", "| 模型 | Recall | Precision | TP | FP | FN | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, metrics in aggregate.items():
        lines.append(f"| {name} | {fmt(metrics['recall'])} | {fmt(metrics['precision'])} | {metrics['tp']} | "
                     f"{metrics['fp']} | {metrics['fn']} | {fmt(metrics.get('map50'))} | {fmt(metrics.get('map50_95'))} |")
    lines += ["", "AP从所有合格帧预测合并后全局重算，非逐视频AP平均。Precision依赖现有审核标签完整性；发现漏标须复核，不把有框帧数当Recall。",
              "", "## 分段结果与完整视频", "",
              "| 原视频 | 旧73 Recall | 最新 Recall | 最新 Precision | mAP50 | 训练来源 | 完整视频 |",
              "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for result in results:
        m = result["new"]["metrics"]
        old = result.get("baseline")
        movie = result["video"]
        local = f"/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/{output.name}/{result['key']}/{Path(movie).name}"
        lines.append(f"| {result['display_path']} | {fmt(old['metrics']['recall'] if old else None)} | {fmt(m['recall'])} | "
                     f"{fmt(m['precision'] if m['gt_boxes'] else None)} | {fmt(m.get('map50'))} | "
                     f"{'是（不代表每帧都入训）' if result['in_training_source'] else '否'} | [播放]({local}) |")
    lines += ["", "逐帧预测、标签快照、版本/源视频哈希与完整覆盖校验记录见manifest.json、results.json及各clip目录。",
              "恢复文件与原已删除MP4的容器哈希不同，不能声称与历史MP4字节一致；restoration记录的是TS与新MP4解码帧一致。"]
    (output / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--extra-protocol", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--restore", action="append", default=[], help="Missing clip key=/path/to/original.ts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.model) != args.model_sha256:
        raise ValueError("Need a fresh output and the pinned latest Student")
    args.output.mkdir(parents=True)
    sources = args.output / "sources"
    sources.mkdir()
    reference = json.loads(args.reference_manifest.read_text())
    items = reference["videos"]
    for item in items:
        item["baseline_cache"] = str(args.reference_manifest.parent / item["key"] / "predictions.jsonl")
    if args.extra_protocol:
        extra = json.loads(args.extra_protocol.read_text())
        items.append(dict(key="clip_11_mixed_00002", video=extra["source_video"], source_sha256=extra["source_sha256"],
            frame_count=int(extra["source_info"]["nb_frames"]), frame_width=extra["source_info"]["width"],
            frame_height=extra["source_info"]["height"], fps=extra["output_fps"],
            display_path="10.28 15时50 晴天 混合 2/00002.ts",
            baseline_cache=str(args.extra_protocol.parent / "predictions.jsonl")))
    if len({x["source_sha256"] for x in items}) != len(items):
        raise ValueError("Duplicate source videos")
    manifest = json.loads(args.training_manifest.read_text())
    snapshot(items, args.database, set(manifest["train_video_hashes"]))
    restores = dict(value.split("=", 1) for value in args.restore)
    for item in items:
        if not Path(item["video"]).is_file():
            if item["key"] not in restores:
                raise FileNotFoundError(item["video"])
            target, proof = restore_video(item, Path(restores[item["key"]]), sources)
            info = probe(target)
            if (info["width"], info["height"]) != (item["frame_width"], item["frame_height"]):
                raise ValueError("Restored source geometry differs from the reference")
            numerator, denominator = map(int, info["avg_frame_rate"].split("/"))
            item.update(video=str(target), restoration=proof, frame_count=proof["decoded_frames"],
                        fps=numerator/denominator)
        elif sha256(Path(item["video"])) != item["source_sha256"]:
            raise ValueError("Video bytes changed since the reference cohort")
    protocol = dict(created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), model=str(args.model),
        model_sha256=args.model_sha256, reference_manifest=str(args.reference_manifest),
        reference_manifest_sha256=sha256(args.reference_manifest), source_database=str(args.database),
        training_manifest=str(args.training_manifest), training_manifest_sha256=sha256(args.training_manifest),
        input_hw=[544, 960], input_color="gray3", precision="FP32", tracker=False, conf=.03, cache_floor=.001,
        nms_iou=.45, matching_iou=.5, max_det=100, full_videos=True, frame_stride=1, device=args.device,
        source_overlap_is_not_independent_test=True, videos=items)
    dump(args.output / "manifest.json", protocol)
    results, new_arrays = [], []
    try:
        for item in items:
            directory = args.output / item["key"]
            dump(args.output / "status.json", dict(stage="rendering", clip=item["key"], complete=len(results), total=len(items)))
            command = [sys.executable, "-u", str(ROOT / "scripts/anti_uav/render_pt_detector_video.py"),
                "--source-video", item["video"], "--model", str(args.model), "--output", str(directory),
                "--output-name", item["key"] + "_P3_repaired74_FP32_conf003.mp4", "--label", "P3 repaired74 | " + item["key"],
                "--device", args.device, "--batch", str(args.batch), "--conf", ".03", "--cache-conf", ".001", "--input-color", "gray3"]
            if item.get("restoration"):
                command.extend(["--timing-policy", "uniform-average"])
            with (args.output / (item["key"] + ".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            rendered = json.loads((directory / "summary.json").read_text())
            expected_hash = item.get("restoration", {}).get("remux_sha256", item["source_sha256"])
            if rendered["weights_sha256"] != args.model_sha256 or rendered["source_sha256"] != expected_hash:
                raise ValueError("Rendered source/model hash mismatch")
            new, arrays = score_cache(item, directory / "predictions.jsonl", calculate_ap=True)
            baseline = None
            old_cache = Path(item["baseline_cache"])
            if (old_cache.is_file() and rendered["source_non_gray_frames"] == 0
                    and not item.get("restoration")):
                old_summary = json.loads((old_cache.parent / "summary.json").read_text())
                if (old_summary["conf"] != .03 or old_summary["nms_iou"] != .45
                        or old_summary["input_hw"] != [544, 960] or old_summary["frames"] != item["frame_count"]
                        or old_summary["source_sha256"] != item["source_sha256"]
                        or old_summary["weights_sha256"] != reference["model_sha256"]):
                    raise ValueError("Baseline cache protocol differs")
                baseline = score_cache(item, old_cache, calculate_ap=False)[0]
            result = dict(key=item["key"], display_path=item["display_path"], in_training_source=item["in_training_source"],
                          annotation_revision=item["revision"], task_status=item["task_status"], new=new,
                          baseline=baseline, video=rendered["video"], video_sha256=rendered["output_sha256"])
            dump(directory / "evaluation.json", result)
            np.savez_compressed(directory / "pr_arrays.npz", **arrays)
            new_arrays.append(arrays)
            results.append(result)
            dump(args.output / "results.json", dict(videos=results))
        aggregate = {"最新74视频修复版 P3 Student": pooled([r["new"] for r in results], new_arrays)}
        if all(r["baseline"] is not None for r in results if r["new"]["reviewed_labels"]):
            aggregate = {"旧73视频 P3 Student（当前GT重评）": pooled([r["baseline"] for r in results if r["baseline"]]), **aggregate}
        if sha256(args.model) != args.model_sha256:
            raise ValueError("Checkpoint changed during evaluation")
        dump(args.output / "results.json", dict(videos=results, pooled=aggregate,
                                                ap_is_global=True, independent_test=False))
        write_report(args.output, protocol, results, aggregate)
        dump(args.output / "status.json", dict(stage="complete", videos=len(results), frames=sum(r["new"]["frames"] for r in results)))
    except BaseException as error:
        dump(args.output / "status.json", dict(stage="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
