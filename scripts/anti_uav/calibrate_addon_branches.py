#!/usr/bin/env python3
"""Cache raw P2/P3 candidates and calibrate thresholds on validation only."""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch
import torchvision
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator, scale_masks
from scripts.anti_uav.run_gray_probability_ablation import sha256, write_json
from scripts.anti_uav.run_native_pool_comparison import clean
from scripts.anti_uav.train_frozen_p3_addon_p2 import verify_legacy_outputs
from ultralytics import YOLO
from ultralytics.utils import SETTINGS, ops
from ultralytics.utils.metrics import box_iou


def matches_at_half(gt, boxes):
    """Use the existing validator's greedy one-to-one IoU policy, including ties."""
    iou = box_iou(gt, boxes).numpy()
    pairs = np.array(np.nonzero(iou >= .5)).T
    if len(pairs) > 1:
        pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs


def branch_nms(raw, legacy_threshold, p2_threshold, iou=.45, max_det=100):
    """Rows are input-space xyxy/conf/branch, with 0=legacy and 1=P2."""
    threshold = torch.where(raw[:, 5] == 0, legacy_threshold, p2_threshold)
    selected = raw[raw[:, 4] >= threshold]
    if len(selected) > 30000:
        selected = selected[selected[:, 4].argsort(descending=True)[:30000]]
    keep = torchvision.ops.nms(selected[:, :4], selected[:, 4], iou)[:max_det]
    return selected[keep].clone()


def native_boxes(record, detections):
    result = detections.clone()
    ops.scale_boxes((544, 960), result[:, :4], record["ori_shape"], ratio_pad=record["ratio_pad"])
    return result


def add_counts(counter, gt, detections, masks):
    pairs = matches_at_half(gt, detections[:, :4])
    matched = set(pairs[:, 1].tolist())
    counter.update(frames=1, gt=len(gt), tp=len(pairs), fp=len(detections)-len(pairs), fn=len(gt)-len(pairs))
    for size, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        counter[f"{size}_gt"] += int(mask.sum())
        counter[f"{size}_tp"] += int(mask[pairs[:, 0]].sum())
    for i, row in enumerate(detections):
        branch = "legacy" if row[5] == 0 else "p2"
        counter[f'{"tp" if i in matched else "fp"}_{branch}'] += 1
    if len(gt) == 0:
        counter["negative_frames"] += 1
        counter["negative_fp_frames"] += int(len(detections) > 0)
    return pairs


def metrics(counter):
    value = dict(counter)
    value.update(precision=counter["tp"] / max(counter["tp"] + counter["fp"], 1),
                 recall=counter["tp"] / max(counter["gt"], 1),
                 fp1000=1000 * counter["fp"] / max(counter["frames"], 1),
                 tiny_recall=counter["long_4to8px_tp"] / max(counter["long_4to8px_gt"], 1))
    return value


def select_recall_safe(rows, baseline):
    # No aggregate recall loss and no loss in any populated scale bucket.
    keys = [key for key in baseline if key.endswith("_gt") and baseline[key] > 0]
    feasible = [r for r in rows if r["tp"] >= baseline["tp"]
                and all(r.get(key[:-3] + "_tp", 0) >= baseline.get(key[:-3] + "_tp", 0) for key in keys)]
    if not feasible:
        raise ValueError("Baseline must be included as a feasible calibration policy")
    return min(feasible, key=lambda r: (r["fp"], -r["tp"],
               abs(r["legacy_threshold"]-.03) + abs(r["p2_threshold"]-.03)))


class BranchCacheValidator(FixedShapeGrayValidator):
    """Keep the established validation loader, transforms, labels and NMS unchanged."""

    def __init__(self, *args, raw_stream, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_stream = raw_stream

    def postprocess(self, preds):
        if not isinstance(preds, (tuple, list)) or len(preds) != 2:
            raise ValueError("Require a PT model with decoded predictions and raw head maps")
        prediction, levels = preds
        if len(levels) != 4 or prediction.shape[1] != 5:
            raise ValueError("Require the single-class four-scale add-on detector")
        count = levels[0].shape[2] * levels[0].shape[3]
        self.pending = []
        for pred in prediction:
            selected = pred[4] > self.args.conf
            indices = torch.arange(pred.shape[1], device=pred.device)[selected]
            raw = torch.cat((ops.xywh2xyxy(pred[:4, selected].T), pred[4, selected, None],
                             (indices < count).float()[:, None]), dim=1)
            self.pending.append(raw.cpu())
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for i, (raw, pred) in enumerate(zip(self.pending, preds)):
            path = batch["im_file"][i]
            if "/zoom_val/" in path:
                continue
            prepared = self._prepare_batch(i, batch)
            if torch.any(prepared["cls"] != 0):
                raise ValueError("Only single-class drone validation is supported")
            gt = prepared["bbox"].cpu()
            record = dict(path=path, ori_shape=prepared["ori_shape"], ratio_pad=prepared["ratio_pad"],
                          gt=gt.tolist(), raw=raw.tolist(),
                          original_nms=self._prepare_pred(pred, prepared).cpu().tolist(),
                          masks={key: value.tolist() for key, value in scale_masks(
                              gt.numpy(), prepared["ori_shape"], (544, 960)).items()})
            self.raw_stream.write(json.dumps(record) + "\n")


def sweep(records, thresholds, output):
    policies = [(float(x), float(y)) for x in thresholds for y in thresholds]
    counters = {pair: Counter() for pair in policies}
    diagnostics, examples = Counter(), []
    for index, record in enumerate(records):
        raw = torch.tensor(record["raw"], dtype=torch.float32).reshape(-1, 6)
        gt = torch.tensor(record["gt"], dtype=torch.float32).reshape(-1, 4)
        original = torch.tensor(record["original_nms"], dtype=torch.float32).reshape(-1, 6)
        original = original[original[:, 4] >= .03]
        baseline = native_boxes(record, branch_nms(raw, .03, .03))
        if baseline.shape != original.shape or not torch.allclose(baseline[:, :5], original[:, :5], atol=1e-4, rtol=0):
            raise ValueError(f"CPU cache replay differs from original NMS: {record['path']}")
        base_matches = matches_at_half(gt, baseline[:, :4])
        legacy = native_boxes(record, branch_nms(raw, .03, 2.))
        p2 = native_boxes(record, branch_nms(raw, 2., .03))
        lm, pm = matches_at_half(gt, legacy[:, :4]), matches_at_half(gt, p2[:, :4])
        legacy_gt, fused_gt = set(lm[:, 0]), set(base_matches[:, 0])
        lost = legacy_gt - fused_gt
        rescued = fused_gt - legacy_gt
        diagnostics.update(legacy_only_tp=len(lm), legacy_only_fp=len(legacy)-len(lm),
                           p2_only_tp=len(pm), p2_only_fp=len(p2)-len(pm),
                           legacy_gt_lost_after_fusion=len(lost), new_gt_after_fusion=len(rescued))
        if lost or (len(baseline) > len(base_matches)):
            examples.append(dict(path=record["path"], gt=record["gt"],
                                 detections=baseline.tolist(), matched=base_matches.tolist(),
                                 lost_legacy_gt=sorted(int(i) for i in lost), fp=len(baseline)-len(base_matches)))
        for pair, counter in counters.items():
            detections = baseline if pair == (.03, .03) else native_boxes(record, branch_nms(raw, *pair))
            add_counts(counter, gt, detections, record["masks"])
        if (index+1) % 200 == 0:
            print(json.dumps(dict(stage="sweep", frames=index+1, total=len(records))), flush=True)
    rows = [dict(legacy_threshold=pair[0], p2_threshold=pair[1], **metrics(c)) for pair, c in counters.items()]
    baseline = next(r for r in rows if (r["legacy_threshold"], r["p2_threshold"]) == (.03, .03))
    selected = select_recall_safe(rows, baseline)
    budget_safe = [r for r in rows if r["fp"] <= baseline["fp"]]
    max_recall = min(budget_safe, key=lambda r: (-r["tp"], -r["tiny_recall"], r["fp"]))
    report = dict(scope="Validation calibration, not independent test or RKNN/tracker results",
                  protocol=dict(input_hw=[544, 960], nms_iou=.45, matching_iou=.5, max_det=100,
                                cache_replay_equal_to_original_nms=True, area_filter=False,
                                unchanged_scores=True, thresholds=thresholds),
                  baseline=baseline, selected_recall_safe=selected, best_at_same_fp_budget=max_recall,
                  diagnostics=dict(diagnostics), policies=rows,
                  selection_rule="Minimum FP with no TP loss overall or in any populated size bucket",
                  selected_on_validation=True, test_evaluated=False)
    write_json(output / "calibration.json", report)
    examples.sort(key=lambda r: (-bool(r["lost_legacy_gt"]), -r["fp"]))
    write_json(output / "error_examples.json", examples[:100])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference-p3", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="2")
    p.add_argument("--thresholds", type=float, nargs="+", default=[.01, .02, .03, .05, .075, .10, .15, .25])
    a = p.parse_args()
    if .03 not in a.thresholds or any(not .001 < t <= 1 for t in a.thresholds):
        raise ValueError("Include baseline .03; thresholds must exceed the .001 cache floor")
    a.thresholds = sorted(set(a.thresholds))
    config = yaml.safe_load(a.data.read_text())
    val_file = Path(config["val"])
    images = [s for s in val_file.read_text().splitlines() if s.strip()]
    if (any("Video00004" in s for s in images) or len(images) != len(set(images))
            or not images or any("gray_val" not in Path(s).parts and "zoom_val" not in Path(s).parts for s in images)):
        raise ValueError("Use only the approved gray validation list, never the test video")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                         mlflow=False, neptune=False, raytune=False))
    os.environ["WANDB_MODE"] = "disabled"
    model, reference = YOLO(str(a.model)), YOLO(str(a.reference_p3))
    frozen = verify_legacy_outputs(reference.model, model.model)
    write_json(a.output / "frozen_check.json", frozen)
    del reference
    metadata = dict(model=str(a.model.resolve()), model_sha256=sha256(a.model),
                    data=str(a.data.resolve()), val_sha256=sha256(val_file), val_frames=len(images),
                    reference_p3=str(a.reference_p3.resolve()), reference_p3_sha256=sha256(a.reference_p3),
                    python_executable=sys.executable, python_version=platform.python_version(),
                    torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                    input_hw=[544, 960], baseline_threshold=.03, thresholds=a.thresholds,
                    test_evaluated=False, output_contract="No deployment model or configuration is modified")
    write_json(a.output / "protocol.json", metadata)
    cache = a.output / "raw_candidates.jsonl.gz"
    with gzip.open(cache, "wt") as stream:
        def factory(**kwargs):
            return BranchCacheValidator(raw_stream=stream, **kwargs)
        result = model.val(data=str(a.data), validator=factory, imgsz=[544, 960], rect=False,
            device=a.device, batch=32, workers=4, conf=.001, iou=.45, max_det=100,
            half=False, plots=False, verbose=False, project=str(a.output), name="validation")
    write_json(a.output / "original_validation.json", clean(result.gray_selection))
    with gzip.open(cache, "rt") as stream:
        records = [json.loads(line) for line in stream]
    expected = {s for s in images if "/zoom_val/" not in s}
    if len(records) != len(expected) or {r["path"] for r in records} != expected:
        raise ValueError("Cached validation coverage is incomplete")
    report = sweep(records, a.thresholds, a.output)
    for key, metric in (("tp", "TP"), ("fp", "FP"), ("fn", "FN"), ("frames", "FRAMES")):
        if report["baseline"][key] != result.gray_selection[f"native/c0.03/{metric}"]:
            raise ValueError(f"Cached replay metric differs from original validator: {key}")
    lines = ["# Add-on branch calibration", "", report["scope"], "",
             "Thresholds are selected on Video00009 validation only. No independent-test improvement is claimed.", "",
             "| Policy | P3 threshold | P2 threshold | P | R | FP | FN | 4-8px R |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in ("baseline", "selected_recall_safe", "best_at_same_fp_budget"):
        r = report[key]
        lines.append(f"| {key} | {r['legacy_threshold']} | {r['p2_threshold']} | {r['precision']:.2%} | "
                     f"{r['recall']:.2%} | {r['fp']} | {r['fn']} | {r['tiny_recall']:.2%} |")
    (a.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(a.output / "status.json", dict(stage="complete", frames=len(records),
                                              raw_cache_sha256=sha256(cache), baseline_reproduced=True))
    print(json.dumps({k: report[k] for k in ("baseline", "selected_recall_safe", "diagnostics")}), flush=True)


if __name__ == "__main__":
    main()
