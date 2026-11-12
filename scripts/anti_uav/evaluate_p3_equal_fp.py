"""Compare P3 checkpoints at shared false-positive budgets on frozen native holdouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator, scale_masks
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

SIZES = ("long_lt4px", "long_4to8px", "long_8to16px", "long_gt16px")
IOU_THRESHOLDS = (.5, .75)


class CaptureNativeValidator(FixedShapeGrayValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.native_records = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for i, pred in enumerate(preds):
            path = str(batch["im_file"][i])
            if self.image_group(path) != "native":
                continue
            gt = self._prepare_batch(i, batch)
            self.native_records.append(dict(
                path=path, original_hw=list(gt["ori_shape"]), input_hw=list(gt["imgsz"]),
                boxes=gt["bbox"].cpu().tolist(), classes=gt["cls"].cpu().tolist(),
                preds=self._prepare_pred(pred, gt).cpu().tolist()))

    def get_stats(self):
        result = super().get_stats()
        self.metrics.native_records = self.native_records
        return result


def frame_events(frame):
    gt = np.asarray(frame["boxes"], dtype=np.float32).reshape(-1, 4)
    pred = np.asarray(frame["preds"], dtype=np.float32).reshape(-1, 6)
    pred = pred[np.argsort(-pred[:, 4], kind="stable")]
    masks = scale_masks(gt, frame["original_hw"], frame["input_hw"])
    groups = np.zeros(len(gt), dtype=np.int32)
    for i, size in enumerate(SIZES):
        groups[masks[size]] = i
    iou = box_iou(torch.from_numpy(pred[:, :4]), torch.from_numpy(gt)).numpy()
    iou *= pred[:, 5:6] == np.asarray(frame["classes"]).reshape(1, -1)
    matched = np.full((len(pred), len(IOU_THRESHOLDS)), -1, dtype=np.int32)
    match_iou, center_error = np.zeros(len(pred)), np.zeros(len(pred))
    gain = min(i/j for i, j in zip(frame["input_hw"], frame["original_hw"]))
    for t, threshold in enumerate(IOU_THRESHOLDS):
        available = np.ones(len(gt), dtype=bool)
        for i in np.flatnonzero((iou >= threshold).any(axis=1)):
            j = np.where(available, iou[i], -1).argmax()
            if iou[i, j] >= threshold and available[j]:
                matched[i, t] = groups[j]
                available[j] = False
                if t == 0:
                    match_iou[i] = iou[i, j]
                    a, b = pred[i, :4], gt[j]
                    center_error[i] = np.linalg.norm((a[:2]+a[2:] - b[:2]-b[2:])*.5)*gain
    return pred[:, 4], matched, match_iou, center_error, np.bincount(groups, minlength=len(SIZES))


def summarize(records, budgets):
    scores, groups, ious, errors, totals = zip(*(frame_events(frame) for frame in records))
    order = np.argsort(-np.concatenate(scores), kind="stable")
    scores, groups, ious, errors = (np.concatenate(values)[order] for values in (scores, groups, ious, errors))
    totals = np.sum(totals, axis=0)
    # Only cut between distinct scores, so each point has an implementable confidence threshold.
    cutoffs = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1, len(scores)]
    fp_prefix = np.r_[0, np.cumsum(groups[:, 0] == -1)]
    output = {}
    for budget in budgets:
        allowed = int(budget*len(records)//1000)
        n = int(cutoffs[fp_prefix[cutoffs] <= allowed][-1])
        fp = int(fp_prefix[n])
        row = dict(conf_threshold=float(scores[n-1]) if n else None, frames=len(records),
            fp=fp, fp_per_1000=1000*fp/len(records), budget_fp=allowed,
            reached_score_floor=n == len(scores), gt=int(totals.sum()),
            recall50=float((n-fp)/totals.sum()) if totals.sum() else None,
            precision50=float((n-fp)/n) if n else None, sizes={})
        for i, name in enumerate(SIZES):
            selected = groups[:n, 0] == i
            gt = int(totals[i])
            tp50, tp75 = int(selected.sum()), int((groups[:n, 1] == i).sum())
            row["sizes"][name] = dict(gt=gt, tp50=tp50, tp75=tp75, recall50=tp50/gt if gt else None,
                recall75=tp75/gt if gt else None, mean_matched_iou50=float(ious[:n][selected].mean()) if tp50 else None,
                median_matched_center_error_input_px=float(np.median(errors[:n][selected])) if tp50 else None)
        output[str(budget)] = row
    return output


def evaluate(arms, splits, device, output, budgets=(100, 200, 400)):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    result = dict(iou_fp_reference=.5, comparison="same capped FP per 1000 native frames",
        confidence_floor=.0001, nms_iou=.45, max_detections=300, fp32=True,
        diagnostic_only_no_test_threshold_selection=True, splits={}, arms={})
    configs = {}
    for name, data in splits.items():
        config = yaml.safe_load(data.read_text())
        config["path"] = str(Path(config.get("path", data.parent)).resolve())
        config["input_color"] = "gray3"
        configs[name] = output / f"{name}.yaml"
        configs[name].write_text(yaml.safe_dump(config, sort_keys=False))
        result["splits"][name] = dict(config=str(data), sha256=sha256(data))

    gt_identity = {}
    for arm, weights in arms.items():
        model = YOLO(str(weights))
        arm_result = result["arms"][arm] = dict(weights=str(weights), sha256=sha256(weights), splits={})
        pooled = []
        for name, data in configs.items():
            metrics = model.val(data=str(data), validator=CaptureNativeValidator, device=device,
                imgsz=[544, 960], batch=32, workers=8, rect=False, conf=.0001, iou=.45, max_det=300,
                plots=False, half=False, project=str(output/"validation"), name=f"{arm}_{name}")
            records = sorted(metrics.native_records, key=lambda row: row["path"])
            if not records or len(records) != len({row["path"] for row in records}):
                raise ValueError("Empty or duplicated native evaluation split")
            gt = [{k: v for k, v in row.items() if k != "preds"} for row in records]
            if gt_identity.setdefault(name, gt) != gt:
                raise ValueError("Holdout frame or GT changed across arms")
            expected = {"Video00004": (2359, 448), "Video00009": (1421, 797)}[name]
            if (len(records), sum(len(row["boxes"]) for row in records)) != expected:
                raise ValueError(f"Unexpected native coverage for {name}")
            dump(output / f"{arm}_{name}_predictions.json", records)
            arm_result["splits"][name] = dict(matched_fp=summarize(records, budgets),
                mAP50=metrics.gray_selection["native/mAP50"], mAP50_95=metrics.gray_selection["native/mAP50-95"])
            pooled.extend(records)
        arm_result["pooled_matched_fp"] = summarize(pooled, budgets)
    dump(output/"comparison.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("control", "offset", "video00004", "video00009", "output"):
        p.add_argument(f"--{arg}", required=True, type=Path)
    p.add_argument("--device", required=True)
    a = p.parse_args()
    evaluate(dict(control=a.control, offset=a.offset),
             dict(Video00004=a.video00004, Video00009=a.video00009), a.device, a.output)


if __name__ == "__main__":
    main()
