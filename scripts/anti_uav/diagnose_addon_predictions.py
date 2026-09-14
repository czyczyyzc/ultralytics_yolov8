#!/usr/bin/env python3
"""Attribute detections to frozen legacy or P2 heads without changing their scores."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.anti_uav.audit_gray_training_regression import bucket, quantiles
from scripts.anti_uav.evaluate_real_gray_yolo_lovo_fold import iou_one_to_many, load_gt
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops


class AttributionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        prediction, levels = preds
        assert len(levels) == 4 and prediction.shape[1] == 5, "Expected single-class four-scale head"
        p2_count = levels[0].shape[2] * levels[0].shape[3]
        tagged = prediction.new_zeros((len(prediction), 6, prediction.shape[-1]))
        tagged[:, :4] = prediction[:, :4]
        tagged[:, 4, p2_count:] = prediction[:, 4, p2_count:]
        tagged[:, 5, :p2_count] = prediction[:, 4, :p2_count]
        # Pseudo-classes are provenance only: agnostic NMS preserves original competition.
        detections = ops.non_max_suppression(tagged, self.args.conf, self.args.iou,
                                             agnostic=True, max_det=self.args.max_det)
        reference = ops.non_max_suppression(prediction.clone(), self.args.conf, self.args.iou,
                                            agnostic=True, max_det=self.args.max_det)
        results = []
        for pred, ref, original, path in zip(detections, reference, orig_imgs, self.batch[0]):
            if pred.shape != ref.shape or not torch.equal(pred[:, :5], ref[:, :5]):
                raise RuntimeError("Attribution changed original detection coordinates/scores/order")
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], original.shape)
            result = Results(original, path=path, names={0: "legacy", 1: "P2"}, boxes=pred)
            result.actual_input_shape = list(img.shape[2:])
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", default="6")
    args = parser.parse_args()
    torch.set_num_threads(4)
    thresholds = (.01, .03, .05, .10, .25, .40, .45)
    counts = {t: Counter() for t in thresholds}
    sizes = {t: defaultdict(Counter) for t in thresholds}
    scores = defaultdict(list)
    misses = {t: [] for t in thresholds}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    predictions = model.predict(source=str(args.image_list), predictor=AttributionPredictor,
                                imgsz=[544, 960], conf=.01, iou=.45, max_det=100,
                                device=args.device, batch=32, stream=True, verbose=False,
                                rect=False, half=False)
    nframes = 0
    with args.output_prefix.with_suffix(".jsonl").open("w") as stream:
        for result in predictions:
            assert result.actual_input_shape == [544, 960]
            h, w = result.orig_shape
            gt = load_gt(Path(result.path), w, h)
            boxes = result.boxes.data.cpu().numpy()
            frame = int(Path(result.path).stem)
            record = dict(frame=frame, path=result.path, shape=[h, w],
                          gt=None if gt is None else gt.tolist(),
                          boxes=boxes.tolist(), columns=["x1", "y1", "x2", "y2", "conf", "branch_0legacy_1P2"])
            stream.write(json.dumps(record) + "\n")
            for threshold in thresholds:
                selected = boxes[boxes[:, 4] >= threshold]
                c = counts[threshold]
                overlaps = iou_one_to_many(gt, selected[:, :4]) if gt is not None and len(selected) else np.zeros(len(selected))
                matched = int(np.argmax(overlaps)) if len(overlaps) and overlaps.max() >= .5 else -1
                if gt is None:
                    c["absent_frames"] += 1
                    c["absent_frames_with_fp"] += int(len(selected) > 0)
                    c["fp_on_absent_frames"] += len(selected)
                else:
                    c["positive_frames"] += 1
                    c["tp"] += int(matched >= 0)
                    c["fn"] += int(matched < 0)
                    edge = max(gt[2]-gt[0], gt[3]-gt[1]) * min(960/w, 544/h)
                    sizes[threshold][bucket(edge)]["positive"] += 1
                    sizes[threshold][bucket(edge)]["tp"] += int(matched >= 0)
                    if matched < 0:
                        misses[threshold].append(frame)
                for i, det in enumerate(selected):
                    branch = "P2" if int(det[5]) else "legacy"
                    status = "tp" if i == matched else "fp"
                    c[f"{status}_{branch}"] += 1
                    if status == "fp":
                        c["fp"] += 1
                    if threshold == .01:
                        scores[f"{status}_{branch}"].append(float(det[4]))
            nframes += 1
    expected = len([x for x in args.image_list.read_text().splitlines() if x.strip()])
    assert nframes == expected
    fixed = {}
    for threshold, c in counts.items():
        fixed[f"{threshold:.2f}"] = dict(c, precision=c["tp"]/max(c["tp"]+c["fp"], 1),
                                        recall=c["tp"]/max(c["tp"]+c["fn"], 1),
                                        size_buckets=dict(sizes[threshold]), misses=misses[threshold])
    summary = dict(model=str(args.model), frames=nframes, actual_input_shape=[544, 960],
                   tagging_identical_to_original_nms=True, fixed=fixed,
                   confidence_quantiles={k: quantiles(v) for k, v in scores.items()},
                   scope="PT FP32, NMS .45, matching IoU .5. Diagnostic only; no model selection.")
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
