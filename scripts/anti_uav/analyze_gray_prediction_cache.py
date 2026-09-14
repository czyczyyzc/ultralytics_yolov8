#!/usr/bin/env python3
"""Localize held-out errors; post-hoc subsets are diagnostic, never replacement test scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.anti_uav.evaluate_real_gray_yolo_lovo_fold import iou_one_to_many
from ultralytics.utils.metrics import ap_per_class


def diagnostic_ap(rows, first_frame):
    correct, confidences, ngt = [], [], 0
    for row in rows:
        if row["frame"] < first_frame:
            continue
        boxes = np.asarray(row["boxes"]).reshape(-1, 6)
        tp = np.zeros((len(boxes), 10), dtype=bool)
        if row["gt"] is not None:
            ngt += 1
            if len(boxes):
                overlap = iou_one_to_many(np.asarray(row["gt"]), boxes[:, :4])
                best = np.argmax(overlap)
                tp[best] = overlap[best] >= np.linspace(.5, .95, 10)
        correct.append(tp)
        confidences.extend(boxes[:, 4])
    if not confidences or not ngt:
        return dict(ground_truth=ngt, ap50=0., ap50_95=0.)
    metrics = ap_per_class(np.concatenate(correct), np.asarray(confidences),
                           np.zeros(len(confidences)), np.zeros(ngt), plot=False)
    return dict(ground_truth=ngt, ap50=float(metrics[5][:, 0].mean()), ap50_95=float(metrics[5].mean()))


def counters(rows, confidence, first_frame=0, stop_frame=None):
    tp = fp = fn = 0
    branches = {"legacy": 0, "P2": 0}
    for row in rows:
        if row["frame"] < first_frame or (stop_frame is not None and row["frame"] >= stop_frame):
            continue
        boxes = np.asarray(row["boxes"]).reshape(-1, 6)
        boxes = boxes[boxes[:, 4] >= confidence]
        match = -1
        if row["gt"] is not None:
            overlaps = iou_one_to_many(np.asarray(row["gt"]), boxes[:, :4]) if len(boxes) else []
            match = int(np.argmax(overlaps)) if len(overlaps) and max(overlaps) >= .5 else -1
            tp += int(match >= 0)
            fn += int(match < 0)
        for i, box in enumerate(boxes):
            if i != match:
                fp += 1
                branches["P2" if box[5] else "legacy"] += 1
    return dict(tp=tp, fp=fp, fn=fn, fp_by_branch=branches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--prefix-stop", type=int, default=170)
    args = parser.parse_args()
    all_rows, findings = {}, {}
    for path in sorted(args.diagnostics_dir.glob("*_attribution.jsonl")):
        name = path.name.removesuffix("_attribution.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 2359, f"Incomplete cache: {path}"
        all_rows[name] = rows
        findings[name] = dict(
            fixed={str(t): dict(full=counters(rows, t),
                               prefix=counters(rows, t, stop_frame=args.prefix_stop),
                               remaining=counters(rows, t, first_frame=args.prefix_stop)) for t in (.01, .03, .25)},
            diagnostic_ap_all=diagnostic_ap(rows, 0),
            diagnostic_ap_without_prefix=diagnostic_ap(rows, args.prefix_stop),
        )
        examples = []
        for frame in (596, 610):
            row = next(x for x in rows if x["frame"] == frame)
            gt = np.asarray(row["gt"])
            boxes = np.asarray(row["boxes"]).reshape(-1, 6)
            overlap = iou_one_to_many(gt, boxes[:, :4]) if len(boxes) else []
            index = int(np.argmax(overlap)) if len(overlap) else -1
            examples.append(dict(frame=frame, gt_input_wh=((gt[2:]-gt[:2])*.5).tolist(),
                                 best_iou=float(overlap[index]) if index >= 0 else 0.,
                                 confidence=float(boxes[index, 4]) if index >= 0 else 0.))
        findings[name]["miss_examples"] = examples
    output = dict(
        caveat="Post-hoc error localization ONLY. Keep original full-video scores. Do not remove difficult frames from official evaluation or train on held-out frames.",
        ap_protocol="Cached FP32 predictions, conf>=0.01, NMS=0.45, same single-GT greedy IoU matching as validator; not the standard conf=0.001/NMS=0.7 report.",
        excluded_prefix_for_diagnosis=[0, args.prefix_stop-1], results=findings)
    (args.diagnostics_dir / "error_localization.json").write_text(json.dumps(output, indent=2) + "\n")

    panels = []
    for name in ("old_addon_best", "new_addon_best", "new_addon_last"):
        row = next(x for x in all_rows[name] if x["frame"] == 2)
        original = cv2.imread(row["path"])
        assert original is not None
        panel = np.full((540, 640, 3), 24, dtype=np.uint8)
        full = cv2.resize(original, (640, 360), interpolation=cv2.INTER_AREA)
        candidates = [b for b in row["boxes"] if .03 <= b[4] and 330 < (b[0]+b[2])/2 < 550
                      and 210 < (b[1]+b[3])/2 < 380]
        box = max(candidates, key=lambda b: b[4])
        color = (80, 200, 255)
        cv2.rectangle(full, (round(box[0]/3), round(box[1]/3)),
                      (round(box[2]/3), round(box[3]/3)), color, 1, cv2.LINE_8)
        panel[42:402] = full
        crop = cv2.resize(original[210:380, 330:550], (165, 128), interpolation=cv2.INTER_NEAREST)
        panel[407:535, 10:175] = crop
        cv2.putText(panel, name.replace("_", " "), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (245,245,245), 1, cv2.LINE_AA)
        cv2.putText(panel, f"frame 2 | conf {box[4]:.3f}", (188, 443), cv2.FONT_HERSHEY_SIMPLEX, .68, color, 1, cv2.LINE_AA)
        cv2.putText(panel, "GT: absent | blurred texture", (188, 476), cv2.FONT_HERSHEY_SIMPLEX, .62, (245,245,245), 1, cv2.LINE_AA)
        cv2.putText(panel, "Inset: untouched source crop", (188, 510), cv2.FONT_HERSHEY_SIMPLEX, .56, (190,190,190), 1, cv2.LINE_AA)
        panels.append(panel)
    assert cv2.imwrite(str(args.diagnostics_dir/"blurred_background_comparison.png"), np.hstack(panels))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
