#!/usr/bin/env python3
"""Score frozen causal prediction traces; GT is only read after inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.causal_roi_policy import choose_region

ARMS = ["p3_full", "p3_roi2x", "p3_roi4x", "addon_full"]
TITLES = {"p3_full": "P3 full", "p3_roi2x": "P3 ROI 2x",
          "p3_roi4x": "P3 ROI 4x", "addon_full": "Frozen-P3 + Add-on P2"}
THRESHOLDS = [.01, .03, .05, .10, .25, .40]


def iou(a, b):
    if a is None or b is None:
        return 0.
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(area, 1e-12)


def gt_boxes(annotation):
    if len(annotation["exist"]) != len(annotation["gt_rect"]):
        raise ValueError("GT arrays have different lengths")
    boxes = []
    for exists, xywh in zip(annotation["exist"], annotation["gt_rect"]):
        if not exists:
            boxes.append(None)
            continue
        x, y, w, h = map(float, xywh)
        if w <= 0 or h <= 0 or not all(map(math.isfinite, (x,y,w,h))):
            raise ValueError("Invalid visible GT")
        boxes.append([x,y,x+w,y+h])
    return boxes


def score_rows(records, gt, kind, threshold=.01):
    rows = []
    for r, target in zip(records, gt):
        if kind == "tracks":
            candidates = [(t["box"], t["id"]) for t in r["tracks"]]
        else:
            candidates = [(d[:4], None) for d in r["detections"] if d[4] >= threshold]
        best_iou, best_id = max(((iou(box,target), tid) for box,tid in candidates),
                                key=lambda pair: pair[0], default=(0.,None))
        tp = int(target is not None and best_iou >= .5)
        rows.append(dict(frame=r["frame"], tp=tp, fp=len(candidates)-tp,
                         fn=int(target is not None)-tp, visible=target is not None,
                         best_iou=best_iou, id=best_id,
                         absent_output=target is None and bool(candidates)))
    return rows


def metrics(rows):
    tp,fp,fn = (sum(r[k] for r in rows) for k in ("tp","fp","fn"))
    p,rec = tp/max(tp+fp,1), tp/max(tp+fn,1)
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=rec,
                f1=2*p*rec/max(p+rec,1e-12),
                absent_false_output_rate=sum(r["absent_output"] for r in rows)/max(sum(not r["visible"] for r in rows),1),
                mean_matched_iou=sum(r["best_iou"] for r in rows if r["tp"])/max(tp,1))


def events(rows, fps):
    spans = []
    for r in rows:
        if r["visible"]:
            if not spans or spans[-1][-1]["frame"]+1 != r["frame"]:
                spans.append([])
            spans[-1].append(r)
    results = []
    for span in spans:
        hits = [r for r in span if r["tp"]]
        switches = sum(a["id"] != b["id"] for a,b in zip(hits,hits[1:]))
        fragments,seen,previous = 0,False,False
        longest,run = 0,0
        for r in span:
            fragments += int(bool(r["tp"]) and seen and not previous)
            seen = seen or bool(r["tp"])
            previous = bool(r["tp"])
            run = 0 if r["tp"] else run+1
            longest = max(longest,run)
        delay = hits[0]["frame"]-span[0]["frame"] if hits else None
        results.append(dict(start=span[0]["frame"],end=span[-1]["frame"],
                            first_match_delay_frames=delay,
                            first_match_delay_ms=delay*1000/fps if delay is not None else None,
                            id_switches=switches,fragments=fragments,longest_miss_run=longest))
    return results


def validate_traces(manifests, traces, count):
    common = ["source_sha256", "source_size", "source_fps", "input_size_wh",
              "conf", "nms_iou", "max_det", "padding", "tracker", "tracker_source_sha256", "runtime"]
    ref = manifests[ARMS[0]]
    for arm in ARMS:
        for key in common:
            if manifests[arm][key] != ref[key]:
                raise ValueError(f"Protocol mismatch: {arm}/{key}")
        if len(traces[arm]) != count or [r["frame"] for r in traces[arm]] != list(range(count)):
            raise ValueError(f"Frame coverage mismatch: {arm}")
        previous,anchor = [],None
        width,height = manifests[arm]["source_size"]
        for record in traces[arm]:
            region,mode,anchor = choose_region(previous,record["frame"],width,height,
                                               manifests[arm]["zoom"],manifests[arm]["refresh_interval"],anchor)
            if (list(region),mode,anchor) != (record["roi"],record["mode"],record["anchor_id"]):
                raise ValueError(f"Historical-only ROI audit failed at {arm}/{record['frame']}")
            previous = record["tracks"]
    if len({manifests[arm]["model_sha256"] for arm in ARMS[:3]}) != 1:
        raise ValueError("P3 baselines must use identical weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()
    annotation = json.loads(args.ground_truth.read_text())
    gt = gt_boxes(annotation)
    manifests = {a: json.loads((args.root/a/"manifest.json").read_text()) for a in ARMS}
    traces = {a: [json.loads(line) for line in (args.root/a/"frames.jsonl").read_text().splitlines()] for a in ARMS}
    validate_traces(manifests,traces,len(gt))
    ref = manifests[ARMS[0]]
    scale = min(960/ref["source_size"][0],544/ref["source_size"][1])
    sizes = [math.sqrt((b[2]-b[0])*(b[3]-b[1]))*scale if b else None for b in gt]
    buckets = [("lt4",0,4),("4_to_6",4,6),("6_to_8",6,8),("8_to_12",8,12),("ge12",12,float("inf"))]
    summary = dict(schema="anti_uav.causal_roi_scored.v1",frames=len(gt),positive_frames=sum(b is not None for b in gt),
                   absent_frames=sum(b is None for b in gt),iou_threshold=.5,
                   gt_sha256=hashlib.sha256(args.ground_truth.read_bytes()).hexdigest(),
                   size_definition="sqrt(GT width*height) at fixed full-frame 960x544 input scale; same bins for all arms",
                   protocol_note="No new training, no GT/future input to ROI. Fixed 2x primary and 4x sensitivity, refresh=10; test not used to tune either.",
                   causal_audit="Every crop and anchor reconstructed exactly from preceding output tracks; all frames passed",
                   thresholds_note="Detector score sweeps filter a fixed conf=0.01 causal trace; changing tracker feedback thresholds needs a new sequential run. These are not standard mAP.",
                   arms={})
    all_track_rows = {}
    for arm in ARMS:
        track_rows = score_rows(traces[arm],gt,"tracks")
        detector_rows = score_rows(traces[arm],gt,"detections")
        all_track_rows[arm] = track_rows
        event_rows = events(track_rows,ref["source_fps"])
        size_rows = {}
        for name,lo,hi in buckets:
            selected = [i for i,s in enumerate(sizes) if s is not None and lo <= s < hi]
            size_rows[name] = dict(gt_count=len(selected),
                                  track_tp=sum(track_rows[i]["tp"] for i in selected),
                                  track_recall=sum(track_rows[i]["tp"] for i in selected)/len(selected) if selected else None)
            size_rows[name]["detector_tp"] = sum(detector_rows[i]["tp"] for i in selected)
            size_rows[name]["detector_recall"] = sum(detector_rows[i]["tp"] for i in selected)/len(selected) if selected else None
        roi_visible = {i for i,r in enumerate(traces[arm]) if r["mode"] == "roi" and gt[i] is not None}
        inside = {i for i in roi_visible if all((gt[i][0]>=traces[arm][i]["roi"][0],
                  gt[i][1]>=traces[arm][i]["roi"][1],gt[i][2]<=traces[arm][i]["roi"][2],gt[i][3]<=traces[arm][i]["roi"][3]))}
        failures = dict(roi_target_outside=0,roi_detector_miss_inside=0,
                        roi_tracking_miss_after_detection=0,full_detector_miss=0,
                        full_tracking_miss_after_detection=0)
        for i,target in enumerate(gt):
            if target is None or track_rows[i]["tp"]:
                continue
            if i in roi_visible:
                if i not in inside:
                    key = "roi_target_outside"
                elif not detector_rows[i]["tp"]:
                    key = "roi_detector_miss_inside"
                else:
                    key = "roi_tracking_miss_after_detection"
            else:
                key = "full_detector_miss" if not detector_rows[i]["tp"] else "full_tracking_miss_after_detection"
            failures[key] += 1
        summary["arms"][arm] = dict(
            manifest=manifests[arm], tracker=metrics(track_rows), events=event_rows,
            id_switches=sum(e["id_switches"] for e in event_rows),fragments=sum(e["fragments"] for e in event_rows),
            detector_thresholds={f"{t:.2f}":metrics(score_rows(traces[arm],gt,"detections",t)) for t in THRESHOLDS},
            size_buckets=size_rows, roi_visible_frames=len(roi_visible),
            miss_breakdown=failures,
            roi_gt_fully_inside_frames=len(inside),roi_gt_outside_frames=len(roi_visible)-len(inside),
            roi_inside_tracker_recall=sum(track_rows[i]["tp"] for i in inside)/len(inside) if inside else None)
    for arm in ARMS[1:]:
        summary["arms"][arm]["paired_vs_p3_full"] = dict(
            rescued_frames=[i for i in range(len(gt)) if all_track_rows[arm][i]["tp"] and not all_track_rows[ARMS[0]][i]["tp"]],
            regressed_frames=[i for i in range(len(gt)) if all_track_rows[ARMS[0]][i]["tp"] and not all_track_rows[arm][i]["tp"]])
    out = args.root / "evaluation"
    out.mkdir(exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (out/"per_frame.csv").open("w",newline="") as handle:
        rows = [dict(arm=a,mode=traces[a][i]["mode"],size_px=sizes[i],**r) for a in ARMS for i,r in enumerate(all_track_rows[a])]
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Causal P3 ROI vs Frozen-P3 + Add-on P2", "", "Strict Video00004 test; PT FP32 reference, native C++ RK-BoT-SORT.", "",
             "All four arms use the same 2359 frames, IoU 0.50, 960x544 model input, detector conf 0.01, NMS 0.45 and tracker high/low/new 0.03/0.01/0.10, min_hits=3, confirmed outputs only.", "",
             "P3 ROI uses previous confirmed observed tracks only. No track: full search; every 10th frame: full refresh; otherwise one crop inference. Sticky target ID; only one ROI. Two zooms were fixed before evaluation. Each method runs one inference per source frame.", "",
             "## Tracking", "", "| Method | TP | FP | FN | Precision | Recall | F1 | ID switches | Fragments |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        s=summary["arms"][a]; m=s["tracker"]
        lines.append(f"| {TITLES[a]} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | {s['id_switches']} | {s['fragments']} |")
    lines += ["", "## Small-target recall", "", summary["size_definition"], "", "| Full-frame input pixels | GT frames | P3 | P3 ROI 2x | P3 ROI 4x | Add-on P2 |", "|---|---:|---:|---:|---:|---:|"]
    for name,_,_ in buckets:
        count=summary["arms"][ARMS[0]]["size_buckets"][name]["gt_count"]
        values=[summary["arms"][a]["size_buckets"][name]["track_recall"] for a in ARMS]
        lines.append(f"| {name} | {count} | " + " | ".join("N/A" if v is None else f"{v:.2%}" for v in values)+" |")
    lines += ["", "## Causal audit and misses", "", summary["causal_audit"], "",
              "Miss categories partition tracking FN. Outside means the GT is not fully inside the selected ROI; it does not establish motion as the cause. Detector miss includes localization IoU below 0.50.", "",
              "| Method | ROI outside | ROI detector miss inside | ROI track miss after detection | Full detector miss | Full track miss after detection |",
              "|---|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        failures = summary["arms"][a]["miss_breakdown"]
        lines.append(f"| {TITLES[a]} | " + " | ".join(str(v) for v in failures.values()) + " |")
    lines += ["", "There is one visible episode, frames 587-1034 (zero-based). First confirmed IoU-matched output is delayed 7 frames / 70 ms for all P3 arms and 2 frames / 20 ms for Add-on P2. These are source-video intervals, not compute latency. This clip does not test multiple disappearance/reappearance episodes."]
    lines += ["", "## Detector thresholds", "", summary["thresholds_note"], "", "| Conf | Method | TP | FP | FN | Precision | Recall | F1 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for t in THRESHOLDS:
        for a in ARMS:
            m=summary["arms"][a]["detector_thresholds"][f"{t:.2f}"]
            lines.append(f"| {t:.2f} | {TITLES[a]} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} |")
    lines += ["", "## Runtime", "", "Server A100, sequential batch 1 FP32; not RK3588/RKNN INT8 results. Includes decode, crop, detector pre/postprocessing and tracker; excludes serialization/rendering and ten synthetic warmup calls.", "", "| Method | Server FPS | Mean ms | P95 ms | ROI frames | Full frames |", "|---|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        m=manifests[a]
        lines.append(f"| {TITLES[a]} | {m['server_serial_fps']:.2f} | {m['pipeline_mean_ms']:.2f} | {m['pipeline_p95_ms']:.2f} | {m['roi_frames']} | {m['full_frames']} |")
    lines += ["", "## Limits", "", "This is a single held-out, repeatedly examined video, not proof of generalization to unseen scenes. The crop strategy needs a prior acquisition; it does not improve P3's first discovery before an anchor exists. Fixed pixel input means ROI zoom does not reduce NPU computation per call. Dependence on the previous track prevents assuming the earlier three-worker throughput. RKNN INT8 export/accuracy and board speed are not validated here.", ""]
    (args.root/"RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines[:19]),flush=True)


if __name__ == "__main__":
    main()
