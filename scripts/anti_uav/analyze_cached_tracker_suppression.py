#!/usr/bin/env python3
"""Audit exact detector/track caches and run offline confirmation/birth ablations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.evaluate_detector_clip_csv import box_iou, match_frame
from scripts.anti_uav.run_causal_roi_comparison import NativeTracker


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def require(condition, text):
    if not condition:
        raise ValueError(text)


def metrics(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    precision, recall = tp / max(1, tp+fp), tp / max(1, tp+fn)
    return dict(counts, precision=precision, recall=recall,
                f1=2*precision*recall/max(1e-15, precision+recall))


def evaluate(frames, gt, included, iou):
    totals = Counter(tp=0, fp=0, fn=0, frames_with_output=0, outputs=0)
    per_frame = {}
    for index in sorted(included):
        counts = match_frame(gt[index], frames[index], iou)
        per_frame[index] = counts
        totals.update(dict(zip(("tp", "fp", "fn"), counts)))
        totals["outputs"] += len(frames[index])
        totals["frames_with_output"] += bool(frames[index])
    return metrics(totals), per_frame


def replay(library, detections, fps, size, birth, hits, reference=None):
    tracker = NativeTracker(library, fps)
    tracker.lib.rk_tracker_destroy(tracker.handle)
    tracker.handle = tracker.lib.rk_tracker_create(.03, .01, birth, .92, .92, 1., 0., fps, hits)
    require(bool(tracker.handle), "Cannot create diagnostic tracker")
    outputs = []
    try:
        for index, row in enumerate(detections):
            raw = tracker.update(row["boxes_xyxy_score"], index/fps, *size)
            if reference is not None:
                expected = reference[index]["raw_tracks"]
                require(len(raw) == len(expected), f"Replay track count differs at {index}")
                for a, b in zip(raw, expected):
                    require(all(a[k] == b[k] for k in ("id", "confirmed", "predicted", "age", "hits")),
                            f"Replay identity/state differs at {index}")
                    require(np.allclose(a["box"], b["box"], rtol=0, atol=1e-5)
                            and abs(a["score"]-b["score"]) < 1e-7,
                            f"Replay geometry differs at {index}")
            outputs.append([t for t in raw if t["confirmed"] and not t["predicted"]])
    finally:
        tracker.close()
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--tracker-dir", type=Path, required=True)
    parser.add_argument("--approved-manifest", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ds = json.loads((args.detector_dir / "summary.json").read_text())
    ts = json.loads((args.tracker_dir / "summary.json").read_text())
    meta = json.loads(args.approved_manifest.read_text())
    coco = json.loads(args.coco.read_text())
    pred_path = args.detector_dir / "predictions.jsonl"
    tracks_path = args.tracker_dir / "tracks.jsonl"
    require(digest(pred_path) == ts["predictions_sha256"], "Tracker did not use this detector cache")
    require(ds["weights_sha256"] == ts["weights_sha256"], "Different weights")
    require(ds["source_sha256"] == ts["source_sha256"] == meta["video"]["sha256"], "Different videos")
    require(ds["conf"] == ts["conf"] == .03 and ts["tracker"]["prediction_sec"] == 0., "Unsupported protocol")
    require(digest(args.coco) == next(r["sha256"] for r in meta["files"] if r["path"] == "coco/annotations.json"),
            "Annotation hash mismatch")
    for filename, expected in ts["tracker_source_sha256"].items():
        require(digest(ROOT / filename) == expected, f"Tracker implementation changed: {filename}")
    detections, tracks = read_jsonl(pred_path), read_jsonl(tracks_path)
    count, fps = meta["video"]["frameCount"], meta["video"]["fps"]
    require(len(detections) == len(tracks) == count, "Frame count mismatch")
    for i, (d, t) in enumerate(zip(detections, tracks)):
        require(d["frame_index"] == t["frame_index"] == i, "Frame ordering mismatch")
        require(abs(d["time_seconds"] - i/fps) < 1e-8 and abs(t["time_seconds"] - i/fps) < 1e-8,
                "Timestamp mismatch")
    included = set(meta["frames"]["includedFrameIndices"])
    included -= set(meta["frames"].get("excludedUncertainFrameIndices", []))
    included -= set(meta["frames"].get("excludedUnreviewedFrameIndices", []))
    require(meta["frames"]["indexBase"] == 0, "Expected zero-based frames")
    by_id = {r["id"]: r["frame_index"] for r in coco["images"]}
    require(set(by_id.values()) == included, "COCO coverage differs from reviewed frames")
    gt = defaultdict(list)
    for a in coco["annotations"]:
        require(a["category_id"] == 0 and not a.get("iscrowd"), "Unexpected category/crowd")
        x, y, w, h = a["bbox"]
        gt[by_id[a["image_id"]]].append((x, y, x+w, y+h))
    require(all(len(v) <= 1 for v in gt.values()), "This diagnostic requires at most one GT per frame")
    frames = dict(detector=[[dict(box=b[:4], score=b[4]) for b in d["boxes_xyxy_score"]] for d in detections],
                  tracker_raw=[t["raw_tracks"] for t in tracks],
                  tracker_current=[t["displayed_tracks"] for t in tracks])
    frame_reasons, box_reasons, missed, per_frame = Counter(), Counter(), [], []
    for i, (d, t) in enumerate(zip(detections, tracks)):
        source = [tuple(b) for b in d["boxes_xyxy_score"]]
        raw = {tuple(t["box"] + [t["score"]]): t for t in t["raw_tracks"]}
        displayed = {tuple(t["box"] + [t["score"]]) for t in t["displayed_tracks"]}
        require(set(raw).issubset(source) and displayed.issubset(raw), f"Output changed boxes at {i}")
        require(all(not v["predicted"] for v in raw.values()), "Unexpected predicted output")
        removed = []
        for box in source:
            if box in displayed:
                continue
            reason = "unconfirmed_hits_lt3" if box in raw else "unmatched_below_birth_0.10"
            if box in raw:
                require(not raw[box]["confirmed"] and raw[box]["hits"] < 3, "Unexpected confirmation filter")
            else:
                require(box[4] < .1, "Unexplained suppression above birth threshold")
            box_reasons[reason] += 1
            removed.append(dict(box=list(box), reason=reason,
                                raw_track=raw.get(box), best_gt_iou=max((box_iou(box, g) for g in gt[i]), default=0)))
        whole_blank_reason = None
        if source and not displayed:
            whole_blank_reason = "only_unconfirmed_tracks" if raw else "unmatched_below_birth_0.10"
            frame_reasons[whole_blank_reason] += 1
        row = dict(frame_index=i, time_seconds=i/fps, reviewed=i in included, gt_count=len(gt[i]),
                   detector_count=len(source), raw_track_count=len(raw), shown_count=len(displayed),
                   blank_reason=whole_blank_reason, removed=removed)
        per_frame.append(row)
        if removed:
            missed.append(row)
    args.output.mkdir(parents=True)
    library = (args.output / "diagnostic_tracker.so").resolve()
    subprocess.run([args.cxx, "-O3", "-std=c++17", "-shared", "-fPIC",
                    str(ROOT / "scripts/anti_uav/rknn_yolov8_native/tracker_c_api.cpp"), "-o", str(library)], check=True)
    size = (meta["video"]["frameWidth"], meta["video"]["frameHeight"])
    # Exact baseline replay checks the compiled implementation before any ablation.
    replay(library, detections, fps, size, .10, 3, reference=tracks)
    for birth, hits in ((.10, 2), (.03, 3), (.03, 2), (.03, 1)):
        name = f"birth{birth:.2f}_hits{hits}"
        frames[name] = replay(library, detections, fps, size, birth, hits)
    results, matching = {}, {}
    for name, outputs in frames.items():
        results[name], matching[name] = evaluate(outputs, gt, included, .5)
    require(results["birth0.03_hits1"]["tp"] == results["detector"]["tp"] and
            results["birth0.03_hits1"]["fp"] == results["detector"]["fp"], "No-gate replay failed to retain detections")
    gt_loss = Counter()
    for i in sorted(included):
        if matching["detector"][i][0] > matching["tracker_current"][i][0]:
            reason = "confirmation_hits_lt3" if matching["tracker_raw"][i][0] > matching["tracker_current"][i][0] else "birth_gate"
            gt_loss[reason] += 1
            per_frame[i]["lost_tp_reason"] = reason
    examples = {}
    for index in (2196, 2197, 2212, 9000):
        examples[str(index)] = dict(per_frame[index], detections=detections[index]["boxes_xyxy_score"],
                                   raw_tracks=tracks[index]["raw_tracks"], gt=gt[index],
                                   counts={k: matching[k].get(index) for k in ("detector", "tracker_current")})
    result = dict(
        scope="Full original Video00009, fixed conf=0.03 PT FP32; not the every-10th-frame validation subset",
        total_video_frames=count, reviewed_frames=len(included), excluded_frames=sorted(set(range(count))-included),
        positive_frames=sum(bool(gt[i]) for i in included), negative_frames=sum(not gt[i] for i in included),
        gt_boxes=sum(len(gt[i]) for i in included), matching="single-GT highest IoU, IoU>=0.50; extra boxes are FP",
        exact_baseline_replay_verified=True, no_deployment_configuration_changed=True,
        detector_visible_frames=sum(bool(x) for x in frames["detector"]),
        raw_track_visible_frames=sum(bool(x) for x in frames["tracker_raw"]),
        confirmed_track_visible_frames=sum(bool(x) for x in frames["tracker_current"]),
        whole_frame_suppression=dict(frame_reasons), removed_boxes=dict(box_reasons),
        lost_gt_detections=dict(gt_loss), metrics=results, examples=examples,
        input_sha256={str(p): digest(p) for p in (pred_path, tracks_path, args.approved_manifest, args.coco)},
        note="Ablations use the SAME >=0.03 cache. No 0.01-0.03 detections exist here. GT is used only for offline scoring, not tracker decisions. Unique track IDs are not ID-switch metrics. These are diagnostics on a model-selection video, not independent held-out tuning results.")
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "suppressed_frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in missed))
    print(json.dumps({k:v for k,v in result.items() if k not in ("examples", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
