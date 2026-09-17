#!/usr/bin/env python3
"""Controlled parameter-only replay on one validation video; never deploys a preset."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.analyze_cached_tracker_suppression import digest, evaluate, read_jsonl, require
from scripts.anti_uav.evaluate_detector_clip_csv import box_iou
from scripts.anti_uav.run_causal_roi_comparison import NativeTracker


def identity_counts(frames, gt, included):
    segments, segment = {}, 0
    for index in sorted(included):
        if not gt.get(index):
            continue
        if index-1 not in segments:
            segment += 1
        segments[index] = segment
    previous_id = previous_frame = None
    changes = continuous = adjacent = 0
    for i in sorted(segments):
        best = max(frames[i], key=lambda t: box_iou(t["box"], gt[i][0]), default=None)
        if best is None or box_iou(best["box"], gt[i][0]) < .5:
            continue
        if previous_id is not None and previous_id != best["id"]:
            changes += 1
            continuous += segments[previous_frame] == segments[i]
            adjacent += previous_frame == i-1
        previous_id, previous_frame = best["id"], i
    return dict(all_matched_observation_id_changes=changes,
                continuous_gt_id_changes=continuous, adjacent_tp_frame_id_changes=adjacent)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector-dir", type=Path, required=True)
    p.add_argument("--tracker-dir", type=Path, required=True)
    p.add_argument("--approved-manifest", type=Path, required=True)
    p.add_argument("--coco", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cxx", default="c++")
    args = p.parse_args()
    require(not args.output.exists(), "Use a fresh output directory")
    ds = json.loads((args.detector_dir / "summary.json").read_text())
    ts = json.loads((args.tracker_dir / "summary.json").read_text())
    meta = json.loads(args.approved_manifest.read_text())
    coco = json.loads(args.coco.read_text())
    cache = args.detector_dir / "predictions.jsonl"
    require(digest(cache) == ts["predictions_sha256"], "Detector cache changed")
    require(ds["source_sha256"] == ts["source_sha256"] == meta["video"]["sha256"], "Video mismatch")
    require(ds["weights_sha256"] == ts["weights_sha256"], "Model mismatch")
    require(ds["conf"] == .03 and ds["full_source_video"] and ds["frame_stride"] == 1, "Unsupported cache")
    require(digest(args.coco) == next(r["sha256"] for r in meta["files"] if r["path"] == "coco/annotations.json"), "GT hash mismatch")
    include = set(meta["frames"]["includedFrameIndices"])
    include -= set(meta["frames"].get("excludedUncertainFrameIndices", []))
    include -= set(meta["frames"].get("excludedUnreviewedFrameIndices", []))
    byid = {im["id"]: im["frame_index"] for im in coco["images"]}
    require(set(byid.values()) == include, "GT frame coverage mismatch")
    gt = defaultdict(list)
    for ann in coco["annotations"]:
        require(ann["category_id"] == 0 and not ann.get("iscrowd"), "Unexpected GT category")
        x, y, w, h = ann["bbox"]
        gt[byid[ann["image_id"]]].append([x, y, x+w, y+h])
    require(all(len(v) <= 1 for v in gt.values()), "Single-GT diagnostic only")
    detections = read_jsonl(cache)
    fps = meta["video"]["fps"]
    require(len(detections) == meta["video"]["frameCount"] == ds["frames"], "Incomplete cache")
    require(all(r["frame_index"] == i and abs(r["time_seconds"]-i/fps) < 1e-8
                for i, r in enumerate(detections)), "Frame order/time mismatch")
    args.output.mkdir(parents=True)
    library = (args.output / "tracker_fixed.so").resolve()
    native = ROOT / "scripts/anti_uav/rknn_yolov8_native"
    subprocess.run([args.cxx, "-O3", "-std=c++17", "-shared", "-fPIC",
                    str(native / "tracker_c_api.cpp"), "-o", str(library)], check=True)
    presets = [dict(match=m, birth=b, hits=h, buffer=1.)
               for m, b, h in itertools.product((.92, .94, .96, .98, .99), (.10, .03), (3, 2))]
    presets += [dict(match=.92, birth=.03, hits=1, buffer=1.)]
    presets += [dict(match=.92, birth=.10, hits=3, buffer=b) for b in (.3, .5, 2.)]
    results = []
    size = (meta["video"]["frameWidth"], meta["video"]["frameHeight"])
    for number, config in enumerate(presets):
        tracker = NativeTracker(library, fps)
        tracker.close()
        tracker.handle = tracker.lib.rk_tracker_create(.03, .01, config["birth"], config["match"],
            config["match"], config["buffer"], 0., fps, config["hits"])
        require(bool(tracker.handle), "Tracker initialization failed")
        frames = []
        try:
            for i, record in enumerate(detections):
                raw = tracker.update(record["boxes_xyxy_score"], i/fps, *size)
                frames.append([t for t in raw if t["confirmed"] and not t["predicted"]])
        finally:
            tracker.close()
        metrics, _ = evaluate(frames, gt, include, .5)
        result = dict(config=config, metrics=metrics, identity_diagnostics=identity_counts(frames, gt, include))
        results.append(result)
        print(json.dumps(dict(trial=number+1, total=len(presets), **result)), flush=True)
    detector_metrics, _ = evaluate([[dict(box=b[:4]) for b in r["boxes_xyxy_score"]] for r in detections], gt, include, .5)
    require(results[0]["metrics"]["tp"] == 5086 and results[0]["metrics"]["fp"] == 2675,
            "Fixed-source baseline no longer matches the audited Video00009 results")
    no_gate = results[20]["metrics"]
    require(all(no_gate[k] == detector_metrics[k] for k in ("tp", "fp", "fn")), "No-gate control changed detections")
    report = dict(scope="Video00009 validation-only parameter sweep, not a new independent test",
        changes="Only match cost, birth score, min hits and buffer; confirmed_first=false, no predicted display, no low supplementation",
        metric_warning="Identity diagnostics are not MOTChallenge IDSW/IDF1. Fewer changes can also come from wrong merges. No preset is deployed.",
        frames=len(detections), reviewed_frames=len(include), detector_metrics=detector_metrics,
        provenance=dict(detector_cache_sha256=digest(cache), coco_sha256=digest(args.coco),
                        tracker_header_sha256=digest(native / "detector_based_tracker.hpp"), library_sha256=digest(library),
                        weights_sha256=ds["weights_sha256"], source_sha256=ds["source_sha256"]),
        results=results, best_validation_f1=max(results, key=lambda r: r["metrics"]["f1"]))
    (args.output / "sweep.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
