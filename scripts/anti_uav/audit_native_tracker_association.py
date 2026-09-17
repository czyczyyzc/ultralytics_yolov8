#!/usr/bin/env python3
"""Trace native associations and score immutable Video00009 detector-cache replay."""
import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.analyze_cached_tracker_suppression import evaluate, read_jsonl, require
from scripts.anti_uav.evaluate_detector_clip_csv import box_iou, match_frame
from scripts.anti_uav.run_causal_roi_comparison import NativeTracker


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_supplement(original, supplemental, low=.01, high=.03):
    require(len(original) == len(supplemental), "Supplement frame count mismatch")
    added = 0
    for old, new in zip(original, supplemental):
        require(old["frame_index"] == new["frame_index"] and old["time_seconds"] == new["time_seconds"],
                "Supplement frame/time mismatch")
        boxes = new["boxes_xyxy_score"]
        require(all(len(b) == 5 and np.all(np.isfinite(b)) and b[4] >= low for b in boxes),
                "Invalid supplemental detection")
        require([b for b in boxes if b[4] >= high] == old["boxes_xyxy_score"], "Original high boxes changed")
        added += sum(b[4] < high for b in boxes)
    return added


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector-dir", type=Path, required=True)
    p.add_argument("--tracker-dir", type=Path, required=True)
    p.add_argument("--approved-manifest", type=Path, required=True)
    p.add_argument("--coco", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expect-baseline", action="store_true")
    p.add_argument("--confirmed-first", action="store_true")
    p.add_argument("--supplement-low-dir", type=Path)
    p.add_argument("--active-first", action="store_true")
    p.add_argument("--match-cost", type=float, default=.92)
    p.add_argument("--retain-raw", action="store_true")
    p.add_argument("--trace-frames", default="")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dpath = args.detector_dir / "predictions.jsonl"
    tpath = args.tracker_dir / "tracks.jsonl"
    detections, original = read_jsonl(dpath), read_jsonl(tpath)
    ds = json.loads((args.detector_dir / "summary.json").read_text())
    ts = json.loads((args.tracker_dir / "summary.json").read_text())
    meta = json.loads(args.approved_manifest.read_text())
    coco = json.loads(args.coco.read_text())
    require(sha(dpath) == ts["predictions_sha256"], "Detector cache changed")
    require(ds["source_sha256"] == meta["video"]["sha256"], "Annotation video mismatch")
    require(sha(args.coco) == next(r["sha256"] for r in meta["files"] if r["path"] == "coco/annotations.json"), "COCO hash mismatch")
    include = set(meta["frames"]["includedFrameIndices"])
    include -= set(meta["frames"].get("excludedUncertainFrameIndices", []))
    include -= set(meta["frames"].get("excludedUnreviewedFrameIndices", []))
    byid = {im["id"]: im["frame_index"] for im in coco["images"]}
    gt = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[byid[a["image_id"]]].append((x, y, x+w, y+h))
    require(all(len(v) <= 1 for v in gt.values()), "Single-target diagnostic only")
    require(len(detections) == len(original) == meta["video"]["frameCount"], "Incomplete cache")
    input_records = detections
    supplement = None
    if args.supplement_low_dir:
        require(not args.expect_baseline, "Supplemented input cannot be an exact baseline")
        sp = args.supplement_low_dir / "predictions.jsonl"
        ss = json.loads((args.supplement_low_dir / "summary.json").read_text())
        require(ss["baseline_predictions_sha256"] == sha(dpath), "Supplement baseline mismatch")
        require(ss["source_sha256"] == ds["source_sha256"] and ss["weights_sha256"] == ds["weights_sha256"],
                "Supplement video/weights mismatch")
        require(ss["predictions_sha256"] == sha(sp), "Supplement hash mismatch")
        input_records = read_jsonl(sp)
        added = validate_supplement(detections, input_records)
        require(added == ss["low_detections_added"], "Supplement low detection count mismatch")
        supplement = dict(predictions_sha256=sha(sp), low_detections_added=added)
    fps = meta["video"]["fps"]
    tracker = NativeTracker(args.library.resolve(), fps)
    require(0 < args.match_cost < 1, "Invalid match cost")
    require(not (args.active_first and args.confirmed_first), "Choose only one priority policy")
    require(not args.expect_baseline or (not args.active_first and args.match_cost == .92), "Baseline policy changed")
    if args.match_cost != .92:
        tracker.close()
        tracker.handle = tracker.lib.rk_tracker_create(.03, .01, .1, args.match_cost, args.match_cost, 1., 0., fps, 3)
        require(bool(tracker.handle), "Tracker initialization failed")
    lib = tracker.lib
    if args.confirmed_first:
        require(not args.expect_baseline, "A changed policy cannot be an exact baseline")
        lib.rk_tracker_set_confirmed_first.argtypes = [ctypes.c_void_p, ctypes.c_int]
        require(lib.rk_tracker_set_confirmed_first(tracker.handle, 1) == 0, "Policy configuration failed")
    if args.active_first:
        lib.rk_tracker_set_active_first.argtypes = [ctypes.c_void_p, ctypes.c_int]
        require(lib.rk_tracker_set_active_first(tracker.handle, 1) == 0, "Policy configuration failed")
    fp, dp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    lib.rk_tracker_trace.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, ctypes.c_double, dp, ctypes.c_int]
    lib.rk_tracker_assign.argtypes = [dp, ctypes.c_int, ctypes.c_int, ctypes.c_double, ip]
    storage = np.zeros((32768, 15), np.float64)
    stats, events, rows, ids = Counter(), [], [], set()
    raw_rows, selected_traces = [], []
    trace_frames = {int(s) for s in args.trace_frames.split(",") if s}
    last_matched_id, last_matched_frame = None, None
    consecutive_switches = all_observation_switches = 0
    identity_events = []
    tracking_ms = []
    try:
        for i, record in enumerate(detections):
            require(record["frame_index"] == original[i]["frame_index"] == i, "Frame order mismatch")
            boxes = np.asarray(input_records[i]["boxes_xyxy_score"], np.float32).reshape(-1, 5)
            count = lib.rk_tracker_trace(tracker.handle, boxes.ctypes.data_as(fp), len(boxes), i/fps,
                                         storage.ctypes.data_as(dp), len(storage))
            require(count >= 0, "Trace capacity or bridge error")
            cost_rows = storage[:count].copy()
            raw_matrix_changed = False
            if count and len(boxes) and supplement is None:
                costs = np.ascontiguousarray(cost_rows[:, 5].reshape(-1, len(boxes)))
                assignment = np.full(costs.shape[0], -1, np.int32)
                masked_assignment = assignment.copy()
                lib.rk_tracker_assign(costs.ctypes.data_as(dp), *costs.shape, args.match_cost, assignment.ctypes.data_as(ip))
                masked = np.ascontiguousarray(np.where(costs < args.match_cost, costs, 1000+args.match_cost))
                lib.rk_tracker_assign(masked.ctypes.data_as(dp), *masked.shape, args.match_cost, masked_assignment.ctypes.data_as(ip))
                raw_matrix_changed = not np.array_equal(assignment, masked_assignment)
                stats["frames_assignment_changes_if_invalid_edges_masked"] += raw_matrix_changed
            before = time.perf_counter()
            raw = tracker.update(boxes, i/fps, meta["video"]["frameWidth"], meta["video"]["frameHeight"])
            tracking_ms.append((time.perf_counter()-before)*1000)
            if args.expect_baseline:
                require(raw == original[i]["raw_tracks"], f"Baseline trace changed behavior at {i}")
            if args.retain_raw:
                raw_rows.append(raw)
            if i in trace_frames:
                selected_traces.append(dict(frame_index=i, detections=boxes.tolist(), raw_tracks=raw,
                                             costs_before_update=cost_rows.tolist()))
            shown = [t for t in raw if t["confirmed"] and not t["predicted"]]
            rows.append(shown)
            ids.update(t["id"] for t in shown)
            if i not in include:
                continue
            det_rows = [dict(box=b[:4]) for b in record["boxes_xyxy_score"]]
            dtp = match_frame(gt[i], det_rows, .5)[0]
            stp = match_frame(gt[i], shown, .5)[0]
            if stp:
                best = max(shown, key=lambda t: box_iou(t["box"], gt[i][0]))
                if last_matched_id is not None and best["id"] != last_matched_id:
                    all_observation_switches += 1
                    consecutive_switches += last_matched_frame == i-1
                    continuously_annotated = all(j in include and gt[j] for j in range(last_matched_frame, i+1))
                    identity_events.append(dict(previous_frame=last_matched_frame, frame_index=i,
                        previous_id=last_matched_id, track_id=best["id"],
                        adjacent_frames=last_matched_frame == i-1,
                        gt_present_in_every_intervening_reviewed_frame=continuously_annotated))
                last_matched_id, last_matched_frame = best["id"], i
            if dtp > stp:
                candidates = [j for j, b in enumerate(boxes) if b[4] >= .03 and box_iou(b, gt[i][0]) >= .5]
                relevant = cost_rows[np.isin(cost_rows[:, 1], candidates)] if count else cost_rows
                confirmed_candidates = relevant[(relevant[:, 2] >= 3) & (relevant[:, 4] <= 1.) & (relevant[:, 5] < args.match_cost)]
                best_cost = float(np.min(relevant[:, 5])) if len(relevant) else None
                stats["missed_tp_frames"] += 1
                stats["missed_tp_with_eligible_confirmed_competitor"] += bool(len(confirmed_candidates))
                stats["missed_tp_with_any_eligible_track"] += bool(len(relevant) and best_cost < args.match_cost)
                if len(relevant) and not best_cost < args.match_cost:
                    stats["missed_tp_no_eligible_association"] += 1
                matched = [t for t in raw if any(np.allclose(t["box"], boxes[j, :4], atol=1e-5, rtol=0) for j in candidates)]
                stats["missed_tp_selected_tentative"] += bool(matched)
                events.append(dict(frame_index=i, seconds=i/fps, candidates=candidates,
                    detector_boxes=boxes.tolist(), gt=gt[i], selected_tracks=matched,
                    confirmed_competitors=confirmed_candidates.tolist(),
                    best_cost=best_cost, assignment_gate_changes=raw_matrix_changed,
                    candidate_cost_rows=sorted(relevant.tolist(), key=lambda r:r[5])[:8]))
    finally:
        tracker.close()
    score, _ = evaluate(rows, gt, include, .5)
    high_only_score, _ = evaluate([[t for t in row if t["score"] >= .03] for row in rows], gt, include, .5)
    args.output.mkdir(parents=True)
    (args.output / "tracks.jsonl").write_text("".join(json.dumps(dict(frame_index=i, displayed_tracks=r,
        **({"raw_tracks": raw_rows[i]} if args.retain_raw else {})))+"\n" for i,r in enumerate(rows)))
    (args.output / "selected_cost_traces.json").write_text(json.dumps(dict(
        columns=["id", "detection_index", "hits", "age", "gap_seconds", "cost", "obs_x1", "obs_y1", "obs_x2", "obs_y2",
                 "pred_x1", "pred_y1", "pred_x2", "pred_y2", "mahalanobis"], frames=selected_traces),indent=2)+"\n")
    (args.output / "missed_tp_events.json").write_text(json.dumps(events, indent=2)+"\n")
    (args.output / "identity_change_events.json").write_text(json.dumps(identity_events, indent=2)+"\n")
    result = dict(name=args.name, metrics=score, metrics_display_score_ge003=high_only_score, diagnostics=dict(stats),
        visible_ids=len(ids), matched_gt_id_changes_including_gaps=all_observation_switches,
        matched_gt_id_changes_adjacent_frames=consecutive_switches,
        matched_gt_id_changes_with_continuous_positive_gt=sum(e["gt_present_in_every_intervening_reviewed_frame"] for e in identity_events),
        metric_warning="ID changes are explicit single-GT diagnostics, NOT standard MOTChallenge IDSW/IDF1. Wall time includes ctypes/Python overhead on this host, not board FPS.",
        tracker_update_mean_ms=float(np.mean(tracking_ms)), baseline_exact=args.expect_baseline,
        reviewed_frames=len(include), total_frames=len(detections),
        config=dict(high=.03, low=.01, birth=.1, min_hits=3, buffer_sec=1., prediction_sec=0., first=args.match_cost, second=args.match_cost,
                    confirmed_first=args.confirmed_first, active_first=args.active_first),
        library_sha256=sha(args.library), detector_cache_sha256=sha(dpath), supplement=supplement,
        provenance=dict(source_sha256=ds["source_sha256"], weights_sha256=ds["weights_sha256"],
                        coco_sha256=sha(args.coco), audit_script_sha256=sha(Path(__file__))))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
