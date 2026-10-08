#!/usr/bin/env python3
"""Audit measured-box coverage separately from confirmed native tracking identity."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.anti_uav.render_cached_tracker_result_video import load_records, validate_records
from scripts.anti_uav.render_pt_detector_video import dump, sha256
from scripts.anti_uav.evaluate_dist_tracker_cache import load_ground_truth
from scripts.anti_uav.evaluate_detector_clip_csv import box_iou
from scripts.anti_uav.analyze_cached_tracker_suppression import evaluate
from scripts.anti_uav.sweep_native_tracker_parameters import identity_counts


def coverage(rows):
    boxes = sum(len(r["boxes_xyxy_score"]) for r in rows)
    confirmed = sum(len(r["displayed_tracks"]) for r in rows)
    has_contract = all("observations" in r for r in rows)
    emitted = sum(len(r["observations"] if has_contract else r["displayed_tracks"]) for r in rows)
    statuses = {}
    if has_contract:
        for r in rows:
            for o in r["observations"]:
                statuses[o["status"]] = statuses.get(o["status"], 0)+1
    return dict(frames=len(rows), detector_boxes=boxes, emitted_measured_boxes=emitted,
        missing_measured_boxes=boxes-emitted, confirmed_track_boxes=confirmed,
        observations_without_confirmed_identity=boxes-confirmed,
        frames_with_detections=sum(bool(r["boxes_xyxy_score"]) for r in rows),
        frames_with_emitted_boxes=sum(bool(r["observations"] if has_contract else r["displayed_tracks"]) for r in rows),
        frames_with_confirmed_tracks=sum(bool(r["displayed_tracks"]) for r in rows),
        confirmed_ids=len({t["id"] for r in rows for t in r["displayed_tracks"]}),
        output_has_identity_status=has_contract, status_counts=statuses)


def labelled(rows, gt, included):
    frames = [r["displayed_tracks"] for r in rows]
    metrics, _ = evaluate(frames, gt, included, .5)
    identities = identity_counts(frames, gt, included)
    missing = []
    for i in sorted(included):
        valid = {di for di, b in enumerate(rows[i]["boxes_xyxy_score"])
                 if any(box_iou(b[:4], target)>=.5 for target in gt.get(i, []))}
        associated = {t["detection_index"] for t in frames[i]}
        if valid and not valid.intersection(associated):
            missing.append(i)
    segments, segment = {}, 0
    for i in sorted(included):
        if not gt.get(i):
            continue
        if i-1 not in segments:
            segment += 1
        segments[i] = segment
    previous, events = None, []
    for i in sorted(segments):
        best = max(frames[i], key=lambda t: box_iou(t["box"], gt[i][0]), default=None)
        if not best or box_iou(best["box"], gt[i][0])<.5:
            continue
        if previous and best["id"]!=previous[1]:
            events.append(dict(previous_frame=previous[0], frame=i, previous_id=previous[1], id=best["id"],
                continuous_gt_segment=segments[previous[0]]==segments[i]))
        previous = (i, best["id"])
    return dict(confirmed_only_metrics=metrics, identity_diagnostics=identities,
        identity_change_events=events, detector_tp_without_confirmed_track_frames=missing,
        detector_tp_without_confirmed_track_count=len(missing),
        identity_scope="Single-target continuous-GT diagnostic, not formal MOT IDSW/IDF1/HOTA")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--tracker-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--approved-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    detector_path = args.detector_dir/"predictions.jsonl"
    detector = json.loads((args.detector_dir/"summary.json").read_text())
    detections = load_records(detector_path)
    current = load_records(args.tracker_dir/"tracks.jsonl")
    baseline = load_records(args.baseline_dir/"tracks.jsonl")
    for directory, rows in ((args.tracker_dir, current), (args.baseline_dir, baseline)):
        summary = json.loads((directory/"summary.json").read_text())
        if summary["detector_cache_sha256"]!=sha256(detector_path):
            raise ValueError("Detector inputs changed")
        if sha256(directory/"tracks.jsonl")!=summary["tracks_sha256"]:
            raise ValueError("Tracker cache changed")
        validate_records(detections, rows, len(detections), detector["output_fps"])
    for i, (old, new) in enumerate(zip(baseline, current)):
        if old["boxes_xyxy_score"]!=new["boxes_xyxy_score"] or old["warp"]!=new["warp"]:
            raise ValueError(f"Controlled tracker inputs changed at {i}")
    if not all("observations" in r for r in current):
        raise ValueError("Current cache has no all-observation output contract")
    result = dict(baseline=coverage(baseline), current=coverage(current),
        detector_cache_sha256=sha256(detector_path),
        source_sha256=detector["source_sha256"], weights_sha256=detector["weights_sha256"],
        tracker_summary=json.loads((args.tracker_dir/"summary.json").read_text()),
        baseline_tracks_sha256=sha256(args.baseline_dir/"tracks.jsonl"),
        current_tracks_sha256=sha256(args.tracker_dir/"tracks.jsonl"),
        independent_test=False, predicted_boxes_used=False, id_remapping=False,
        scope="Fixed FP32 detector/GMC offline regression; no board FPS or generalization claim")
    if args.approved_manifest:
        _, gt, included = load_ground_truth(args.approved_manifest, args.approved_manifest.parent/"coco/annotations.json")
        if any(len(v)>1 for v in gt.values()):
            raise ValueError("Identity diagnostic requires single-target GT")
        manifest = json.loads(args.approved_manifest.read_text())
        if manifest["video"]["sha256"]!=detector["source_sha256"]:
            raise ValueError("Approved GT does not match detector source")
        result["approved_manifest_sha256"] = sha256(args.approved_manifest)
        result["coco_sha256"] = sha256(args.approved_manifest.parent/"coco/annotations.json")
        result["baseline"].update(labelled(baseline, gt, included))
        result["current"].update(labelled(current, gt, included))
    dump(args.output, result)
    print(json.dumps({k: result[k] for k in ("baseline", "current")}), flush=True)


if __name__=="__main__":
    main()
