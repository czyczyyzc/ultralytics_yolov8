#!/usr/bin/env python3
"""Audit fixed-detection GMC/identity-confirmation ablations, without claiming GT accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.render_cached_tracker_result_video import load_records, validate_records
from scripts.anti_uav.render_pt_detector_video import dump, sha256
from scripts.anti_uav.motion_native_runtime import DEFAULTS


def top_observation_ids(rows, indices):
    observations = []
    for i in indices:
        row = rows[i]
        if not row["boxes_xyxy_score"]:
            continue
        # Detector caches are score-sorted. This is a diagnostic, not a GT identity.
        if any(b[4]>row["boxes_xyxy_score"][0][4] for b in row["boxes_xyxy_score"]):
            raise ValueError("Top-observation diagnostic needs score-sorted detections")
        obs = next(o for o in row["observations"] if o["detection_index"]==0)
        observations.append((i, obs["id"]))
    identified = [(i, identity) for i, identity in observations if identity is not None]
    return dict(frames_with_top_detection=len(observations), frames_with_top_id=len(identified),
        top_detection_without_id=len(observations)-len(identified),
        top_ids=sorted({identity for _, identity in identified}),
        id_changes_between_identified_observations=sum(a[1]!=b[1] for a,b in zip(identified,identified[1:])),
        adjacent_frame_id_changes=sum(a[1]!=b[1] and b[0]==a[0]+1 for a,b in zip(identified,identified[1:])),
        frame_ids=[dict(frame=i, id=identity) for i, identity in observations])


def case_counts(rows, fps):
    states = Counter(o["status"] for r in rows for o in r["observations"])
    indices = [i for i in range(len(rows)) if 13.<=i/fps<14.]
    return dict(detector_boxes=sum(len(r["boxes_xyxy_score"]) for r in rows),
        output_boxes=sum(len(r["observations"]) for r in rows),
        boxes_with_id=sum(o["id"] is not None for r in rows for o in r["observations"]),
        boxes_without_id=sum(o["id"] is None for r in rows for o in r["observations"]),
        distinct_output_ids=len({o["id"] for r in rows for o in r["observations"] if o["id"] is not None}),
        status_counts=dict(states), critical_window_13_14=top_observation_ids(rows, indices),
        frames_without_id=[dict(frame=i, detection_index=o["detection_index"], status=o["status"], score=o["score"])
            for i,r in enumerate(rows) for o in r["observations"] if o["id"] is None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(models={}, scope="000002 fixed FP32 detections; no GT provided, no precision/recall/MOT-IDSW claims",
        inference_rerun=False, id_remapping=False, predicted_boxes_shown=False,
        no_gmc_definition="Identity warp, quality zero, unknown_gmc_speed_px_s=0; model total image-coordinate velocity",
        immediate_id_definition="birth=.03, confirmation_hits=1, confirmation_window=1; ambiguity rejection retained")
    native_hashes = set()
    for model in ("p3", "p2p3"):
        det_dir = args.detector_root/(model+"_detection")
        detector = json.loads((det_dir/"summary.json").read_text())
        detections = load_records(det_dir/"predictions.jsonl")
        fps = detector["output_fps"]
        if len(detections)!=1800 or fps!=30. or detector["conf"]!=.03 or detector["input_hw"]!=[544,960]:
            raise ValueError("Unexpected 000002 detector protocol")
        reference = load_records(args.reference_root/(model+"_geometry_trial")/"tracks.jsonl")
        data, estimated_inputs = {}, None
        for policy in ("balanced", "immediate"):
            for mode in ("estimate", "disabled"):
                name = model+"_"+policy+"_"+mode
                directory = args.experiment_root/name
                summary = json.loads((directory/"summary.json").read_text())
                rows = load_records(directory/"tracks.jsonl")
                provenance = summary["provenance"]
                if (summary["tracks_sha256"]!=sha256(directory/"tracks.jsonl") or
                        summary["detector_cache_sha256"]!=sha256(det_dir/"predictions.jsonl") or
                        provenance["source_sha256"]!=detector["source_sha256"] or
                        provenance["weights_sha256"]!=detector["weights_sha256"]):
                    raise ValueError("Cache/source/weights provenance changed")
                native_hashes.add(provenance["tracker_library_sha256"])
                validate_records(detections, rows, len(detections), fps)
                if any(r["boxes_xyxy_score"]!=d["boxes_xyxy_score"] for r,d in zip(rows,detections)):
                    raise ValueError("Detector measurements changed")
                expected = dict(DEFAULTS, nominal_fps=fps)
                if policy=="immediate":
                    expected.update(birth=.03, confirmation_hits=1, confirmation_window=1)
                if mode=="disabled":
                    expected["unknown_gmc_speed_px_s"] = 0.
                    if any(r["warp"]!=[[1.,0.,0.],[0.,1.,0.]] or r["gmc_quality"]!=0.
                           or not r["gmc_meta"].get("disabled") for r in rows):
                        raise ValueError("Disabled GMC contains non-identity motion")
                else:
                    inputs = [(r["warp"],r["gmc_quality"]) for r in rows]
                    if estimated_inputs is not None and inputs!=estimated_inputs:
                        raise ValueError("Confirmation policies have different GMC inputs")
                    estimated_inputs = inputs
                if summary["config"]!=expected:
                    raise ValueError("Unexpected ablation configuration")
                if policy=="balanced" and mode=="estimate":
                    if any(a["observations"]!=b["observations"] or a["displayed_tracks"]!=b["displayed_tracks"]
                           for a,b in zip(rows,reference)) or len(rows)!=len(reference):
                        raise ValueError("Latest balanced + GMC differs from previous verified regression")
                data[policy+"_"+mode] = dict(case_counts(rows, fps), config=summary["config"],
                    motion_stats=summary["motion_stats"], tracks_sha256=summary["tracks_sha256"], provenance=provenance)
        result["models"][model] = dict(cases=data, fps=fps, frames=len(detections), detector_summary=detector)
    if len(native_hashes)!=1:
        raise ValueError("Ablations use different native association libraries")
    result["native_library_sha256"] = next(iter(native_hashes))
    dump(args.output, result)
    for model,data in result["models"].items():
        for name,row in data["cases"].items():
            print(json.dumps(dict(model=model, case=name, boxes_with_id=row["boxes_with_id"],
                boxes_without_id=row["boxes_without_id"], ids=row["distinct_output_ids"],
                critical=row["critical_window_13_14"])), flush=True)


if __name__=="__main__":
    main()
