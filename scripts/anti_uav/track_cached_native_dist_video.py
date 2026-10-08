#!/usr/bin/env python3
"""Replay native Dist association plus efficient GMC on an unlabelled video cache."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time
import statistics

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.dist_numpy_runtime import CONFIG
from scripts.anti_uav.efficient_gmc import EfficientGMC
from scripts.anti_uav.native_dist_runtime import NativeDist
from scripts.anti_uav.motion_native_runtime import NativeMotion
from scripts.anti_uav.render_cached_tracker_result_video import validate_records
from scripts.anti_uav.render_pt_detector_video import dump, probe, sha256


def motion_config_for_replay(fps, custom, gmc_mode):
    from scripts.anti_uav.motion_native_runtime import DEFAULTS
    config = dict(DEFAULTS, nominal_fps=float(fps)) if custom is None else dict(custom)
    if set(config)!=set(DEFAULTS):
        raise ValueError("Motion configuration must contain exactly the documented keys")
    if gmc_mode=="disabled":
        # Model the combined image-coordinate velocity, not an unknown failed compensation.
        # No camera-variance term is injected into the observation velocity fit in this mode.
        config["unknown_gmc_speed_px_s"] = 0.
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--tracker-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracker-kind", choices=("dist", "motion"), default="dist")
    parser.add_argument("--baseline-dir", type=Path, help="Verify GMC and detections against a frozen baseline")
    parser.add_argument("--motion-config", type=Path)
    parser.add_argument("--cached-gmc", action="store_true", help="Offline ablation only; requires quality-aware baseline cache")
    parser.add_argument("--gmc-mode", choices=("estimate", "disabled", "unavailable"), default="estimate",
        help="Disabled: image-coordinate motion; unavailable: no warp but retain default unknown-camera uncertainty")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.cached_gmc and not args.baseline_dir:
        parser.error("--cached-gmc requires --baseline-dir")
    if args.cached_gmc and args.gmc_mode!="estimate":
        parser.error("--cached-gmc requires estimated GMC")
    detector = json.loads((args.detector_dir / "summary.json").read_text())
    cache = args.detector_dir / "predictions.jsonl"
    source = probe(args.source)
    count, fps = int(source["nb_frames"]), float(Fraction(source["avg_frame_rate"]))
    if (Fraction(source["avg_frame_rate"]) != Fraction(source["r_frame_rate"])
            or fps <= 0 or not detector["full_source_video"] or detector["frame_stride"] != 1
            or detector["frames"] != count or detector["output_fps"] != fps
            or detector["source_sha256"] != sha256(args.source)):
        raise ValueError("Need a complete, matching constant-frame-rate detector cache")
    records = [json.loads(line) for line in cache.read_text().splitlines()]
    baseline = None
    if args.baseline_dir:
        original = json.loads((args.baseline_dir / "summary.json").read_text())
        if original["detector_cache_sha256"] != sha256(cache):
            raise ValueError("Baseline detections differ")
        baseline = [json.loads(line) for line in (args.baseline_dir / "tracks.jsonl").read_text().splitlines()]
        if len(baseline) != count:
            raise ValueError("Baseline frame coverage differs")
        if args.cached_gmc and any("gmc_quality" not in row for row in baseline):
            raise ValueError("Cached GMC must contain per-frame quality")
    if len(records) != count:
        raise ValueError("Detector cache does not cover every frame")
    for index, row in enumerate(records):
        if row["frame_index"] != index or abs(row["time_seconds"]-index/fps) > 1e-7:
            raise ValueError(f"Unordered detector frame/timestamp at {index}")
    args.output.mkdir(parents=True)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(20260924)
    config = dict(CONFIG)
    if args.tracker_kind == "motion":
        custom = json.loads(args.motion_config.read_text()) if args.motion_config else None
        tracker = NativeMotion(args.tracker_library, fps=fps,
            config=motion_config_for_replay(fps, custom, args.gmc_mode))
        config = tracker.config
    else:
        tracker = NativeDist(args.tracker_library, fps=fps, config=config)
    provenance = dict(source_sha256=detector["source_sha256"],
        weights_sha256=detector["weights_sha256"], detector_cache_sha256=sha256(cache),
        tracker_library=str(args.tracker_library.resolve()), tracker_library_sha256=sha256(args.tracker_library),
        gmc_implementation_sha256=sha256(ROOT / "scripts/anti_uav/efficient_gmc.py"),
        tracker_source_sha256=sha256(ROOT / "scripts/anti_uav/dist_native" /
            ("motion_tracker.cpp" if args.tracker_kind == "motion" else "tracker.cpp")),
        adapter_sha256=sha256(Path(__file__)),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    if args.cached_gmc:
        provenance["gmc_cache_sha256"] = sha256(args.baseline_dir / "tracks.jsonl")
    protocol = dict(config=config, provenance=provenance, detector_cache_sha256=sha256(cache),
        gmc=("EfficientGMC(width=320,corners=128,refresh=5,resize_first=True)" if args.gmc_mode=="estimate"
            else "Disabled; identity transform; image-coordinate motion; no camera uncertainty" if args.gmc_mode=="disabled"
            else "Unavailable; identity transform; retain configured unknown-camera uncertainty"),
        gmc_mode=args.gmc_mode,
        tracker_implementation=("native C++ motion-aware-v1; NOT public Dist/OC-SORT parity" if
            args.tracker_kind == "motion" else "native C++ Dist; no ReID or score fusion"),
        ground_truth_used=False, inference_rerun=False, predicted_boxes_shown=False,
        id_remapping=False, frame_stride=1, fps=fps, frames=count,
        cached_gmc=args.cached_gmc,
        observation_output_contract=("all current detections; null ID until confirmed; no predictions/remapping"
            if args.tracker_kind=="motion" else "legacy confirmed-only"),
        scope="Offline server visualization; not board throughput or labelled tracking accuracy")
    dump(args.output / "protocol.json", protocol)
    cap = None if args.cached_gmc or args.gmc_mode!="estimate" else cv2.VideoCapture(str(args.source))
    if cap is not None and not cap.isOpened():
        raise ValueError("Cannot open source video")
    gmc = EfficientGMC(320, 128, 5, True)
    tracks, identities = [], set()
    association_times = []
    started = time.monotonic()
    try:
        with (args.output / "tracks.jsonl").open("x") as stream:
            for index, row in enumerate(records):
                boxes = np.asarray(row["boxes_xyxy_score"], dtype=np.float32).reshape(-1, 5)
                if args.gmc_mode!="estimate":
                    warp = np.eye(2, 3, dtype=np.float64)
                    quality, gmc_meta = 0., dict(estimated=False, support=0, inlier_ratio=0., disabled=True)
                elif args.cached_gmc:
                    warp = np.asarray(baseline[index]["warp"], dtype=np.float64)
                    quality = baseline[index]["gmc_quality"]
                    gmc_meta = baseline[index]["gmc_meta"]
                else:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Source ended at {index}, expected {count}")
                    warp = gmc.apply(frame)
                    quality, gmc_meta = gmc.last_quality, gmc.last_meta
                if baseline is not None:
                    ref = baseline[index]
                    if (ref["frame_index"] != index or ref["boxes_xyxy_score"] != row["boxes_xyxy_score"]
                            or (args.gmc_mode=="estimate" and not np.array_equal(warp, np.asarray(ref["warp"])) )):
                        raise ValueError(f"Baseline inputs/GMC changed at {index}")
                association_started = time.perf_counter()
                outputs = (tracker.update(boxes, warp, quality, index/fps) if
                           args.tracker_kind == "motion" else tracker.update(boxes, warp))
                association_times.append((time.perf_counter()-association_started)*1000)
                shown = [dict(t, confirmed=True, predicted=False) for t in outputs]
                record = dict(frame_index=index, time_seconds=index/fps, raw_tracks=shown,
                    displayed_tracks=shown, boxes_xyxy_score=row["boxes_xyxy_score"], warp=warp.tolist(),
                    gmc_quality=quality, gmc_meta=gmc_meta)
                if args.tracker_kind=="motion":
                    record["observations"] = tracker.observations()
                stream.write(json.dumps(record) + "\n")
                tracks.append(record)
                identities.update(t["id"] for t in shown)
                if (index+1) % 300 == 0 or index+1 == count:
                    status = dict(stage="tracking", frames=index+1, total=count,
                        seconds=round(time.monotonic()-started, 2))
                    dump(args.output / "status.json", status)
                    print(json.dumps(status), flush=True)
            if cap is not None and cap.read()[0]:
                raise ValueError("Source exceeds detector cache frame count")
        validate_records(records, tracks, count, fps)
        if sha256(cache) != protocol["detector_cache_sha256"]:
            raise ValueError("Detector cache changed during tracking")
        if args.cached_gmc and sha256(args.baseline_dir / "tracks.jsonl") != provenance["gmc_cache_sha256"]:
            raise ValueError("GMC cache changed during tracking")
        summary = dict(protocol, gmc_counts=original["gmc_counts"] if args.cached_gmc else gmc.counts,
            gmc_disabled_frames=count if args.gmc_mode!="estimate" else 0,
            displayed_tracks=sum(len(t["displayed_tracks"]) for t in tracks),
            frames_with_track=sum(bool(t["displayed_tracks"]) for t in tracks),
            visible_ids=len(identities), maximum_visible_id=max(identities, default=0),
            tracks_sha256=sha256(args.output / "tracks.jsonl"),
            association_bridge_mean_ms=statistics.mean(association_times),
            association_bridge_p95_ms=float(np.percentile(association_times, 95)),
            timing_scope="Server CPU association plus ctypes/output conversion; NOT board pipeline FPS",
            seconds=round(time.monotonic()-started, 2))
        if args.tracker_kind == "motion":
            summary["motion_stats"] = tracker.stats()
            summary["observation_boxes"] = sum(len(t["observations"]) for t in tracks)
            summary["unconfirmed_observations"] = sum(not o["confirmed"] for t in tracks for o in t["observations"])
            summary["frames_with_observations"] = sum(bool(t["observations"]) for t in tracks)
        dump(args.output / "summary.json", summary)
        dump(args.output / "status.json", dict(stage="complete", frames=count))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        dump(args.output / "status.json", dict(stage="failed", error=repr(error)))
        raise
    finally:
        if cap is not None:
            cap.release()
        tracker.close()


if __name__ == "__main__":
    main()
