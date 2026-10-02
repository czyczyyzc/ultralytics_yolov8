#!/usr/bin/env python3
"""Replay native Dist association plus efficient GMC on an unlabelled video cache."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.dist_numpy_runtime import CONFIG
from scripts.anti_uav.efficient_gmc import EfficientGMC
from scripts.anti_uav.native_dist_runtime import NativeDist
from scripts.anti_uav.render_cached_tracker_result_video import validate_records
from scripts.anti_uav.render_pt_detector_video import dump, probe, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--tracker-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
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
    if len(records) != count:
        raise ValueError("Detector cache does not cover every frame")
    for index, row in enumerate(records):
        if row["frame_index"] != index or abs(row["time_seconds"]-index/fps) > 1e-7:
            raise ValueError(f"Unordered detector frame/timestamp at {index}")
    args.output.mkdir(parents=True)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(20260924)
    config = dict(CONFIG)
    provenance = dict(source_sha256=detector["source_sha256"],
        weights_sha256=detector["weights_sha256"], detector_cache_sha256=sha256(cache),
        tracker_library=str(args.tracker_library.resolve()), tracker_library_sha256=sha256(args.tracker_library),
        gmc_implementation_sha256=sha256(ROOT / "scripts/anti_uav/efficient_gmc.py"),
        tracker_source_sha256=sha256(ROOT / "scripts/anti_uav/dist_native/tracker.cpp"),
        adapter_sha256=sha256(Path(__file__)),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    protocol = dict(config=config, provenance=provenance, detector_cache_sha256=sha256(cache),
        gmc="EfficientGMC(width=320,corners=128,refresh=5,resize_first=True)",
        tracker_implementation="native C++ Dist; no ReID or score fusion",
        ground_truth_used=False, inference_rerun=False, predicted_boxes_shown=False,
        id_remapping=False, frame_stride=1, fps=fps, frames=count,
        scope="Offline server visualization; not board throughput or labelled tracking accuracy")
    dump(args.output / "protocol.json", protocol)
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise ValueError("Cannot open source video")
    tracker = NativeDist(args.tracker_library, fps=fps, config=config)
    gmc = EfficientGMC(320, 128, 5, True)
    tracks, identities = [], set()
    started = time.monotonic()
    try:
        with (args.output / "tracks.jsonl").open("x") as stream:
            for index, row in enumerate(records):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Source ended at {index}, expected {count}")
                boxes = np.asarray(row["boxes_xyxy_score"], dtype=np.float32).reshape(-1, 5)
                warp = gmc.apply(frame)
                shown = [dict(t, confirmed=True, predicted=False) for t in tracker.update(boxes, warp)]
                record = dict(frame_index=index, time_seconds=index/fps, raw_tracks=shown,
                    displayed_tracks=shown, boxes_xyxy_score=row["boxes_xyxy_score"], warp=warp.tolist())
                stream.write(json.dumps(record) + "\n")
                tracks.append(record)
                identities.update(t["id"] for t in shown)
                if (index+1) % 300 == 0 or index+1 == count:
                    status = dict(stage="tracking", frames=index+1, total=count,
                        seconds=round(time.monotonic()-started, 2))
                    dump(args.output / "status.json", status)
                    print(json.dumps(status), flush=True)
            if cap.read()[0]:
                raise ValueError("Source exceeds detector cache frame count")
        validate_records(records, tracks, count, fps)
        if sha256(cache) != protocol["detector_cache_sha256"]:
            raise ValueError("Detector cache changed during tracking")
        summary = dict(protocol, gmc_counts=gmc.counts,
            displayed_tracks=sum(len(t["displayed_tracks"]) for t in tracks),
            frames_with_track=sum(bool(t["displayed_tracks"]) for t in tracks),
            visible_ids=len(identities), maximum_visible_id=max(identities, default=0),
            tracks_sha256=sha256(args.output / "tracks.jsonl"),
            seconds=round(time.monotonic()-started, 2))
        dump(args.output / "summary.json", summary)
        dump(args.output / "status.json", dict(stage="complete", frames=count))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        dump(args.output / "status.json", dict(stage="failed", error=repr(error)))
        raise
    finally:
        cap.release()
        tracker.close()


if __name__ == "__main__":
    main()
