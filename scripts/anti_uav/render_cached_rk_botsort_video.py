#!/usr/bin/env python3
"""Replay full-frame detector results through the board C++ tracker and render IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.render_pt_detector_video import dump, panel, probe, sha256
from scripts.anti_uav.run_causal_roi_comparison import NativeTracker


def load_records(path, count, fps):
    records = []
    with path.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row["frame_index"] != index or abs(row["time_seconds"] - index / fps) > 1e-6:
                raise ValueError(f"Nonsequential prediction cache at frame {index}")
            boxes = np.asarray(row["boxes_xyxy_score"], dtype=np.float32)
            if boxes.size == 0:
                boxes = np.empty((0, 5), dtype=np.float32)
            if boxes.ndim != 2 or boxes.shape[1] != 5 or not np.isfinite(boxes).all():
                raise ValueError(f"Invalid prediction boxes at frame {index}")
            if len(boxes) and (np.any(boxes[:, 2:4] <= boxes[:, :2]) or
                               np.any((boxes[:, 4] < 0) | (boxes[:, 4] > 1))):
                raise ValueError(f"Invalid box extent or confidence at frame {index}")
            records.append(boxes)
    if len(records) != count:
        raise ValueError(f"Cache contains {len(records)} frames, expected {count}")
    return records


def confirmed_observed(tracks):
    return sorted((t for t in tracks if t["confirmed"] and not t["predicted"]),
                  key=lambda t: (-t["score"], t["id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, help="Explicit prefix for smoke testing only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("Frame limit must be positive")
    summary_path = args.detector_dir / "summary.json"
    detection_summary = json.loads(summary_path.read_text())
    if (not detection_summary["full_source_video"] or detection_summary["frame_stride"] != 1
            or detection_summary["ground_truth_used"] or detection_summary["tracker_used"]):
        raise ValueError("A complete detector-only, non-GT-guided cache is required")
    source = Path(detection_summary["source_video"])
    if sha256(source) != detection_summary["source_sha256"]:
        raise ValueError("Source video hash differs from the detector run")
    info = probe(source)
    if info != detection_summary["source_info"]:
        raise ValueError("Source video metadata changed")
    fps = float(detection_summary["output_fps"])
    count = int(info["nb_frames"])
    if count != detection_summary["frames"] or fps <= 0:
        raise ValueError("Invalid frame count or FPS")
    predictions_path = args.detector_dir / "predictions.jsonl"
    records = load_records(predictions_path, count, fps)
    if sum(map(len, records)) != detection_summary["total_detections"]:
        raise ValueError("Detection counts differ from the completed run")
    if sum(bool(len(b)) for b in records) != detection_summary["frames_with_detections"]:
        raise ValueError("Detected-frame count differs from the completed run")
    limit = min(count, args.max_frames) if args.max_frames else count
    args.output.mkdir(parents=True)
    library = args.output / "librk_tracker.so"
    native_dir = ROOT / "scripts/anti_uav/rknn_yolov8_native"
    subprocess.run(["g++", "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC",
                    str(native_dir / "tracker_c_api.cpp"), "-o", str(library)], check=True)
    tracker = NativeTracker(library, fps)
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError("Could not open source video")
    destination = args.output / "Video00009_expanded28_frozenP3_addonP2_RKBoTSORT_conf003.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", "1600x784", "-r", info["avg_frame_rate"],
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(destination)]
    protocol = dict(
        source_video=str(source), source_sha256=detection_summary["source_sha256"],
        source_info=info, detector_summary=str(summary_path),
        detector_summary_sha256=sha256(summary_path), predictions_sha256=sha256(predictions_path),
        weights=detection_summary["weights"], weights_sha256=detection_summary["weights_sha256"],
        input_hw=detection_summary["input_hw"], conf=detection_summary["conf"],
        nms_iou=detection_summary["nms_iou"], precision=detection_summary["precision"],
        exact_detector_cache_reused=True, frame_stride=1, ground_truth_used=False,
        full_source_video=args.max_frames is None, output_fps=fps,
        tracker=dict(name="RK-BoT-SORT", implementation="board C++ via NativeTracker bridge",
                     high=.03, low=.01, birth=.10, first_match_cost=.92, second_match_cost=.92,
                     buffer_sec=1., prediction_sec=0., min_hits=3, fps=fps,
                     confirmed_only=True, predicted_outputs_displayed=False, pose_input=False),
        tracker_source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (
            native_dir / "detector_based_tracker.hpp", native_dir / "tracker_c_api.cpp",
            ROOT / "scripts/anti_uav/run_causal_roi_comparison.py")},
        note="Exact conf=0.03 detector comparison; 0.01-0.03 detections are not available for low-score association. Playback FPS is not board throughput.")
    dump(args.output / "protocol.json", protocol)
    started = time.monotonic()
    frame_count = total_tracks = frames_with_tracks = raw_count = 0
    identities = set()
    title = "28-video Frozen-P3 + Add-on P2 + RK-BoT-SORT | 960x544 | conf 0.03"
    encoder = None
    try:
        with (args.output / "encode.log").open("x") as log, (args.output / "tracks.jsonl").open("x") as output:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            for index in range(limit):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video ended before frame {index}")
                raw = tracker.update(records[index], index / fps, info["width"], info["height"])
                tracks = confirmed_observed(raw)
                boxes = np.asarray([(*t["box"], t["score"]) for t in tracks], dtype=np.float32).reshape(-1, 5)
                rendered = panel(frame, boxes, index, limit, fps, title, tracks=tracks)
                encoder.stdin.write(rendered.tobytes())
                output.write(json.dumps(dict(frame_index=index, time_seconds=index/fps,
                                             raw_tracks=raw, displayed_tracks=tracks)) + "\n")
                if index in (0, min(limit-1, 300), min(limit-1, 4000), min(limit-1, 9000)):
                    if not cv2.imwrite(str(args.output / f"preview_{index:06d}.jpg"), rendered):
                        raise RuntimeError("Failed to write preview")
                frame_count += 1
                total_tracks += len(tracks)
                raw_count += len(raw)
                frames_with_tracks += bool(tracks)
                identities.update(t["id"] for t in tracks)
                if frame_count % 512 == 0 or frame_count == limit:
                    progress = dict(stage="rendering", frames=frame_count, total=limit,
                                    seconds=round(time.monotonic()-started, 2))
                    dump(args.output / "status.json", progress)
                    print(json.dumps(progress), flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Encoder failed; see encode.log")
        video_info = probe(destination)
        if int(video_info["nb_frames"]) != limit or abs(float(video_info["duration"]) - limit/fps) > .02:
            raise AssertionError("Output frame count or duration mismatch")
        result = dict(protocol, video=str(destination), frames=frame_count, output_info=video_info,
                      total_displayed_tracks=total_tracks, frames_with_tracks=frames_with_tracks,
                      raw_track_outputs=raw_count, unique_displayed_ids=len(identities),
                      output_sha256=sha256(destination), seconds=round(time.monotonic()-started, 2))
        dump(args.output / "summary.json", result)
        dump(args.output / "status.json", dict(stage="complete", frames=frame_count, video=str(destination)))
        print(json.dumps(result), flush=True)
    except BaseException as error:
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        dump(args.output / "status.json", dict(stage="failed", frames=frame_count, error=repr(error)))
        raise
    finally:
        tracker.close()
        cap.release()


if __name__ == "__main__":
    main()
