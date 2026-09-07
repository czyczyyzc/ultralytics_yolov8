#!/usr/bin/env python3
"""Run sequential PT detection with the actual C++ RK-BoT-SORT and causal ROI."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.causal_roi_policy import choose_region, restore_boxes


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class NativeTracker:
    def __init__(self, library, fps):
        import numpy as np
        self.lib = ctypes.CDLL(str(library))
        self.lib.rk_tracker_create.argtypes = [ctypes.c_double] * 8 + [ctypes.c_int]
        self.lib.rk_tracker_create.restype = ctypes.c_void_p
        self.lib.rk_tracker_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rk_tracker_error.restype = ctypes.c_char_p
        self.lib.rk_tracker_update.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_double, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ]
        self.handle = self.lib.rk_tracker_create(.03, .01, .10, .92, .92, 1., 0., fps, 3)
        if not self.handle:
            raise RuntimeError(self.lib.rk_tracker_error().decode())
        self.output = np.zeros((4096, 12), dtype=np.float64)

    def close(self):
        if self.handle:
            self.lib.rk_tracker_destroy(self.handle)
            self.handle = None

    def update(self, boxes, timestamp, width, height):
        import numpy as np
        detections = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
        count = self.lib.rk_tracker_update(
            self.handle, detections.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(detections), timestamp, width, height,
            self.output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(self.output),
        )
        if count < 0:
            raise RuntimeError(self.lib.rk_tracker_error().decode())
        return [dict(id=int(t[0]), box=t[1:5].tolist(), score=float(t[5]),
                     confirmed=bool(t[6]), predicted=bool(t[7]), age=int(t[8]),
                     hits=int(t[9]), time_since_update_sec=float(t[10]))
                for t in self.output[:count]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--zoom", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--refresh-interval", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(20260907)
    torch.backends.cudnn.benchmark = False
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    args.output.mkdir(parents=True)
    library = args.output / "librk_tracker.so"
    subprocess.run(["g++", "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC",
                    str(ROOT / "scripts/anti_uav/rknn_yolov8_native/tracker_c_api.cpp"),
                    "-o", str(library)], check=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError("A valid source FPS is required for causal tracking")
    tracker = NativeTracker(library, fps)
    model = YOLO(str(args.model))
    predict_options = dict(imgsz=[544, 960], conf=.01, iou=.45, max_det=100,
                           device=args.device, verbose=False, rect=False, half=False)
    for _ in range(10):
        model.predict(np.zeros((1080, 1920, 3), dtype=np.uint8), **predict_options)
    nframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    previous_tracks, anchor = [], None
    records = []
    measured_seconds = 0.
    try:
        with (args.output / "frames.jsonl").open("w") as handle:
            for index in range(nframes if args.max_frames <= 0 else min(nframes, args.max_frames)):
                start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Unexpected decode failure at frame {index}")
                height, width = frame.shape[:2]
                region, mode, anchor = choose_region(
                    previous_tracks, index, width, height, args.zoom,
                    args.refresh_interval, anchor)
                x1, y1, x2, y2 = region
                crop = frame[y1:y2, x1:x2]
                t1 = time.perf_counter()
                result = model.predict(crop, **predict_options)[0]
                boxes = result.boxes.data[:, :5].detach().cpu().numpy().tolist()
                boxes = restore_boxes(boxes, region, width, height)
                t2 = time.perf_counter()
                raw_tracks = tracker.update(boxes, index / fps, width, height)
                previous_tracks = raw_tracks
                tracks = [t for t in raw_tracks if t["confirmed"] and not t["predicted"]]
                end = time.perf_counter()
                # Exclude JSON/video rendering I/O, include decode, ROI, YOLO and tracker.
                measured_seconds += end - start
                record = dict(frame=index, timestamp_sec=index/fps, mode=mode,
                              roi=list(region), anchor_id=anchor,
                              detections=boxes, tracks=tracks,
                              inference_pre_post_ms=(t2-t1)*1000,
                              tracking_ms=(end-t2)*1000,
                              pipeline_ms=(end-start)*1000)
                records.append(record)
                handle.write(json.dumps(record) + "\n")
                if (index + 1) % 250 == 0:
                    handle.flush()
                    print(f"{args.name}: {index+1}/{nframes} frames", flush=True)
    finally:
        capture.release()
        tracker.close()
    tracked = [r["pipeline_ms"] for r in records]
    manifest = dict(
        schema="anti_uav.causal_roi_comparison.v1", name=args.name,
        runtime="A100_PyTorch_FP32_and_native_CPP_RK_BoT_SORT",
        model=str(args.model.resolve()), model_sha256=sha256(args.model),
        source=str(args.video.resolve()), source_sha256=sha256(args.video),
        source_size=[width, height], source_fps=fps, frames=len(records),
        input_size_wh=[960, 544], conf=.01, nms_iou=.45, max_det=100,
        padding="Ultralytics default 114, fixed rect=False for every arm",
        zoom=args.zoom, refresh_interval=args.refresh_interval,
        feedback="confirmed observation-backed previous frame track; no GT; sticky ID then hits/score",
        inference_calls=len(records), full_frames=sum(r["mode"] != "roi" for r in records),
        roi_frames=sum(r["mode"] == "roi" for r in records),
        tracker=dict(high=.03, low=.01, new=.10, first_match=.92, second_match=.92,
                     buffer_sec=1., prediction_sec=0., min_hits=3, confirmed_only=True),
        tracker_source_sha256=sha256(ROOT / "scripts/anti_uav/rknn_yolov8_native/detector_based_tracker.hpp"),
        server_serial_fps=len(records)/measured_seconds,
        pipeline_mean_ms=float(np.mean(tracked)), pipeline_p95_ms=float(np.percentile(tracked,95)),
        tracker_mean_ms=float(np.mean([r["tracking_ms"] for r in records])),
        timing_note="Batch 1, sequential FP32. Includes video read; excludes model load, ten synthetic warmups and output I/O. Not RK3588 FPS.",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
