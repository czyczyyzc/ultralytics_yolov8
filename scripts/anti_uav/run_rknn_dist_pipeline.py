#!/usr/bin/env python3
"""Native INT8 RKNN detection + causal public Dist/GMC, with bounded ordered NPU workers."""
import time
ENTRY_TIME = time.perf_counter()

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace

import cv2
import numpy as np

from dist_numpy_runtime import CONFIG, as_results, load_dist, observations, source_hashes


class Detector:
    def __init__(self, library, model, core, threads, conf, iou):
        self.lib = ct.CDLL(str(library))
        self.lib.au_detector_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int]
        self.lib.au_detector_create.restype = ct.c_void_p
        self.lib.au_detector_destroy.argtypes = [ct.c_void_p]
        self.lib.au_detector_error.restype = ct.c_char_p
        self.lib.au_detector_infer.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int,
            ct.c_int, ct.c_size_t, ct.c_float, ct.c_float, ct.c_int, ct.c_void_p, ct.c_void_p]
        self.lib.au_detector_infer.restype = ct.c_int
        self.handle = self.lib.au_detector_create(os.fsencode(model), core.encode(), threads)
        if not self.handle:
            raise RuntimeError(self.lib.au_detector_error().decode())
        self.conf, self.iou = conf, iou

    def infer(self, frame):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected uint8 BGR frame")
        frame = np.ascontiguousarray(frame)
        boxes, stages = np.empty((100, 5), np.float32), np.empty(3, np.float64)
        count = self.lib.au_detector_infer(self.handle, frame.ctypes.data,
            frame.shape[1], frame.shape[0], frame.strides[0], self.conf, self.iou, 100,
            boxes.ctypes.data, stages.ctypes.data)
        if count < 0:
            raise RuntimeError(self.lib.au_detector_error().decode())
        found = boxes[:count].copy()
        # Clipping at the image border can create zero-area boxes, which have no observation.
        valid = np.all(found[:, 2:4] > found[:, :2], axis=1)
        return found[valid], stages, int((~valid).sum())

    def close(self):
        if self.handle:
            self.lib.au_detector_destroy(self.handle)
            self.handle = None


def hardware():
    data = dict(host=platform.node(), kernel=platform.release())
    for pattern in ("/sys/class/devfreq/*/cur_freq", "/sys/class/devfreq/*/governor",
                    "/sys/class/thermal/thermal_zone*/temp"):
        import glob
        for name in glob.glob(pattern):
            try:
                data[name] = Path(name).read_text().strip()
            except OSError:
                pass
    return data


def stats(values):
    array = np.asarray(values, float)
    return dict(mean=float(array.mean()), p50=float(np.percentile(array, 50)),
                p95=float(np.percentile(array, 95)), max=float(array.max())) if len(array) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--core", default="0", choices=("0", "1", "2", "all"))
    p.add_argument("--frames", type=int, default=0, help="0: full video")
    p.add_argument("--warmup", type=int, default=50, help="Exclude initial outputs from steady FPS, not startup")
    p.add_argument("--cv-threads", type=int, default=2)
    p.add_argument("--conf", type=float, default=.03)
    p.add_argument("--iou", type=float, default=.45)
    p.add_argument("--detector-only", action="store_true")
    p.add_argument("--save-observations", action="store_true")
    p.add_argument("--preload", type=int, default=0,
                   help="Repeat this many predecoded frames; decoder-excluded synthetic benchmark only")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    if args.warmup < 0 or args.frames < 0 or args.preload < 0:
        p.error("Frame counts must be nonnegative")
    args.output.mkdir(parents=True)
    cv2.setNumThreads(args.cv_threads)
    cv2.setRNGSeed(20260917)
    initial_hardware = hardware()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Video FPS must be known for the tracker buffer")
    limit = args.frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = None
    if not args.detector_only:
        tracker = load_dist(args.upstream)(SimpleNamespace(**CONFIG), frame_rate=fps)
    load_start = time.perf_counter()
    detectors, pools, stream = [], [], None
    try:
        for i in range(args.workers):
            detectors.append(Detector(args.library, args.model, str(i) if args.workers > 1 else args.core,
                                      args.cv_threads, args.conf, args.iou))
            pools.append(ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"npu{i}"))
        model_load_ms = (time.perf_counter() - load_start) * 1000
        ready_at = time.perf_counter()
        preloaded = []
        for _ in range(args.preload):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Video too short for preload")
            preloaded.append(frame)
        if args.save_observations:
            stream = (args.output / "observations.jsonl").open("x")
        pending = deque()
        submitted = processed = total_detections = total_tracks = invalid = 0
        decode_ms, stages, tracker_ms, latencies = [], [], [], []
        first_result = first_frame_latency = first_visible = first_visible_frame = None
        first_stages = None
        steady_start = None
        loop_start = time.perf_counter()
        if args.warmup == 0:
            steady_start = loop_start
        while processed < limit:
            # The first result is not delayed by filling the parallel queue.
            capacity = 1 if processed == 0 else args.workers * 2
            while submitted < limit and len(pending) < capacity:
                begin = time.perf_counter()
                if preloaded:
                    frame = preloaded[submitted % len(preloaded)]
                else:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Decode failed before expected frame {submitted}")
                decoded = time.perf_counter()
                worker = submitted % args.workers
                future = pools[worker].submit(detectors[worker].infer, frame)
                pending.append((submitted, frame, begin, (decoded-begin)*1000, future))
                submitted += 1
            index, frame, begin, decode_time, future = pending.popleft()
            if index != processed:
                raise RuntimeError("Out-of-order frame")
            boxes, native_ms, bad = future.result()
            track_start = time.perf_counter()
            shown = []
            if tracker is not None:
                returned = tracker.update(as_results(boxes), img=frame)
                shown = observations(tracker, boxes)
                if len(shown) != len(returned):
                    raise RuntimeError("Upstream returned/current-track mismatch")
            finished = time.perf_counter()
            track_time = (finished-track_start)*1000
            latency = (finished-begin)*1000
            if processed == 0:
                first_result = (finished-ENTRY_TIME)*1000
                first_frame_latency = latency
                first_stages = dict(decode_ms=decode_time, native_ms=native_ms.tolist(), tracker_gmc_ms=track_time)
                print(json.dumps(dict(event="first_result", frame=0, detections=len(boxes),
                    tracks=len(shown), python_entry_to_result_ms=first_result,
                    frame_to_result_ms=latency)), flush=True)
            if shown and first_visible is None:
                first_visible, first_visible_frame = (finished-ENTRY_TIME)*1000, index
            if processed >= args.warmup:
                decode_ms.append(decode_time)
                stages.append(native_ms)
                tracker_ms.append(track_time)
                latencies.append(latency)
            if stream:
                stream.write(json.dumps(dict(frame_index=index, boxes_xyxy_score=boxes.tolist(),
                                              displayed_tracks=shown))+"\n")
            processed += 1
            invalid += bad
            total_detections += len(boxes)
            total_tracks += len(shown)
            if processed == args.warmup:
                steady_start = time.perf_counter()
            if processed % 500 == 0:
                print(json.dumps(dict(event="progress", frames=processed,
                    elapsed_seconds=time.perf_counter()-loop_start)), flush=True)
        end = time.perf_counter()
        measured = processed-args.warmup
        if steady_start is None or measured <= 0:
            raise ValueError("Need more frames than warmup")
        native = np.asarray(stages)
        result = dict(protocol="RKNN native INT8 + actual public Dist BOTSORT/sparseOptFlow; no cached detections/warps",
            args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
            tracker_config=None if args.detector_only else CONFIG,
            upstream_files=source_hashes(args.upstream), source_fps=fps,
            source_resolution=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
            input_resolution=[960,544], padding_value=0,
            frames=processed, measured_frames=measured,
            measured_seconds=end-steady_start, steady_fps=measured/(end-steady_start),
            all_frames_fps=processed/(end-loop_start), total_detections=total_detections,
            total_displayed_tracks=total_tracks, rejected_degenerate_boxes=invalid,
            first_result_from_python_entry_ms=first_result,
            initialization_to_models_ready_ms=(ready_at-ENTRY_TIME)*1000,
            model_load_ms=model_load_ms, first_frame_read_to_result_ms=first_frame_latency,
            first_visible_track_from_entry_ms=first_visible, first_visible_track_frame=first_visible_frame,
            first_frame_stages=first_stages,
            stages_ms=dict(decode=stats(decode_ms), preprocess=stats(native[:,0]),
                npu_run=stats(native[:,1]), postprocess=stats(native[:,2]),
                tracker_gmc=stats(tracker_ms), frame_read_to_ordered_result=stats(latencies)),
            hardware_before=initial_hardware, hardware_after=hardware(),
            timing_scope=("Includes CPU video decode, letterbox/BGR-to-RGB, native NPU inference/DFL/NMS, "
                "ordered association and real causal GMC unless detector-only. No rendering/encoding/network/camera exposure. "
                "Startup starts at Python script entry, includes imports/load, but not OS exec. "
                "Preload mode excludes decode and is NOT full-video tracking evidence."))
        (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(dict(event="complete", frames=processed, steady_fps=result["steady_fps"],
                              summary=str(args.output / "summary.json"))), flush=True)
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
        for detector in detectors:
            detector.close()
        cap.release()
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
