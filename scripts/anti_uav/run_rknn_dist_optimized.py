#!/usr/bin/env python3
"""Bounded decode -> parallel RKNN / ordered live GMC -> ordered Dist pipeline."""
import time
ENTRY = time.perf_counter()
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import cv2
import numpy as np
from dist_numpy_runtime import CONFIG, as_results, load_dist, observations, source_hashes
from efficient_gmc import EfficientGMC, SuppliedWarp
from run_rknn_dist_pipeline import Detector, hardware, stats


def bind(cpu):
    os.sched_setaffinity(0, {cpu})


def bind_shared(cpus):
    os.sched_setaffinity(0, set(cpus))


def flow_init(cpus):
    os.sched_setaffinity(0, set(cpus))
    cv2.setRNGSeed(20260917)


def make_detectors(args, masks):
    if args.contexts == "independent":
        return [Detector(args.library, args.model, core, 1, args.conf, args.iou,
                         cached=args.preprocess == "cached") for core in masks]
    lib = ct.CDLL(str(args.library.resolve()))
    lib.au_detector_create_pool.argtypes = [ct.c_char_p, ct.POINTER(ct.c_char_p),
                                           ct.c_int, ct.c_int, ct.POINTER(ct.c_void_p)]
    lib.au_detector_create_pool.restype = ct.c_int
    lib.au_detector_error.restype = ct.c_char_p
    cores = (ct.c_char_p * len(masks))(*(s.encode() for s in masks))
    handles = (ct.c_void_p * len(masks))()
    if lib.au_detector_create_pool(os.fsencode(args.model), cores, len(masks), 1, handles):
        raise RuntimeError(lib.au_detector_error().decode())
    return [Detector(args.library, args.model, core, 1, args.conf, args.iou,
                     handle=handles[i], cached=args.preprocess == "cached") for i,core in enumerate(masks)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "upstream", "video", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1,2,3), default=3)
    p.add_argument("--core-mode", choices=("split","all"), default="split")
    p.add_argument("--contexts", choices=("shared","independent"), default="shared")
    p.add_argument("--preprocess", choices=("native","cached"), default="cached")
    p.add_argument("--cpus", default="4,5,6,7")
    p.add_argument("--worker-affinity", choices=("pinned", "shared"), default="pinned")
    p.add_argument("--dispatch", choices=("round-robin", "ready"), default="round-robin")
    p.add_argument("--inflight", type=int, default=4)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--conf", type=float, default=.03)
    p.add_argument("--iou", type=float, default=.45)
    p.add_argument("--detector-only", action="store_true")
    p.add_argument("--gmc", choices=("public","compact"), default="public")
    p.add_argument("--gmc-width", type=int, default=480)
    p.add_argument("--gmc-corners", type=int, default=256)
    p.add_argument("--gmc-refresh", type=int, default=1)
    p.add_argument("--gmc-resize-first", action="store_true")
    p.add_argument("--save-observations", action="store_true")
    args = p.parse_args()
    cpus = [int(x) for x in args.cpus.split(",")]
    if not cpus or args.inflight < args.workers or args.warmup < 0 or args.frames < 0:
        p.error("Invalid CPUs/frame count; inflight must be at least workers")
    os.sched_setaffinity(0, set(cpus))
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    before_hw = hardware()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("Video open failed")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = args.frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= args.warmup:
        raise ValueError("Need known video FPS and more frames than warmup")
    source_shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    tracker = flow = supplied = None
    if not args.detector_only:
        tracker = load_dist(args.upstream)(SimpleNamespace(**CONFIG), frame_rate=fps)
        flow = tracker.gmc if args.gmc == "public" else EfficientGMC(args.gmc_width, args.gmc_corners, args.gmc_refresh, args.gmc_resize_first)
        supplied = SuppliedWarp()
        tracker.gmc = supplied
    masks = [str(i) if args.core_mode == "split" else "all" for i in range(args.workers)]
    loading = time.perf_counter()
    detectors = make_detectors(args, masks)
    load_ms = (time.perf_counter()-loading)*1000
    npu_pools = [ThreadPoolExecutor(max_workers=1,
                    initializer=bind if args.worker_affinity == "pinned" else bind_shared,
                    initargs=(cpus[i % len(cpus)],) if args.worker_affinity == "pinned" else (cpus,))
                 for i in range(args.workers)]
    flow_pool = ThreadPoolExecutor(max_workers=1, initializer=flow_init, initargs=(cpus,)) if flow else None
    ready_ms = (time.perf_counter()-ENTRY)*1000
    slots, results, stop = threading.Semaphore(args.inflight), queue.Queue(), threading.Event()
    available = queue.Queue()
    for worker in range(args.workers):
        available.put(worker)
    assigned = [0] * args.workers

    def run_flow(frame):
        start = time.perf_counter()
        warp = flow.apply(frame)
        if np.shape(warp) != (2,3) or not np.isfinite(warp).all():
            raise RuntimeError("Invalid GMC warp; no silent benchmark fallback")
        return warp, (time.perf_counter()-start)*1000

    def produce():
        try:
            for index in range(count):
                while not slots.acquire(timeout=.1):
                    if stop.is_set():
                        return
                if stop.is_set():
                    slots.release()
                    return
                start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    slots.release()
                    raise RuntimeError(f"Decode failed at frame {index}")
                decode_ms = (time.perf_counter()-start)*1000
                worker = index % args.workers
                if args.dispatch == "ready":
                    while not stop.is_set():
                        try:
                            worker = available.get(timeout=.1)
                            break
                        except queue.Empty:
                            continue
                    else:
                        slots.release()
                        return
                detected = npu_pools[worker].submit(detectors[worker].infer, frame)
                assigned[worker] += 1
                if args.dispatch == "ready":
                    detected.add_done_callback(lambda future, worker=worker: available.put(worker))
                motion = flow_pool.submit(run_flow, frame) if flow_pool else None
                results.put((index, frame, start, decode_ms, detected, motion))
        except BaseException as error:
            results.put(error)
        finally:
            results.put(None)

    stream = (args.output / "observations.jsonl").open("x") if args.save_observations else None
    producer = threading.Thread(target=produce, name="decode", daemon=True)
    start = time.perf_counter()
    producer.start()
    measured_start = start if args.warmup == 0 else None
    timing = {name:[] for name in ("decode", "preprocess", "npu", "postprocess", "gmc", "association", "ordered_latency")}
    samples, first, processed, det_count, track_count, invalid = [], None, 0, 0, 0, 0
    try:
        while True:
            item = results.get()
            if isinstance(item, BaseException):
                raise item
            if item is None:
                break
            index, frame, began, decode_ms, detected, motion = item
            if index != processed:
                raise RuntimeError("Non-causal result ordering")
            boxes, native_ms, bad = detected.result()
            warp, flow_ms = motion.result() if motion else (None,0.)
            association_start = time.perf_counter()
            shown = []
            if tracker:
                supplied.put(warp)
                returned = tracker.update(as_results(boxes), img=frame)
                shown = observations(tracker, boxes)
                if supplied.warp is not None or len(returned) != len(shown):
                    raise RuntimeError("Live GMC or observation association mismatch")
            end_frame = time.perf_counter()
            association_ms, latency = (end_frame-association_start)*1000, (end_frame-began)*1000
            if processed == 0:
                first = dict(process_entry_ms=(end_frame-ENTRY)*1000, frame_read_ms=latency,
                             decode_ms=decode_ms, npu_ms=native_ms.tolist(), gmc_ms=flow_ms,
                             association_ms=association_ms)
                print(json.dumps(dict(event="first_result", **first)), flush=True)
            if processed >= args.warmup:
                values = (decode_ms, *native_ms, flow_ms, association_ms, latency)
                for key, value in zip(timing, values):
                    timing[key].append(float(value))
            if stream:
                stream.write(json.dumps(dict(frame_index=index, boxes_xyxy_score=boxes.tolist(),
                    displayed_tracks=shown, warp=None if warp is None else np.asarray(warp).tolist()))+"\n")
            processed += 1
            det_count += len(boxes)
            track_count += len(shown)
            invalid += bad
            slots.release()
            if processed == args.warmup:
                measured_start = time.perf_counter()
            if processed % 500 == 0:
                snapshot = hardware()
                samples.append(dict(frame=processed, elapsed_seconds=time.perf_counter()-start, hardware=snapshot))
                print(json.dumps(dict(event="progress", frame=processed, elapsed=time.perf_counter()-start)), flush=True)
        ended = time.perf_counter()
        if processed != count or measured_start is None:
            raise RuntimeError("Incomplete video processing")
        report = dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            frames=processed, measured_frames=processed-args.warmup,
            measured_seconds=ended-measured_start, steady_fps=(processed-args.warmup)/(ended-measured_start),
            all_frames_fps=processed/(ended-start), first_result=first, model_load_ms=load_ms,
            initialization_to_models_ready_ms=ready_ms, stages_ms={k:stats(v) for k,v in timing.items()},
            input_wh=[960,544], source_wh=source_shape, source_fps=fps,
            npu_core_masks=masks, detector_count=det_count, displayed_tracks=track_count,
            npu_worker_frame_counts=assigned,
            rejected_degenerate_boxes=invalid, tracker_config=CONFIG if tracker else None,
            compact_gmc_counts=flow.counts if isinstance(flow,EfficientGMC) else None,
            compact_gmc_seconds=flow.seconds if isinstance(flow,EfficientGMC) else None,
            hardware_before=before_hw, hardware_after=hardware(), hardware_samples=samples,
            model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
            upstream_files=source_hashes(args.upstream),
            scope="Actual video decode + C++ RKNN preprocessing/inference/postprocessing + per-frame live GMC + ordered public Dist association. Bounded pipeline, no frame skipping, no cached detections/warps, no rendering/encoding. Compact GMC is explicitly an algorithm variant; public retains the upstream algorithm.")
        (args.output / "summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(dict(event="complete", frames=processed, fps=report["steady_fps"])), flush=True)
    finally:
        stop.set()
        producer.join(timeout=30)
        if producer.is_alive():
            raise RuntimeError("Decoder did not stop")
        for pool in npu_pools:
            pool.shutdown(wait=True, cancel_futures=True)
        if flow_pool:
            flow_pool.shutdown(wait=True, cancel_futures=True)
        for detector in detectors:
            detector.close()
        cap.release()
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
