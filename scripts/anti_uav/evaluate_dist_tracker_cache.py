#!/usr/bin/env python3
"""Evaluate the actual public Dist-Tracker code on fixed UAV detector observations."""
import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.analyze_cached_tracker_suppression import digest, evaluate, read_jsonl, require
from scripts.anti_uav.sweep_native_tracker_parameters import identity_counts


def as_results(boxes):
    array = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    require(np.isfinite(array).all(), "Non-finite detection")
    require(np.all(array[:, 2:4] > array[:, :2]), "Invalid box geometry")
    xywh = array[:, :4].copy()
    xywh[:, :2] = (array[:, :2] + array[:, 2:4]) / 2
    xywh[:, 2:4] = array[:, 2:4] - array[:, :2]
    return SimpleNamespace(conf=array[:, 4], cls=np.zeros(len(array), np.float32), xywh=xywh)


def recover_observation(track, boxes, frame_id):
    require(track.frame_id == frame_id, "Output is not a current observation")
    index = int(track.idx)
    require(index == track.idx and 0 <= index < len(boxes), "Invalid detector index")
    box = boxes[index]
    require(abs(float(track.score)-box[4]) < 1e-6, "Detection confidence mismatch")
    return dict(id=int(track.track_id), box=box[:4], score=box[4],
                kalman_box=track.xyxy.tolist(), confirmed=bool(track.is_activated),
                predicted=False, detection_index=index)


def load_ground_truth(manifest, coco_path):
    meta, coco = json.loads(manifest.read_text()), json.loads(coco_path.read_text())
    require(digest(coco_path) == next(f["sha256"] for f in meta["files"]
                                    if f["path"] == "coco/annotations.json"), "GT hash mismatch")
    included = set(meta["frames"]["includedFrameIndices"])
    for key in ("excludedUncertainFrameIndices", "excludedUnreviewedFrameIndices"):
        included -= set(meta["frames"].get(key, []))
    by_id = {im["id"]: im["frame_index"] for im in coco["images"]}
    require(set(by_id.values()) == included, "GT coverage mismatch")
    gt = defaultdict(list)
    for ann in coco["annotations"]:
        require(ann["category_id"] == 0 and not ann.get("iscrowd"), "Unexpected GT class")
        x, y, w, h = ann["bbox"]
        gt[by_id[ann["image_id"]]].append([x, y, x+w, y+h])
    require(all(len(v) <= 1 for v in gt.values()), "Identity diagnostic supports one GT per frame")
    return meta, gt, included


class CachedGMC:
    def __init__(self, warps):
        self.warps, self.index = warps, 0

    def apply(self, image, detections=None):
        require(self.index < len(self.warps), "GMC cache exhausted")
        warp = self.warps[self.index]
        self.index += 1
        return warp.copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--detector-dir", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--approved-manifest", type=Path, required=True)
    p.add_argument("--coco", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gmc-cache", type=Path)
    p.add_argument("--skip-gmc", action="store_true")
    args = p.parse_args()
    require(not args.output.exists(), "Use a fresh output directory")
    meta, gt, included = load_ground_truth(args.approved_manifest, args.coco)
    ds = json.loads((args.detector_dir / "summary.json").read_text())
    bs = json.loads((args.baseline / "summary.json").read_text())
    cache = args.detector_dir / "predictions.jsonl"
    detections = read_jsonl(cache)
    baseline = read_jsonl(args.baseline / "tracks.jsonl")
    require(bs["predictions_sha256"] == digest(cache), "Different baseline detections")
    require(ds["source_sha256"] == meta["video"]["sha256"] == digest(args.source), "Source mismatch")
    require(ds["weights_sha256"] == bs["weights_sha256"], "Model mismatch")
    require(ds["conf"] == .03 and ds["frame_stride"] == 1 and ds["full_source_video"], "Unexpected protocol")
    fps = meta["video"]["fps"]
    require(len(detections) == len(baseline) == meta["video"]["frameCount"] == ds["frames"], "Incomplete cache")
    require(all(d["frame_index"] == b["frame_index"] == i and abs(d["time_seconds"]-i/fps) < 1e-8
                for i,(d,b) in enumerate(zip(detections,baseline))), "Frame ordering mismatch")
    upstream = args.upstream.resolve()
    require("ultralytics" not in sys.modules, "Do not mix Ultralytics forks")
    sys.path.insert(0, str(upstream))
    bot = importlib.import_module("ultralytics.trackers.bot_sort")
    gmc_module = importlib.import_module("ultralytics.trackers.utils.gmc")
    matching = importlib.import_module("ultralytics.trackers.utils.matching")
    base_track = importlib.import_module("ultralytics.trackers.basetrack").BaseTrack
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(20260917)
    for module in (bot, gmc_module, matching):
        require(upstream in Path(module.__file__).resolve().parents, "Wrong upstream module imported")
    config_path = upstream / "ultralytics/cfg/trackers/botsort.yaml"
    import yaml
    defaults = yaml.safe_load(config_path.read_text())
    files = list((upstream / "ultralytics/trackers").rglob("*.py"))
    files += [upstream / "ultralytics/utils/metrics.py", config_path]
    provenance = dict(detector_cache_sha256=digest(cache), source_sha256=ds["source_sha256"],
        weights_sha256=ds["weights_sha256"], coco_sha256=digest(args.coco),
        upstream_commit=subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip(),
        upstream_files={str(f.relative_to(upstream)):digest(f) for f in sorted(files)},
        adapter_sha256=digest(Path(__file__)))
    args.output.mkdir(parents=True)
    warps, gmc_times = [], []
    if not args.skip_gmc:
        if args.gmc_cache:
            gp = json.loads((args.gmc_cache / "gmc_protocol.json").read_text())
            require(gp["source_sha256"] == ds["source_sha256"] and
                    gp["implementation_sha256"] == digest(Path(gmc_module.__file__)), "GMC provenance mismatch")
            require(gp["cache_sha256"] == digest(args.gmc_cache / "gmc.npz"), "GMC cache changed")
            with np.load(args.gmc_cache / "gmc.npz", allow_pickle=False) as data:
                warps, gmc_times = data["warps"], data["seconds"]
        else:
            cap = cv2.VideoCapture(str(args.source))
            gmc = gmc_module.GMC(method="sparseOptFlow")
            try:
                for i in range(len(detections)):
                    ok, image = cap.read()
                    require(ok, f"Decode failure at frame {i}")
                    start = time.perf_counter()
                    warp = gmc.apply(image)
                    gmc_times.append(time.perf_counter()-start)
                    require(warp is not None and warp.shape == (2,3) and np.isfinite(warp).all(),
                            f"Upstream GMC failed at frame {i}; no silent identity fallback")
                    warps.append(warp)
                    if i % 1000 == 0:
                        print(json.dumps(dict(stage="upstream_gmc",frame=i,total=len(detections))),flush=True)
            finally:
                cap.release()
            np.savez_compressed(args.output / "gmc.npz", warps=warps, seconds=gmc_times)
            gp = dict(source_sha256=ds["source_sha256"], frames=len(warps), downscale=2,
                method="unmodified upstream sparseOptFlow, causal previous/current images only",
                implementation_sha256=digest(Path(gmc_module.__file__)),
                cache_sha256=digest(args.output / "gmc.npz"), seconds=sum(gmc_times))
            (args.output / "gmc_protocol.json").write_text(json.dumps(gp,indent=2)+"\n")
        require(np.asarray(warps).shape == (len(detections),2,3), "Incomplete GMC cache")
    prepared = [as_results(r["boxes_xyxy_score"]) for r in detections]
    presets = [("repo_defaults",dict(defaults)),
               ("conf003_fused",dict(defaults,track_high_thresh=.03,track_low_thresh=.01,new_track_thresh=.10)),
               ("conf003_unfused",dict(defaults,track_high_thresh=.03,track_low_thresh=.01,
                                      new_track_thresh=.10,fuse_score=False))]
    results = []
    for use_gmc in ([False] if args.skip_gmc else [False,True]):
        for preset, config in presets:
            name = preset + ("_gmc" if use_gmc else "_no_gmc")
            folder = args.output / name
            folder.mkdir()
            config = dict(config,gmc_method="sparseOptFlow" if use_gmc else "none")
            tracker = bot.BOTSORT(SimpleNamespace(**config),frame_rate=fps)
            if use_gmc:
                tracker.gmc = CachedGMC(warps)
            frames, smooth, elapsed, visible_ids, all_ids = [], [], [], set(), set()
            with (folder / "tracks.jsonl").open("x") as stream:
                for i, det in enumerate(prepared):
                    start = time.perf_counter()
                    returned = tracker.update(det, img=np.empty((1,1,3),np.uint8) if use_gmc else None)
                    elapsed.append(time.perf_counter()-start)
                    current = [t for t in tracker.tracked_stracks if t.frame_id == i+1]
                    raw = [recover_observation(t,detections[i]["boxes_xyxy_score"],i+1) for t in current]
                    shown = [t for t in raw if t["confirmed"]]
                    require(len(shown) == len(returned), "Returned/current track mismatch")
                    require(len({t["detection_index"] for t in shown}) == len(shown), "Duplicate detector association")
                    frames.append(shown)
                    smooth.append([dict(t,box=t["kalman_box"]) for t in shown])
                    visible_ids.update(t["id"] for t in shown)
                    all_ids.update(t["id"] for t in raw)
                    stream.write(json.dumps(dict(frame_index=i,time_seconds=i/fps,raw_tracks=raw,
                                                  displayed_tracks=shown))+"\n")
            if use_gmc:
                require(tracker.gmc.index == len(detections), "GMC advanced incorrectly")
            measured, _ = evaluate(frames,gt,included,.5)
            kf_metrics, _ = evaluate(smooth,gt,included,.5)
            result = dict(name=name,config=config,metrics=measured,kalman_box_metrics=kf_metrics,
                identity_diagnostics=identity_counts(frames,gt,included),
                visible_ids=len(visible_ids),observed_candidate_ids=len(all_ids),
                allocated_ids=int(base_track._count),max_id=max(all_ids,default=0),
                detector_cache_sha256=digest(cache),provenance=provenance,
                timing=dict(host=platform.node(),cpu=platform.processor(),tracker_seconds=sum(elapsed),
                    tracker_only_fps=len(elapsed)/sum(elapsed),gmc_seconds=sum(gmc_times) if use_gmc else 0,
                    tracker_plus_gmc_fps=len(elapsed)/(sum(elapsed)+(sum(gmc_times) if use_gmc else 0)),
                    note="Server CPU replay only; excludes detector, decode, render, IO. GMC cached once and added to time. Not RK board FPS."))
            (folder / "summary.json").write_text(json.dumps(result,indent=2)+"\n")
            results.append(result)
            print(json.dumps({k:result[k] for k in ("name","metrics","identity_diagnostics","max_id","timing")}),flush=True)
    old = [r["displayed_tracks"] for r in baseline]
    old_metrics,_ = evaluate(old,gt,included,.5)
    detector_metrics,_ = evaluate([[dict(box=b[:4]) for b in d["boxes_xyxy_score"]] for d in detections],gt,included,.5)
    report = dict(provenance=provenance,frames=len(detections),reviewed_frames=len(included),
        detector_metrics=detector_metrics,baseline=dict(metrics=old_metrics,identity_diagnostics=identity_counts(old,gt,included)),
        results=results,
        scope="Actual public Dist-Tracker repository BOTSORT entry point, not claimed reproduction of paper L2-IoU. Fixed conf=.03 NMS=.45 detections; no detector retraining, no extra low detections, no area/aspect filtering, no ReID, no predicted-box display. Primary metrics use exact associated detector boxes; native KF boxes separately scored. GT only for offline evaluation. Identity diagnostics are not standard MOT IDSW/IDF1. Video00009 is validation, not independent tuning evidence.")
    (args.output / "comparison.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
