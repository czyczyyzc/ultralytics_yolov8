#!/usr/bin/env python3
"""Evaluate board observations or replay a GMC variant on fixed RKNN detections.

Replay times are not NPU/pipeline FPS. Ground truth is used only for scoring.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np
from dist_numpy_runtime import CONFIG, as_results, load_dist, observations
from efficient_gmc import EfficientGMC
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.evaluate_dist_tracker_cache import load_ground_truth
from scripts.anti_uav.analyze_cached_tracker_suppression import evaluate, digest
from scripts.anti_uav.sweep_native_tracker_parameters import identity_counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("detections", "manifest", "coco", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--tracked", type=Path)
    p.add_argument("--upstream", type=Path)
    p.add_argument("--video", type=Path)
    p.add_argument("--gmc", choices=("public","compact"), default="compact")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--corners", type=int, default=256)
    p.add_argument("--refresh", type=int, default=1)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    meta, gt, included = load_ground_truth(args.manifest,args.coco)
    detections = [json.loads(s) for s in args.detections.read_text().splitlines()]
    if len(detections) != meta["video"]["frameCount"]:
        raise RuntimeError("Need complete detector cache")
    assert all(row["frame_index"] == i for i,row in enumerate(detections))
    tracked, timing = [], []
    if args.tracked:
        other = [json.loads(s) for s in args.tracked.read_text().splitlines()]
        assert len(other) == len(detections)
        for index,(a,b) in enumerate(zip(detections,other)):
            assert b["frame_index"] == index
            assert a["boxes_xyxy_score"] == b["boxes_xyxy_score"], f"Different detector at frame {index}"
            tracked.append(b["displayed_tracks"])
    else:
        if not args.video or not args.upstream or digest(args.video) != meta["video"]["sha256"]:
            raise RuntimeError("Need matching source video and upstream for replay")
        cv2.setNumThreads(2)
        cv2.setRNGSeed(20260917)
        tracker = load_dist(args.upstream)(SimpleNamespace(**CONFIG), frame_rate=meta["video"]["fps"])
        if args.gmc == "compact":
            tracker.gmc = EfficientGMC(args.width,args.corners,args.refresh)
        cap = cv2.VideoCapture(str(args.video))
        try:
            with (args.output / "observations.jsonl").open("x") as stream:
                for index,row in enumerate(detections):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Decode failed at {index}")
                    boxes = np.asarray(row["boxes_xyxy_score"],np.float32).reshape(-1,5)
                    start = time.perf_counter()
                    tracker.update(as_results(boxes),img=frame)
                    shown = observations(tracker,boxes)
                    timing.append(time.perf_counter()-start)
                    tracked.append(shown)
                    stream.write(json.dumps(dict(frame_index=index, boxes_xyxy_score=boxes.tolist(), displayed_tracks=shown))+"\n")
                    if index % 2000 == 0:
                        print(json.dumps(dict(frame=index)),flush=True)
        finally:
            cap.release()
    for index,shown in enumerate(tracked):
        seen = set()
        for track in shown:
            key = track["detection_index"]
            assert key not in seen
            seen.add(key)
            box = detections[index]["boxes_xyxy_score"][key]
            np.testing.assert_allclose(track["box"], box[:4], rtol=1e-6, atol=1e-4)
            assert abs(track["score"]-box[4]) < 1e-6
    metrics,_ = evaluate(tracked,gt,included,.5)
    det_metrics,_ = evaluate([[dict(box=b[:4]) for b in r["boxes_xyxy_score"]] for r in detections],gt,included,.5)
    ids = {t["id"] for rows in tracked for t in rows}
    result = dict(frames=len(tracked), metrics=metrics, detector_metrics=det_metrics,
        identity_diagnostics=identity_counts(tracked,gt,included), visible_ids=len(ids),
        maximum_visible_id=max(ids,default=0), detector_cache_sha256=digest(args.detections),
        source_sha256=meta["video"]["sha256"], coco_sha256=digest(args.coco),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        replay_tracker_gmc_seconds=sum(timing),
        scope="IoU .5 on reviewed Video00009 frames; uncertain/unreviewed excluded. Same RKNN detections; no GT in tracker. ID changes are single-GT diagnostics, not standard MOT IDSW. This is tuning/evaluation on Video00009, not independent generalization evidence. Replay time is NOT pipeline FPS.")
    (args.output / "summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
