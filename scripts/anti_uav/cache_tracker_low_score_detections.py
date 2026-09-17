#!/usr/bin/env python3
"""Supplement an immutable full-video detector cache with low-score detections."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.render_pt_detector_video import dump, sha256


def merge_low_detections(original, reinferred, low, high):
    original = np.asarray(original, dtype=np.float32).reshape(-1, 5)
    reinferred = np.asarray(reinferred, dtype=np.float32).reshape(-1, 5)
    new_high = reinferred[reinferred[:, 4] >= high]
    if new_high.shape != original.shape or not np.allclose(new_high, original, atol=1e-4, rtol=0):
        raise ValueError("High-score detections changed; this is not a controlled low-score ablation")
    additions = reinferred[(reinferred[:, 4] >= low) & (reinferred[:, 4] < high)]
    return np.concatenate((original, additions)), len(additions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="6")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--low", type=float, default=.01)
    args = parser.parse_args()
    summary_path = args.detector_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    cache = args.detector_dir / "predictions.jsonl"
    original = [json.loads(line) for line in cache.read_text().splitlines()]
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (0 < args.low < summary["conf"] < 1) or args.batch < 1:
        raise ValueError("Invalid confidence or batch")
    if not summary["full_source_video"] or summary["frame_stride"] != 1 or summary["tracker_used"]:
        raise ValueError("Full detector-only cache required")
    if summary["precision"] != "PT FP32" or summary["ground_truth_used"]:
        raise ValueError("Non-GT-guided FP32 baseline required")
    if len(original) != summary["frames"]:
        raise ValueError("Incomplete baseline cache")
    if sha256(Path(summary["source_video"])) != summary["source_sha256"]:
        raise ValueError("Source video changed")
    if sha256(Path(summary["weights"])) != summary["weights_sha256"]:
        raise ValueError("Weights changed")
    args.output.mkdir(parents=True)
    protocol = dict(baseline_summary_sha256=sha256(summary_path),
                    baseline_predictions_sha256=sha256(cache), source_sha256=summary["source_sha256"],
                    weights_sha256=summary["weights_sha256"], low=args.low, high=summary["conf"],
                    input_hw=summary["input_hw"], batch=args.batch, precision="PT FP32",
                    ground_truth_used=False, device=args.device,
                    policy="Preserve exact original high boxes, append re-inferred low boxes; reject high drift >1e-4")
    dump(args.output / "protocol.json", protocol)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    model = YOLO(summary["weights"])
    cap = cv2.VideoCapture(summary["source_video"])
    count, additions = 0, 0
    start = time.monotonic()
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open source video")
        with (args.output / "predictions.jsonl").open("x") as stream:
            while count < len(original):
                frames = []
                for _ in range(min(args.batch, len(original)-count)):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Early EOF at {count+len(frames)}")
                    frames.append(frame)
                results = model.predict(source=frames, imgsz=summary["input_hw"], rect=False,
                    conf=args.low, iou=summary["nms_iou"], max_det=summary["max_det"],
                    device=args.device, half=False, batch=args.batch, verbose=False, save=False, stream=False)
                if len(results) != len(frames):
                    raise RuntimeError("Frame/result count mismatch")
                for result in results:
                    record = original[count]
                    if record["frame_index"] != count:
                        raise ValueError("Baseline frame order mismatch")
                    boxes = np.column_stack((result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()))
                    boxes = boxes[np.argsort(-boxes[:, 4], kind="stable")]
                    try:
                        merged, added = merge_low_detections(record["boxes_xyxy_score"], boxes, args.low, summary["conf"])
                    except ValueError as error:
                        raise ValueError(f"Frame {count}: {error}") from error
                    stream.write(json.dumps(dict(record, boxes_xyxy_score=merged.tolist()))+"\n")
                    count += 1
                    additions += added
                if count % 512 == 0 or count == len(original):
                    status = dict(stage="inference", frames=count, total=len(original), low_detections_added=additions,
                                  seconds=round(time.monotonic()-start, 2))
                    dump(args.output / "status.json", status)
                    print(json.dumps(status), flush=True)
        report = dict(protocol, frames=count, low_detections_added=additions,
                      high_detections_preserved_exactly=True, full_source_video=True,
                      predictions_sha256=sha256(args.output / "predictions.jsonl"),
                      seconds=round(time.monotonic()-start, 2))
        dump(args.output / "summary.json", report)
        dump(args.output / "status.json", dict(stage="complete", frames=count))
        print(json.dumps(report), flush=True)
    except BaseException as error:
        dump(args.output / "status.json", dict(stage="failed", frames=count, error=repr(error)))
        raise
    finally:
        cap.release()


if __name__ == "__main__":
    main()
