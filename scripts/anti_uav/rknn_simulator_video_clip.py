#!/usr/bin/env python3
"""Run an exact-build RKNN simulator replay on a contiguous video clip."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True, help="Exclusive source frame index.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rebuilt-rknn", type=Path, required=True)
    parser.add_argument("--expected-rknn-sha256", required=True)
    parser.add_argument("--input-size", default="544,960", help="Model input size as H,W.")
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="Continue simulator replay while recording a non-reproducible RKNN artifact hash.",
    )
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    return parser.parse_args()


def parse_input_size(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2 or any(part <= 0 for part in parts):
        raise ValueError(f"Expected positive H,W input size, got: {value}")
    return parts[0], parts[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def letterbox(
    frame_bgr: np.ndarray, input_height: int, input_width: int
) -> tuple[np.ndarray, float, float, float]:
    height, width = frame_bgr.shape[:2]
    ratio = min(input_width / width, input_height / height)
    resized_width = max(1, int(round(width * ratio)))
    resized_height = max(1, int(round(height * ratio)))
    dw = (input_width - resized_width) * 0.5
    dh = (input_height - resized_height) * 0.5
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))
    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return canvas, ratio, dw, dh


def dfl(position: np.ndarray) -> np.ndarray:
    batch, channels, height, width = position.shape
    bins = channels // 4
    values = position.reshape(batch, 4, bins, height, width).astype(np.float32)
    values -= values.max(axis=2, keepdims=True)
    values = np.exp(values)
    values /= values.sum(axis=2, keepdims=True)
    weights = np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1)
    return (values * weights).sum(axis=2)


def decode_boxes(position: np.ndarray, input_height: int, input_width: int) -> np.ndarray:
    grid_height, grid_width = position.shape[2:4]
    columns, rows = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
    grid = np.concatenate(
        (columns.reshape(1, 1, grid_height, grid_width), rows.reshape(1, 1, grid_height, grid_width)),
        axis=1,
    )
    stride = np.asarray(
        [input_width / grid_width, input_height / grid_height], dtype=np.float32
    ).reshape(1, 2, 1, 1)
    distances = dfl(position)
    top_left = grid + 0.5 - distances[:, 0:2]
    bottom_right = grid + 0.5 + distances[:, 2:4]
    return np.concatenate((top_left * stride, bottom_right * stride), axis=1)


def flatten_spatial(value: np.ndarray) -> np.ndarray:
    channels = value.shape[1]
    return value.transpose(0, 2, 3, 1).reshape(-1, channels)


def postprocess(
    outputs: list[np.ndarray],
    conf: float,
    nms_iou: float,
    input_height: int,
    input_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(outputs) != 9:
        raise RuntimeError(f"Expected 9 outputs, got {len(outputs)}")
    boxes = []
    probabilities = []
    for branch in range(3):
        offset = branch * 3
        boxes.append(flatten_spatial(decode_boxes(outputs[offset], input_height, input_width)))
        probabilities.append(flatten_spatial(outputs[offset + 1]))
    boxes_array = np.concatenate(boxes).astype(np.float32)
    scores = np.concatenate(probabilities).max(axis=1).astype(np.float32)
    selected = scores >= conf
    boxes_array = boxes_array[selected]
    scores = scores[selected]
    if not len(scores):
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
    boxes_xywh = boxes_array.copy()
    boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), scores.astype(float).tolist(), 0.0, nms_iou, eta=1.0, top_k=128
    )
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    return boxes_array[indices], scores[indices]


def restore_boxes(
    boxes: np.ndarray, ratio: float, dw: float, dh: float, width: int, height: int
) -> np.ndarray:
    restored = boxes.astype(np.float32, copy=True)
    if not restored.size:
        return restored.reshape(0, 4)
    restored[:, [0, 2]] = (restored[:, [0, 2]] - dw) / ratio
    restored[:, [1, 3]] = (restored[:, [1, 3]] - dh) / ratio
    restored[:, [0, 2]] = restored[:, [0, 2]].clip(0, width - 1)
    restored[:, [1, 3]] = restored[:, [1, 3]].clip(0, height - 1)
    return restored


def main() -> None:
    args = parse_args()
    input_height, input_width = parse_input_size(args.input_size)
    if args.start_frame < 0 or args.end_frame <= args.start_frame:
        raise ValueError("Invalid frame range")
    from rknn.api import RKNN

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.rebuilt_rknn.parent.mkdir(parents=True, exist_ok=True)

    runtime = RKNN(verbose=False)
    if runtime.config(
        target_platform="rk3588", mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]]
    ) != 0:
        raise RuntimeError("RKNN config failed")
    if runtime.load_onnx(model=str(args.onnx.resolve())) != 0:
        raise RuntimeError("RKNN ONNX load failed")
    if runtime.build(do_quantization=True, dataset=str(args.calibration.resolve())) != 0:
        raise RuntimeError("RKNN INT8 build failed")
    if runtime.export_rknn(str(args.rebuilt_rknn.resolve())) != 0:
        raise RuntimeError("RKNN export failed")
    rebuilt_sha = sha256_file(args.rebuilt_rknn)
    expected_sha = args.expected_rknn_sha256.lower()
    hash_matches = rebuilt_sha == expected_sha
    if not hash_matches and not args.allow_hash_mismatch:
        raise RuntimeError(
            f"Rebuilt artifact hash mismatch: {rebuilt_sha} != {expected_sha}"
        )
    if not hash_matches:
        print(
            f"WARNING: continuing with rebuilt artifact hash {rebuilt_sha}; expected {expected_sha}",
            flush=True,
        )
    if runtime.init_runtime() != 0:
        raise RuntimeError("RKNN simulator init failed")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    inference_ms: list[float] = []
    detection_count = 0
    processed = 0
    try:
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["frame", "source_frame", "timestamp_sec", "x", "y", "width", "height", "confidence", "class_id"]
            )
            for source_frame in range(args.start_frame, args.end_frame):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Decode stopped at source frame {source_frame}")
                input_rgb, ratio, dw, dh = letterbox(frame, input_height, input_width)
                started = time.perf_counter_ns()
                outputs = runtime.inference(inputs=[input_rgb[np.newaxis, ...]])
                inference_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
                boxes, scores = postprocess(
                    outputs, args.conf, args.nms_iou, input_height, input_width
                )
                boxes = restore_boxes(boxes, ratio, dw, dh, width, height)
                local_frame = source_frame - args.start_frame
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = (float(value) for value in box)
                    writer.writerow(
                        [
                            local_frame,
                            source_frame,
                            f"{local_frame / fps:.6f}",
                            f"{x1:.6f}",
                            f"{y1:.6f}",
                            f"{x2 - x1:.6f}",
                            f"{y2 - y1:.6f}",
                            f"{float(score):.6f}",
                            0,
                        ]
                    )
                    detection_count += 1
                processed += 1
                if processed % 50 == 0 or source_frame + 1 == args.end_frame:
                    print(f"Processed {processed}/{args.end_frame - args.start_frame}", flush=True)
    finally:
        capture.release()
        runtime.release()

    payload = {
        "schema_version": "anti_uav.rknn_simulator_video_clip.v1",
        "backend": "rknn-toolkit2-2.3.2-in-memory-int8-build-simulator",
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": sha256_file(args.onnx),
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256_file(args.calibration),
        "rebuilt_rknn": str(args.rebuilt_rknn.resolve()),
        "rebuilt_rknn_sha256": rebuilt_sha,
        "expected_rknn_sha256": expected_sha,
        "rknn_hash_matches_expected": hash_matches,
        "video": str(args.video.resolve()),
        "input_size_height_width": [input_height, input_width],
        "source_frame_range": [args.start_frame, args.end_frame],
        "frames": processed,
        "source_fps": fps,
        "confidence": args.conf,
        "nms_iou": args.nms_iou,
        "detections": detection_count,
        "simulator_inference_ms": {
            "mean": statistics.mean(inference_ms),
            "median": statistics.median(inference_ms),
            "max": max(inference_ms),
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
