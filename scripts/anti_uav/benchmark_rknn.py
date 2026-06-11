#!/usr/bin/env python3
"""Benchmark RKNN model forward latency on images or video frames and save a JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="RKNN model path.")
    parser.add_argument("--source", required=True, help="Image directory, image file, or video file.")
    parser.add_argument("--input-size", default="640,640", help="Input size as H,W.")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames to benchmark.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before timing.")
    parser.add_argument("--input-mode", default="rgb", choices=("rgb", "gray", "ir"), help="Input preprocessing mode.")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE for gray/IR preprocessing.")
    parser.add_argument("--output-json", default="", help="Optional summary JSON path.")
    parser.add_argument("--preview-dir", default="", help="Optional directory to save a few preprocessed frames.")
    return parser.parse_args()


def parse_input_size(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected H,W input size, got: {value}")
    return parts[0], parts[1]


def load_runtime():
    try:
        from rknnlite.api import RKNNLite

        runtime = RKNNLite()
        runtime_type = "rknnlite"
    except ImportError:
        from rknn.api import RKNN

        runtime = RKNN()
        runtime_type = "rknn"
    return runtime, runtime_type


def infer_with_layout_fallback(runtime, tensor_nhwc: np.ndarray, cached_mode: str | None):
    rgb = tensor_nhwc[0]
    candidates = {
        "nhwc_batch": (np.ascontiguousarray(tensor_nhwc), "nhwc"),
        "nchw_batch": (np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), "nchw"),
        "raw_hwc": (np.ascontiguousarray(rgb), None),
    }
    order = [cached_mode] if cached_mode else []
    order.extend(mode for mode in candidates if mode not in order)

    last_error = None
    for mode in order:
        input_tensor, data_format = candidates[mode]
        kwargs = {"inputs": [input_tensor]}
        if data_format is not None:
            kwargs["data_format"] = [data_format]
        try:
            outputs = runtime.inference(**kwargs)
        except TypeError:
            kwargs.pop("data_format", None)
            try:
                outputs = runtime.inference(**kwargs)
            except Exception as exc:
                last_error = exc
                continue
        except Exception as exc:
            last_error = exc
            continue
        if outputs is not None:
            return outputs, mode

    raise RuntimeError(f"RKNN inference failed for all input layouts, last error: {last_error}")


def iter_frames(source: str):
    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        for image_path in sorted(source_path.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                frame = cv2.imread(str(image_path))
                if frame is not None:
                    yield image_path.name, frame
        return

    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise RuntimeError(f"Unable to read image: {source_path}")
        yield source_path.name, frame
        return

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source_path}")
    index = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            index += 1
            yield f"frame_{index:06d}", frame
    finally:
        cap.release()


def prepare_frame(frame: np.ndarray, input_mode: str, clahe: bool, input_size: tuple[int, int]) -> np.ndarray:
    if input_mode in {"gray", "ir"}:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if clahe:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(frame, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * ratio))))
    return sorted(values)[index]


def main() -> None:
    args = parse_args()
    input_size = parse_input_size(args.input_size)
    runtime, runtime_type = load_runtime()

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"RKNN model not found: {model_path}")

    preview_dir = Path(args.preview_dir).expanduser().resolve() if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    ret = runtime.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed with code {ret}")
    ret = runtime.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed with code {ret}")

    frames = []
    for name, frame in iter_frames(args.source):
        frames.append((name, frame))
        if len(frames) >= args.max_frames:
            break
    if not frames:
        raise RuntimeError("No frames found for benchmarking")

    tensors = [prepare_frame(frame, args.input_mode, args.clahe, input_size) for _, frame in frames]

    input_layout = None
    for tensor in tensors[: min(args.warmup, len(tensors))]:
        _, input_layout = infer_with_layout_fallback(runtime, tensor, input_layout)

    times_ms = []
    output_shapes = None
    for index, tensor in enumerate(tensors, start=1):
        start = perf_counter()
        outputs, input_layout = infer_with_layout_fallback(runtime, tensor, input_layout)
        elapsed_ms = (perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        if output_shapes is None:
            output_shapes = [list(np.asarray(output).shape) for output in outputs]
        if preview_dir is not None and index <= 4:
            preview = tensor[0]
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(preview_dir / f"preview_{index:02d}.jpg"), preview_bgr)

    runtime.release()

    mean_ms = float(sum(times_ms) / len(times_ms))
    summary = {
        "runtime": runtime_type,
        "input_layout": input_layout,
        "model": str(model_path),
        "frames": len(times_ms),
        "input_size": list(input_size),
        "mean_ms": mean_ms,
        "p50_ms": percentile(times_ms, 0.50),
        "p95_ms": percentile(times_ms, 0.95),
        "max_ms": max(times_ms),
        "fps_mean": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "output_shapes": output_shapes,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
