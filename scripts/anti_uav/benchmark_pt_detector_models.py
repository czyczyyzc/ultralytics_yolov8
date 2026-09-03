#!/usr/bin/env python3
"""Benchmark multiple PyTorch detectors with identical tensor inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat once for each model",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, received {value!r}")
    label, path = value.split("=", 1)
    model_path = Path(path).expanduser().resolve()
    if not label or not model_path.is_file():
        raise FileNotFoundError(f"Invalid model specification: {value}")
    return label, model_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(label: str, path: Path, args: argparse.Namespace, device: torch.device) -> dict:
    wrapped = YOLO(str(path))
    model = wrapped.model.to(device).eval()
    if args.half:
        model.half()
    dtype = torch.float16 if args.half else torch.float32
    generator = torch.Generator(device=device).manual_seed(20260903)
    image = torch.rand((1, 3, args.height, args.width), generator=generator, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        synchronize(device)
        samples_ms = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            for _ in range(args.iterations):
                model(image)
            synchronize(device)
            samples_ms.append((time.perf_counter() - started) * 1000 / args.iterations)

    median_ms = statistics.median(samples_ms)
    return {
        "label": label,
        "model": str(path),
        "sha256": sha256(path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latency_ms_samples": samples_ms,
        "latency_ms_median": median_ms,
        "fps_median": 1000 / median_ms,
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative; iterations and repeats must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    results = [benchmark(*parse_model(value), args, device) for value in args.model]
    summary = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "input_nchw": [1, 3, args.height, args.width],
        "dtype": "float16" if args.half else "float32",
        "warmup": args.warmup,
        "iterations_per_repeat": args.iterations,
        "repeats": args.repeats,
        "measurement": "model forward only; synchronized after each repeat",
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
