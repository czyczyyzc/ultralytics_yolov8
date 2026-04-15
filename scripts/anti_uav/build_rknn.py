#!/usr/bin/env python3
"""Build an RKNN model for the alerting-only anti-UAV perception pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="Input ONNX path.")
    parser.add_argument("--output", type=Path, required=True, help="Output RKNN path.")
    parser.add_argument("--target", default="rk3588", help="Target platform, for example rk3588.")
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 quantization. Requires --dataset.")
    parser.add_argument("--dataset", type=Path, default=None, help="Calibration dataset txt file.")
    parser.add_argument("--mean-values", default="0,0,0", help="Comma-separated channel means.")
    parser.add_argument("--std-values", default="255,255,255", help="Comma-separated channel std values.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose RKNN logging.")
    return parser.parse_args()


def parse_triplet(value: str) -> list[list[int]]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated integers, got: {value}")
    return [parts]


def main() -> None:
    args = parse_args()
    from rknn.api import RKNN

    onnx_path = args.onnx.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve() if args.dataset else None

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if args.quantize and dataset_path is None:
        raise ValueError("--dataset is required when --quantize is set")
    if dataset_path and not dataset_path.exists():
        raise FileNotFoundError(f"Calibration dataset file not found: {dataset_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=args.verbose)
    ret = rknn.config(
        target_platform=args.target,
        mean_values=parse_triplet(args.mean_values),
        std_values=parse_triplet(args.std_values),
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config failed with code {ret}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

    ret = rknn.build(
        do_quantization=args.quantize,
        dataset=str(dataset_path) if dataset_path else None,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.build failed with code {ret}")

    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed with code {ret}")

    rknn.release()
    print(f"Saved RKNN model to: {output_path}")


if __name__ == "__main__":
    main()
