#!/usr/bin/env python3
"""Build an RKNN model from an RKNN-optimized ONNX export."""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


DEFAULT_ONNX = Path(
    "/mnt/chenziye/codes/ultralytics_yolov8/runs/visdrone/yolov8n_visdrone/weights/best_rknnopt.onnx"
)
DEFAULT_RKNN = Path("/mnt/chenziye/codes/ultralytics_yolov8/runs/visdrone/yolov8n_visdrone/weights/best.rknn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help=f"Input ONNX path. Default: {DEFAULT_ONNX}")
    parser.add_argument("--output", type=Path, default=DEFAULT_RKNN, help=f"Output RKNN path. Default: {DEFAULT_RKNN}")
    parser.add_argument(
        "--target",
        default="rk3588",
        help="Target platform, e.g. rk3588, rk3568, rk3566, rv1106.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization. Requires --dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Calibration dataset txt file. Required when --quantize is set.",
    )
    parser.add_argument(
        "--mean-values",
        default="0,0,0",
        help="Comma-separated mean values for RGB channels. Default: 0,0,0",
    )
    parser.add_argument(
        "--std-values",
        default="255,255,255",
        help="Comma-separated std values for RGB channels. Default: 255,255,255",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose RKNN logs.",
    )
    return parser.parse_args()


def parse_triplet(value: str) -> list[list[int]]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated integers, got: {value}")
    return [parts]


def main() -> None:
    args = parse_args()
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

    mean_values = parse_triplet(args.mean_values)
    std_values = parse_triplet(args.std_values)

    print(f"Input ONNX  : {onnx_path}")
    print(f"Output RKNN : {output_path}")
    print(f"Target      : {args.target}")
    print(f"Quantize    : {args.quantize}")
    if dataset_path:
        print(f"Dataset txt : {dataset_path}")
    print(f"Mean values : {mean_values}")
    print(f"Std values  : {std_values}")

    rknn = RKNN(verbose=args.verbose)

    ret = rknn.config(
        target_platform=args.target,
        mean_values=mean_values,
        std_values=std_values,
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
