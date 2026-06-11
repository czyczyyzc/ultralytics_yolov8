#!/usr/bin/env python3
"""Export the pair-head presence verifier to ONNX and optionally build an RKNN artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input pair-head .pt checkpoint.")
    parser.add_argument("--onnx", type=Path, required=True, help="Output ONNX path.")
    parser.add_argument("--metadata-json", type=Path, default=None, help="Optional metadata sidecar JSON path.")
    parser.add_argument("--rknn", type=Path, default=None, help="Optional output RKNN path.")
    parser.add_argument("--target", default="rk3588", help="RKNN target platform.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--build-rknn", action="store_true", help="Also build an RKNN model if rknn-toolkit2 is available.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose RKNN build logging.")
    return parser.parse_args()


def export_onnx(
    checkpoint_path: Path,
    onnx_path: Path,
    metadata_path: Path,
    *,
    opset: int,
) -> dict:
    import torch

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    patch_size = int(payload.get("patch_size", 64))
    feature_names = tuple(payload.get("feature_names", solutions.HeuristicPresenceVerifier.feature_names))
    hidden_dim = int(payload.get("hidden_dim", 64))
    loss_mode = str(payload.get("loss_mode", "ce")).lower()
    use_metadata = bool(payload.get("use_metadata", True))
    metadata_dim = len(feature_names) if use_metadata else 0

    network = solutions.PairPresenceNet(in_channels=2, metadata_dim=metadata_dim, hidden_dim=hidden_dim)
    network.load_state_dict(payload["state_dict"])
    model = network.model.cpu().eval()

    image_pair = torch.randn(1, 2, patch_size, patch_size, dtype=torch.float32)
    args = (image_pair,)
    input_names = ["image_pair"]
    dynamic_axes = None
    if use_metadata:
        metadata = torch.randn(1, metadata_dim, dtype=torch.float32)
        args = (image_pair, metadata)
        input_names.append("metadata")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args,
        str(onnx_path),
        input_names=input_names,
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    metadata_payload = {
        "feature_names": list(feature_names),
        "patch_size": patch_size,
        "loss_mode": loss_mode,
        "use_metadata": use_metadata,
        "metadata_dim": metadata_dim,
        "hidden_dim": hidden_dim,
        "input_names": input_names,
        "output_names": ["logits"],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata_payload


def build_rknn(onnx_path: Path, rknn_path: Path, metadata: dict, *, target: str, verbose: bool) -> None:
    from rknn.api import RKNN

    rknn = RKNN(verbose=verbose)
    ret = rknn.config(target_platform=target)
    if ret != 0:
        raise RuntimeError(f"rknn.config failed with code {ret}")

    input_size_list = [[2, int(metadata["patch_size"]), int(metadata["patch_size"])]]
    if metadata.get("use_metadata", False):
        input_size_list.append([int(metadata["metadata_dim"])])

    ret = rknn.load_onnx(model=str(onnx_path), input_size_list=input_size_list)
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"rknn.build failed with code {ret}")

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed with code {ret}")
    rknn.release()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    onnx_path = args.onnx.expanduser().resolve()
    metadata_path = (
        args.metadata_json.expanduser().resolve() if args.metadata_json else onnx_path.with_suffix(".json")
    )
    rknn_path = args.rknn.expanduser().resolve() if args.rknn else onnx_path.with_suffix(".rknn")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    metadata = export_onnx(checkpoint_path, onnx_path, metadata_path, opset=args.opset)
    summary = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "metadata_json": str(metadata_path),
    }

    if args.build_rknn:
        build_rknn(onnx_path, rknn_path, metadata, target=args.target, verbose=args.verbose)
        summary["rknn"] = str(rknn_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
