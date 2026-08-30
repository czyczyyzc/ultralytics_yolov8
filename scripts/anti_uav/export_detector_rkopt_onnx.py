#!/usr/bin/env python3
"""Export a YOLOv8 detector to the airockchip/RKNN-optimized ONNX layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="YOLO detector checkpoint, for example best.pt.")
    parser.add_argument("--imgsz", default="640", help="Detector export size. Use N for square or H,W for rectangular.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output ONNX path.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset. RKNN YOLOv8 examples use opset 12.")
    parser.add_argument("--device", default="cpu", help="Export device.")
    parser.add_argument("--simplify", action="store_true", help="Run ONNX simplification during export.")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata JSON path.")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Do not fail when the graph does not look like a 3/4-scale RK-optimized YOLOv8 graph.",
    )
    return parser.parse_args()


def parse_imgsz(value: str) -> int | tuple[int, int]:
    parts = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Expected --imgsz as N or H,W, got: {value}")


def onnx_output_shapes(onnx_path: Path) -> list[list[int | str | None]]:
    import onnx

    model = onnx.load(str(onnx_path))
    shapes: list[list[int | str | None]] = []
    for output in model.graph.output:
        dims: list[int | str | None] = []
        tensor_type = output.type.tensor_type
        for dim in tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(int(dim.dim_value))
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        shapes.append(dims)
    return shapes


def validate_rkopt_shapes(shapes: list[list[int | str | None]]) -> dict:
    output_layouts = {
        6: (3, 2),
        8: (4, 2),
        9: (3, 3),
        12: (4, 3),
    }
    if len(shapes) not in output_layouts:
        raise ValueError(
            "RK-optimized YOLOv8 should expose 6/9 tensors for P3-P5 or 8/12 tensors for P2-P5 "
            f"(bbox/class[/score_sum] per scale), got {len(shapes)} output(s): {shapes}"
        )

    branch_count, pair_per_branch = output_layouts[len(shapes)]
    branches = []
    for branch_index in range(branch_count):
        start = branch_index * pair_per_branch
        branch = shapes[start : start + pair_per_branch]
        if any(len(shape) != 4 for shape in branch):
            raise ValueError(f"RK-optimized outputs must be NCHW rank-4 tensors, got branch {branch_index}: {branch}")
        box_shape = branch[0]
        cls_shape = branch[1]
        if isinstance(box_shape[1], int) and box_shape[1] % 4 != 0:
            raise ValueError(f"Branch {branch_index} bbox channels should be divisible by 4, got {box_shape}")
        if box_shape[2:] != cls_shape[2:]:
            raise ValueError(f"Branch {branch_index} bbox/class spatial shapes differ: {box_shape} vs {cls_shape}")
        if pair_per_branch == 3:
            score_shape = branch[2]
            if score_shape[2:] != box_shape[2:]:
                raise ValueError(f"Branch {branch_index} score_sum spatial shape differs: {score_shape} vs {box_shape}")
        branches.append(
            {"bbox": box_shape, "class": cls_shape, "score_sum": branch[2] if pair_per_branch == 3 else None}
        )

    if len(shapes) == 1 and len(shapes[0]) == 3:
        raise ValueError(f"Ultralytics single-output graph was exported instead of RK-optimized layout: {shapes[0]}")
    return {
        "output_count": len(shapes),
        "branch_count": branch_count,
        "pair_per_branch": pair_per_branch,
        "branches": branches,
    }


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {weights}")
    imgsz = parse_imgsz(args.imgsz)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = (
        Path(
            model.export(
                format="rknn",
                imgsz=imgsz,
                batch=1,
                device=args.device,
                opset=args.opset,
                simplify=args.simplify,
                dynamic=False,
            )
        )
        .expanduser()
        .resolve()
    )

    output_path = args.output.expanduser().resolve() if args.output else exported
    if output_path != exported:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported, output_path)

    shapes = onnx_output_shapes(output_path)
    validation: dict
    try:
        validation = validate_rkopt_shapes(shapes)
    except ValueError:
        if not args.allow_unverified:
            raise
        validation = {"output_count": len(shapes), "pair_per_branch": None, "branches": [], "warning": "unverified"}

    metadata = {
        "source_weights": str(weights),
        "onnx": str(output_path),
        "imgsz": list(imgsz) if isinstance(imgsz, tuple) else imgsz,
        "format": "rknn_yolov8_rkopt",
        "layout": "branch-major NCHW bbox/class/score_sum outputs; DFL and NMS run in runtime postprocess",
        "output_shapes": shapes,
        "validation": validation,
    }
    metadata_path = args.metadata.expanduser().resolve() if args.metadata else output_path.with_suffix(".rkopt.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved RK-optimized ONNX: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
