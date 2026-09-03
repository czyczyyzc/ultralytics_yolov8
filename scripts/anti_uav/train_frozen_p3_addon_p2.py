#!/usr/bin/env python3
"""Train a one-way P2 add-on while preserving a completed YOLOv8 P3-P5 detector."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    part for part in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if part
)

from scripts.anti_uav.frozen_p3_addon_p2_trainer import FrozenP3AddOnP2Trainer
from ultralytics import YOLO
from ultralytics.nn.modules import FrozenP3AddOnP2Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds


DEFAULT_CFG = REPO_ROOT / "ultralytics/cfg/models/v8/yolov8-frozen-p3-addon-p2.yaml"


def parse_imgsz(value: str) -> list[int]:
    dimensions = [int(item) for item in value.split(",")]
    if len(dimensions) != 2 or any(item <= 0 for item in dimensions):
        raise argparse.ArgumentTypeError("imgsz must be H,W with positive dimensions")
    return dimensions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-model", type=Path, required=True, help="Frozen P3 reference checkpoint")
    parser.add_argument("--resume-model", type=Path, help="Existing add-on checkpoint for another training stage")
    parser.add_argument("--model-cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--initialized-checkpoint", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=parse_imgsz, default=[544, 960])
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def transfer_frozen_p3_weights(source: DetectionModel, target: DetectionModel) -> dict[str, int]:
    """Copy the complete legacy graph and remap the three Detect towers behind the new P2 tower."""
    source_state = source.float().state_dict()
    target_state = target.state_dict()
    transferred = {
        key: value for key, value in source_state.items() if key in target_state and value.shape == target_state[key].shape
    }
    direct_count = len(transferred)
    source_head_index = len(source.model) - 1
    target_head_index = len(target.model) - 1
    source_prefix = f"model.{source_head_index}."
    target_prefix = f"model.{target_head_index}."
    remapped_count = 0

    for source_key, value in source_state.items():
        if not source_key.startswith(source_prefix):
            continue
        suffix = source_key[len(source_prefix):]
        parts = suffix.split(".")
        if parts[0] in {"cv2", "cv3"} and len(parts) >= 2:
            parts[1] = str(int(parts[1]) + 1)
            target_key = target_prefix + ".".join(parts)
        elif parts[0] == "dfl":
            target_key = target_prefix + suffix
        else:
            continue
        if target_key in target_state and target_state[target_key].shape == value.shape:
            transferred[target_key] = value
            remapped_count += 1

    target.load_state_dict(transferred, strict=False)
    expected_legacy = {
        key
        for key in target_state
        if key.startswith(target_prefix + "dfl.")
        or any(key.startswith(target_prefix + f"{branch}.{level}.") for branch in ("cv2", "cv3") for level in (1, 2, 3))
    }
    missing_legacy = expected_legacy.difference(transferred)
    if missing_legacy:
        raise RuntimeError(f"Failed to initialize {len(missing_legacy)} legacy tensors: {sorted(missing_legacy)[:3]}")
    return {
        "direct_tensors": direct_count,
        "legacy_head_remapped_tensors": remapped_count,
        "transferred_tensors": len(transferred),
        "target_tensors": len(target_state),
    }


def raw_features(model: DetectionModel, image: torch.Tensor) -> list[torch.Tensor]:
    output = model(image)
    return output[1] if isinstance(output, tuple) else output


def verify_legacy_outputs(reference: DetectionModel, addon: DetectionModel, size: int = 256) -> dict[str, object]:
    """Prove that adding P2 did not numerically change P3/P4/P5 raw outputs."""
    reference = reference.float().cpu().eval()
    addon = addon.float().cpu().eval()
    torch.manual_seed(20260903)
    image = torch.rand(1, 3, size, size)
    with torch.inference_mode():
        reference_raw = raw_features(reference, image)
        addon_raw = raw_features(addon, image)
    if len(reference_raw) != 3 or len(addon_raw) != 4:
        raise RuntimeError(f"Unexpected levels: reference={len(reference_raw)}, addon={len(addon_raw)}")
    maximum_absolute_errors = [
        float((source_tensor - addon_tensor).abs().max())
        for source_tensor, addon_tensor in zip(reference_raw, addon_raw[1:])
    ]
    report = {
        "input_shape": [1, 3, size, size],
        "reference_levels": len(reference_raw),
        "addon_levels": len(addon_raw),
        "legacy_max_abs_error": maximum_absolute_errors,
        "bit_exact": all(error == 0.0 for error in maximum_absolute_errors),
    }
    if not report["bit_exact"]:
        raise RuntimeError(f"Frozen P3 regression failed: {report}")
    return report


def initialize_addon_model(p3_model: Path, model_cfg: Path, output: Path) -> tuple[YOLO, dict[str, object]]:
    source = YOLO(str(p3_model))
    target = YOLO(str(model_cfg))
    target.model = DetectionModel(str(model_cfg), nc=source.model.nc, verbose=True)
    if not isinstance(target.model.model[-1], FrozenP3AddOnP2Detect):
        raise TypeError("The add-on model must end with FrozenP3AddOnP2Detect")
    target.model.names = source.model.names
    target.ckpt = {}
    transfer_report = transfer_frozen_p3_weights(source.model, target.model)
    regression_report = verify_legacy_outputs(source.model, target.model)
    report = {
        "source": str(p3_model.resolve()),
        "model_cfg": str(model_cfg.resolve()),
        "transfer": transfer_report,
        "legacy_regression": regression_report,
        "strides": target.model.stride.tolist(),
        "parameters": sum(parameter.numel() for parameter in target.model.parameters()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    target.save(output)
    reloaded = YOLO(str(output))
    report["reloaded_legacy_regression"] = verify_legacy_outputs(source.model, reloaded.model)
    return reloaded, report


def main() -> None:
    args = parse_args()
    init_seeds(args.seed, deterministic=True)
    initialized_checkpoint = args.initialized_checkpoint or args.project / f"{args.name}_initialized.pt"
    if args.resume_model:
        model = YOLO(str(args.resume_model))
        reference = YOLO(str(args.p3_model))
        initialization = {
            "resume_model": str(args.resume_model.resolve()),
            "legacy_regression": verify_legacy_outputs(reference.model, model.model),
        }
    else:
        model, initialization = initialize_addon_model(args.p3_model, args.model_cfg, initialized_checkpoint)

    initialization_manifest = args.project / f"{args.name}_initialization.json"
    initialization_manifest.parent.mkdir(parents=True, exist_ok=True)
    initialization_manifest.write_text(json.dumps(initialization, indent=2) + "\n")
    print(json.dumps({"initialization": initialization}, indent=2))
    if args.initialize_only:
        return

    model.train(
        trainer=FrozenP3AddOnP2Trainer,
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        epochs=args.epochs,
        patience=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        save=True,
        save_period=args.save_period,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        nbs=128,
        seed=args.seed,
        deterministic=True,
        single_cls=True,
        rect=False,
        cos_lr=True,
        close_mosaic=0,
        amp=True,
        val=True,
        plots=True,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    reference = YOLO(str(args.p3_model))
    checkpoint_regression = {}
    for label, checkpoint in (("best", model.trainer.best), ("last", model.trainer.last)):
        if checkpoint.exists():
            checkpoint_regression[label] = verify_legacy_outputs(reference.model, YOLO(str(checkpoint)).model)
    manifest = {
        "p3_model": str(args.p3_model.resolve()),
        "resume_model": str(args.resume_model.resolve()) if args.resume_model else None,
        "model_cfg": str(args.model_cfg.resolve()),
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz_height_width": args.imgsz,
        "device": args.device,
        "seed": args.seed,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
        "trainable_policy": "P2 adapter + P2 cv2/cv3 only; legacy modules and BN buffers frozen",
        "initialization": initialization,
        "checkpoint_legacy_regression": checkpoint_regression,
        "save_dir": str(model.trainer.save_dir),
    }
    output = Path(model.trainer.save_dir) / "training_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
