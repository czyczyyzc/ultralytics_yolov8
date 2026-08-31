#!/usr/bin/env python3
"""Fine-tune one YOLOv8n real-gray LOVO fold with an Anti-UAV300 RGB rehearsal mix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(part for part in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if part)

from scripts.anti_uav.lovo_detection_trainer import LovoDetectionTrainer
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-cfg",
        type=Path,
        default=None,
        help="Optional model architecture YAML. When set, initialize that architecture from --model weights.",
    )
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--save-period", type=int, default=5)
    return parser.parse_args()


def p2_target_key(source_key: str) -> str | None:
    """Map reusable standard YOLOv8 P3-P5 neck/head tensors into YOLOv8-P2."""
    layer_map = {16: 22, 18: 24, 19: 25, 21: 27}
    parts = source_key.split(".")
    if len(parts) < 3 or parts[0] != "model":
        return None
    try:
        source_layer = int(parts[1])
    except ValueError:
        return None

    if source_layer in layer_map:
        parts[1] = str(layer_map[source_layer])
        return ".".join(parts)
    if source_layer != 22 or len(parts) < 4:
        return None

    parts[1] = "28"
    if parts[2] in {"cv2", "cv3"}:
        try:
            parts[3] = str(int(parts[3]) + 1)
        except ValueError:
            return None
    elif parts[2] != "dfl":
        return None
    return ".".join(parts)


def initialize_p2_model(initial_weights: Path, model_cfg: Path) -> tuple[YOLO, dict[str, int]]:
    """Build a one-class P2 model and retain every shape-compatible standard-model tensor."""
    source = YOLO(str(initial_weights))
    target = YOLO(str(model_cfg))
    target.model = DetectionModel(str(model_cfg), nc=source.model.nc, verbose=True)
    target.model.names = source.model.names
    target.ckpt = {}

    source_state = source.model.float().state_dict()
    target_state = target.model.state_dict()
    transferred = {
        key: value for key, value in source_state.items() if key in target_state and value.shape == target_state[key].shape
    }
    direct_count = len(transferred)
    remapped_count = 0
    for source_key, value in source_state.items():
        target_key = p2_target_key(source_key)
        if target_key and target_key not in transferred and target_key in target_state:
            if value.shape == target_state[target_key].shape:
                transferred[target_key] = value
                remapped_count += 1

    target.model.load_state_dict(transferred, strict=False)
    report = {
        "direct_tensors": direct_count,
        "p2_remapped_tensors": remapped_count,
        "transferred_tensors": len(transferred),
        "target_tensors": len(target_state),
    }
    print(json.dumps({"p2_initialization": report}, indent=2))
    return target, report


def build_model(
    initial_weights: Path,
    model_cfg: Path | None = None,
    initialized_checkpoint: Path | None = None,
) -> tuple[YOLO, dict[str, int] | None]:
    if model_cfg is None:
        return YOLO(str(initial_weights)), None
    if initialized_checkpoint is None:
        raise ValueError("initialized_checkpoint is required with --model-cfg so DDP workers retain warm-start weights")

    if model_cfg.stem.endswith("-p2"):
        model, report = initialize_p2_model(initial_weights, model_cfg)
    else:
        model = YOLO(str(model_cfg)).load(str(initial_weights))
        report = None
    initialized_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(initialized_checkpoint)
    return YOLO(str(initialized_checkpoint)), report


def main() -> None:
    args = parse_args()
    fold = args.fold_dir.name
    data = args.fold_dir / "train_rgb_monitor.yaml"
    if not data.exists():
        raise FileNotFoundError(data)
    init_seeds(args.seed, deterministic=True)
    initialized_checkpoint = args.project / f"{fold}_initialized.pt" if args.model_cfg else None
    model, initialization = build_model(args.model, args.model_cfg, initialized_checkpoint)
    model.train(
        trainer=LovoDetectionTrainer,
        data=str(data),
        imgsz=[544, 960],
        epochs=args.epochs,
        patience=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=fold,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        cos_lr=True,
        amp=True,
        deterministic=True,
        seed=args.seed,
        nbs=128,
        rect=False,
        cache=False,
        val=True,
        save=True,
        save_period=args.save_period,
        plots=True,
        single_cls=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        close_mosaic=0,
        verbose=True,
    )
    summary = {
        "fold": fold,
        "initial_model": str(args.model),
        "model_cfg": str(args.model_cfg) if args.model_cfg else None,
        "initialized_checkpoint": str(initialized_checkpoint) if initialized_checkpoint else None,
        "initialization": initialization,
        "data": str(data),
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "imgsz_height_width": [544, 960],
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
        "save_period": args.save_period,
        "save_dir": str(model.trainer.save_dir),
    }
    output = args.project / fold / "training_manifest.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
