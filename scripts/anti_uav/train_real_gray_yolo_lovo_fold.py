#!/usr/bin/env python3
"""Fine-tune one YOLOv8n real-gray LOVO fold with an Anti-UAV300 RGB rehearsal mix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold = args.fold_dir.name
    data = args.fold_dir / "train_rgb_monitor.yaml"
    if not data.exists():
        raise FileNotFoundError(data)
    model = YOLO(str(args.model))
    model.train(
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
        lr0=0.001,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
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
        save_period=5,
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
        "data": str(data),
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "imgsz_height_width": [544, 960],
        "save_dir": str(model.trainer.save_dir),
    }
    output = args.project / fold / "training_manifest.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
