#!/usr/bin/env python3
"""Fine-tune only the YOLO detection classification branches on recall-safe negatives."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel


def parse_imgsz(value: str) -> list[int]:
    dimensions = [int(item) for item in value.split(",")]
    if len(dimensions) not in {1, 2} or any(item <= 0 for item in dimensions):
        raise argparse.ArgumentTypeError("imgsz must be N or H,W with positive dimensions")
    return dimensions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="class_head_only")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=parse_imgsz, default=[544, 960])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def classification_branch_parameter_ids(model) -> set[int]:
    detector = de_parallel(model).model[-1]
    if not hasattr(detector, "cv3"):
        raise TypeError("The final detector module does not expose YOLOv8 cv3 classification branches")
    return {id(parameter) for parameter in detector.cv3.parameters()}


class ClassificationHeadOnlyTrainer(DetectionTrainer):
    """Freeze feature extraction and box regression while updating class logits."""

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        trainable_ids = classification_branch_parameter_ids(model)
        trainable_names = []
        trainable_count = 0
        total_count = 0
        for parameter_name, parameter in model.named_parameters():
            parameter.requires_grad = id(parameter) in trainable_ids
            total_count += parameter.numel()
            if parameter.requires_grad:
                trainable_names.append(parameter_name)
                trainable_count += parameter.numel()
        if not trainable_names:
            raise RuntimeError("No cv3 classification parameters were selected")
        LOGGER.info(
            "Classification-head-only fine-tuning: %d/%d parameters trainable across %d tensors",
            trainable_count,
            total_count,
            len(trainable_names),
        )
        LOGGER.info("Trainable tensors: %s", ", ".join(trainable_names))
        return super().build_optimizer(model, name, lr, momentum, decay, iterations)

    def preprocess_batch(self, batch):
        # BaseTrainer calls model.train() every epoch. Keep all frozen BatchNorm
        # buffers fixed while leaving Detect in raw-output mode for loss computation.
        model = de_parallel(self.model)
        model.eval()
        model.model[-1].training = True
        return super().preprocess_batch(batch)


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    model.train(
        trainer=ClassificationHeadOnlyTrainer,
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        save=True,
        save_period=1,
        patience=args.epochs,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0.0,
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
        plots=False,
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
    )


if __name__ == "__main__":
    main()
