#!/usr/bin/env python3
"""Write a NanoTrack training config that points to converted Anti-UAV300 data."""

from __future__ import annotations

import argparse
from pathlib import Path


VARIANT_PRESETS = {
    "v1": {
        "ban_version": "v2",
        "backbone_type": "mobilenetv3_small",
        "backbone_pretrained": "./models/pretrained/mobilenetv3_small_1.0.pth",
        "adjust_channels": 64,
        "point_stride": 16,
        "window_influence": 0.462,
        "penalty_k": 0.148,
        "track_lr": 0.390,
        "track_output_size": 16,
        "train_output_size": 16,
        "batch_size": 32,
        "videos_per_epoch": 100000,
    },
    "v2": {
        "ban_version": "v2",
        "backbone_type": "mobilenetv3_small",
        "backbone_pretrained": "./models/pretrained/mobilenetv3_small_1.0.pth",
        "adjust_channels": 64,
        "point_stride": 16,
        "window_influence": 0.490,
        "penalty_k": 0.150,
        "track_lr": 0.385,
        "track_output_size": 16,
        "train_output_size": 16,
        "batch_size": 32,
        "videos_per_epoch": 120000,
    },
    "v3": {
        "ban_version": "v3",
        "backbone_type": "mobilenetv3_small_v3",
        "backbone_pretrained": "./models/pretrained/mobilenetv3_small_1.0.pth",
        "adjust_channels": 96,
        "point_stride": 16,
        "window_influence": 0.455,
        "penalty_k": 0.138,
        "track_lr": 0.348,
        "track_output_size": 15,
        "train_output_size": 15,
        "batch_size": 64,
        "videos_per_epoch": 120000,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Path to the output NanoTrack yaml config.")
    parser.add_argument("--dataset-name", default="ANTIUAV300", help="Dataset name key used inside the NanoTrack config.")
    parser.add_argument("--crop-root", type=Path, required=True, help="Path to the modality-specific crop511 root.")
    parser.add_argument("--train-json", type=Path, required=True, help="Path to the modality-specific train.json.")
    parser.add_argument("--variant", choices=sorted(VARIANT_PRESETS), default="v2", help="NanoTrack architecture preset.")
    parser.add_argument("--pretrained", default="", help="Optional pretrained checkpoint used for fine-tuning.")
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Directory for NanoTrack checkpoints.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory for NanoTrack logs.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override train batch size. 0 uses the preset default.")
    parser.add_argument("--num-workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--videos-per-epoch", type=int, default=0, help="Override dataset videos per epoch. 0 uses preset.")
    parser.add_argument("--frame-range", type=int, default=30, help="Positive pair sampling range.")
    parser.add_argument("--base-lr", type=float, default=0.005, help="Base learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = VARIANT_PRESETS[args.variant]
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or preset["batch_size"]
    videos_per_epoch = args.videos_per_epoch or preset["videos_per_epoch"]
    pretrained = args.pretrained.strip()

    config_text = f"""META_ARC: "nanotrack"

BACKBONE:
    TYPE: "{preset['backbone_type']}"
    KWARGS:
        used_layers: [4]
    PRETRAINED: "{preset['backbone_pretrained']}"
    TRAIN_LAYERS: ['features']
    TRAIN_EPOCH: 10
    LAYERS_LR: 0.1

ADJUST:
    ADJUST: True
    TYPE: 'AdjustLayer'
    KWARGS:
        in_channels: {preset['adjust_channels']}
        out_channels: {preset['adjust_channels']}

BAN:
    BAN: True
    TYPE: DepthwiseBAN
    VERSION: "{preset['ban_version']}"
    KWARGS:
        in_channels: {preset['adjust_channels']}
        out_channels: {preset['adjust_channels']}

CUDA: True

POINT:
    STRIDE: {preset['point_stride']}

TRACK:
    TYPE: 'NanoTracker'
    WINDOW_INFLUENCE: {preset['window_influence']}
    PENALTY_K: {preset['penalty_k']}
    LR: {preset['track_lr']}
    EXEMPLAR_SIZE: 127
    INSTANCE_SIZE: 255
    BASE_SIZE: 7
    CONTEXT_AMOUNT: 0.5
    OUTPUT_SIZE: {preset['track_output_size']}

TRAIN:
    EPOCH: {args.epochs}
    START_EPOCH: 0
    BATCH_SIZE: {batch_size}
    NUM_WORKERS: {args.num_workers}
    BASE_LR: {args.base_lr}
    CLS_WEIGHT: 1.0
    LOC_WEIGHT: 1.0
    NUM_CONVS: 4
    BASE_SIZE: 7
    OUTPUT_SIZE: {preset['train_output_size']}
    RESUME: ''
    PRETRAINED: "{pretrained}"
    SNAPSHOT_DIR: "{args.snapshot_dir.expanduser().resolve()}"
    LOG_DIR: "{args.log_dir.expanduser().resolve()}"
    LR:
        TYPE: 'log'
        KWARGS:
            start_lr: {args.base_lr}
            end_lr: {args.base_lr / 10.0}
    LR_WARMUP:
        TYPE: 'step'
        EPOCH: 5
        KWARGS:
            start_lr: {args.base_lr / 5.0}
            end_lr: {args.base_lr}
            step: 1

DATASET:
    NAMES:
    - '{args.dataset_name}'
    VIDEOS_PER_EPOCH: {videos_per_epoch}
    TEMPLATE:
        SHIFT: 4
        SCALE: 0.05
        BLUR: 0.0
        FLIP: 0.0
        COLOR: 1.0
    SEARCH:
        SHIFT: 64
        SCALE: 0.18
        BLUR: 0.2
        FLIP: 0.0
        COLOR: 1.0
    NEG: 0.2
    GRAY: 0.0
    {args.dataset_name}:
        ROOT: "{args.crop_root.expanduser().resolve()}"
        ANNO: "{args.train_json.expanduser().resolve()}"
        FRAME_RANGE: {args.frame_range}
        NUM_USE: -1
"""
    output_path.write_text(config_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
