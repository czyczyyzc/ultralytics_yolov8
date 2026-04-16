#!/usr/bin/env python3
"""Train the vendored NanoTrack package on converted Anti-UAV300 crops."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
VENDOR_ROOT = ROOT / "third_party" / "nanotrack_vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from nanotrack.core.config import cfg
from nanotrack.datasets.dataset import BANDataset
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.utils.model_load import load_pretrain

LOGGER = logging.getLogger("nanotrack_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="NanoTrack yaml config path.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=123456, help="Random seed.")
    parser.add_argument("--save-every", type=int, default=5, help="Checkpoint interval in epochs.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: ModelBuilder) -> torch.optim.Optimizer:
    for param in model.backbone.parameters():
        param.requires_grad = False
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    if cfg.BACKBONE.TRAIN_LAYERS:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
    trainable = [{"params": filter(lambda x: x.requires_grad, model.backbone.parameters()), "lr": cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]
    if cfg.ADJUST.ADJUST:
        trainable.append({"params": model.neck.parameters(), "lr": cfg.TRAIN.BASE_LR})
    trainable.append({"params": model.ban_head.parameters(), "lr": cfg.TRAIN.BASE_LR})
    return torch.optim.SGD(trainable, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)


def build_scheduler(optimizer: torch.optim.Optimizer):
    warmup_epochs = int(cfg.TRAIN.LR_WARMUP.EPOCH)
    total_epochs = int(cfg.TRAIN.EPOCH)
    start_lr = float(cfg.TRAIN.LR_WARMUP.KWARGS.get("start_lr", cfg.TRAIN.BASE_LR / 5.0))
    end_lr = float(cfg.TRAIN.LR.KWARGS.get("end_lr", cfg.TRAIN.BASE_LR / 10.0))
    base_lr = float(cfg.TRAIN.BASE_LR)

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            alpha = epoch / max(warmup_epochs - 1, 1)
            value = start_lr + alpha * (base_lr - start_lr)
        else:
            remaining = max(total_epochs - warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / remaining
            value = base_lr * ((end_lr / base_lr) ** max(progress, 0.0))
        return value / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def prepare_batch(batch: dict, device: torch.device) -> dict:
    prepared = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if torch.is_tensor(value):
            prepared[key] = value.to(device=device, non_blocking=True)
        else:
            prepared[key] = value
    return prepared


def main() -> None:
    args = parse_args()
    cfg.merge_from_file(args.cfg)
    device = torch.device(args.device)
    cfg.CUDA = device.type == "cuda"
    Path(cfg.TRAIN.LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.TRAIN.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(cfg.TRAIN.LOG_DIR) / "train.log", encoding="utf-8"),
        ],
    )
    seed_everything(args.seed)

    model = ModelBuilder().to(device).train()
    if cfg.TRAIN.PRETRAINED and Path(cfg.TRAIN.PRETRAINED).exists():
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
        LOGGER.info("Loaded pretrained checkpoint: %s", cfg.TRAIN.PRETRAINED)
    elif cfg.TRAIN.PRETRAINED:
        LOGGER.warning("Pretrained checkpoint not found, starting from scratch: %s", cfg.TRAIN.PRETRAINED)

    dataset = BANDataset()
    loader = DataLoader(dataset, batch_size=int(cfg.TRAIN.BATCH_SIZE), num_workers=int(cfg.TRAIN.NUM_WORKERS), pin_memory=device.type == "cuda", shuffle=False)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    best_loss = float("inf")
    history = []
    LOGGER.info("Training config:\n%s", cfg.dump())
    LOGGER.info("Dataset length per epoch: %d", len(dataset))

    for epoch in range(int(cfg.TRAIN.EPOCH)):
        model.train()
        dataset.resample()
        epoch_loss = 0.0
        cls_loss_sum = 0.0
        loc_loss_sum = 0.0
        start = time.time()
        for step, batch in enumerate(loader, start=1):
            batch = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = outputs["total_loss"]
            loss.backward()
            clip_grad_norm_(model.parameters(), float(cfg.TRAIN.GRAD_CLIP))
            optimizer.step()
            epoch_loss += float(loss.item())
            cls_loss_sum += float(outputs["cls_loss"].item())
            loc_loss_sum += float(outputs["loc_loss"].item())
            if step % int(cfg.TRAIN.PRINT_FREQ) == 0 or step == len(loader):
                LOGGER.info(
                    "epoch=%d step=%d/%d loss=%.4f cls=%.4f loc=%.4f lr=%.6f",
                    epoch + 1,
                    step,
                    len(loader),
                    epoch_loss / step,
                    cls_loss_sum / step,
                    loc_loss_sum / step,
                    optimizer.param_groups[-1]["lr"],
                )
        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)
        avg_cls = cls_loss_sum / max(len(loader), 1)
        avg_loc = loc_loss_sum / max(len(loader), 1)
        elapsed = time.time() - start
        history.append({"epoch": epoch + 1, "loss": avg_loss, "cls_loss": avg_cls, "loc_loss": avg_loc, "seconds": elapsed})
        LOGGER.info("epoch=%d done loss=%.4f cls=%.4f loc=%.4f elapsed=%.1fs", epoch + 1, avg_loss, avg_cls, avg_loc, elapsed)

        checkpoint = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history}
        torch.save(checkpoint, Path(cfg.TRAIN.SNAPSHOT_DIR) / "last.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, Path(cfg.TRAIN.SNAPSHOT_DIR) / "best.pth")
        if (epoch + 1) % max(args.save_every, 1) == 0:
            torch.save(checkpoint, Path(cfg.TRAIN.SNAPSHOT_DIR) / f"epoch_{epoch+1:03d}.pth")

    history_path = Path(cfg.TRAIN.LOG_DIR) / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    LOGGER.info("Training complete. best_loss=%.4f history=%s", best_loss, history_path)


if __name__ == "__main__":
    main()
