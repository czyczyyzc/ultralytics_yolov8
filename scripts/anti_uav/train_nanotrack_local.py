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
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)), help="Torch distributed local rank.")
    parser.add_argument("--backend", default="nccl", help="Distributed backend.")
    return parser.parse_args()


def parse_device_spec(device_spec: str) -> tuple[torch.device, list[int]]:
    """Parse single- or multi-GPU device strings such as `cuda:0` or `cuda:0,1,2,3`."""
    value = str(device_spec).strip()
    if value.lower() == "cpu":
        return torch.device("cpu"), []
    if not value.startswith("cuda"):
        return torch.device(value), []

    suffix = value[4:].lstrip(":")
    if not suffix:
        return torch.device("cuda:0"), [0] if torch.cuda.is_available() else []

    if "," in suffix:
        ids = [int(part.strip()) for part in suffix.split(",") if part.strip()]
        if not ids:
            raise ValueError(f"Invalid CUDA device list: {device_spec}")
        return torch.device(f"cuda:{ids[0]}"), ids

    return torch.device(value), [int(suffix)]


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


def configure_logging(log_path: Path, rank: int) -> None:
    """Configure rank-aware logging."""
    handlers = [logging.StreamHandler(sys.stdout)] if rank == 0 else []
    if rank == 0:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, bool]:
    """Initialize distributed state when launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.local_rank >= 0 and world_size > 1:
        if not args.device.startswith("cuda"):
            raise ValueError("DDP NanoTrack training currently expects a CUDA device")
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.backend)
        return torch.device(f"cuda:{args.local_rank}"), args.local_rank, world_size, True

    device, _ = parse_device_spec(args.device)
    return device, 0, 1, False


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


def reduce_loss_tensor(value):
    """Reduce DataParallel per-device scalar outputs back to a single scalar."""
    if torch.is_tensor(value) and value.ndim > 0:
        return value.mean()
    return value


def reduce_for_logging(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    """All-reduce a detached scalar for logging."""
    reduced = value.detach()
    if distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= dist.get_world_size()
    return reduced


def worker_init_fn_builder(base_seed: int, rank: int):
    """Create a deterministic worker seed initializer."""

    def _init(worker_id: int) -> None:
        seed = base_seed + rank * 1000 + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    return _init


def main() -> None:
    args = parse_args()
    cfg.merge_from_file(args.cfg)
    device, rank, world_size, distributed = setup_distributed(args)
    cfg.CUDA = device.type == "cuda"
    if rank == 0:
        Path(cfg.TRAIN.LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(cfg.TRAIN.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    configure_logging(Path(cfg.TRAIN.LOG_DIR) / "train.log", rank)
    seed_everything(args.seed)

    model = ModelBuilder().to(device).train()
    if cfg.TRAIN.PRETRAINED and Path(cfg.TRAIN.PRETRAINED).exists():
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
        LOGGER.info("Loaded pretrained checkpoint: %s", cfg.TRAIN.PRETRAINED)
    elif cfg.TRAIN.PRETRAINED:
        LOGGER.warning("Pretrained checkpoint not found, starting from scratch: %s", cfg.TRAIN.PRETRAINED)

    if distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        LOGGER.info("Enabled DDP on rank %d/%d", rank, world_size)

    dataset = BANDataset()
    if distributed and int(cfg.TRAIN.BATCH_SIZE) % world_size != 0:
        raise ValueError(f"Global batch size {cfg.TRAIN.BATCH_SIZE} must be divisible by world size {world_size}")
    local_batch_size = int(cfg.TRAIN.BATCH_SIZE) // world_size if distributed else int(cfg.TRAIN.BATCH_SIZE)
    local_num_workers = max(0, int(cfg.TRAIN.NUM_WORKERS) // world_size) if distributed else int(cfg.TRAIN.NUM_WORKERS)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        num_workers=local_num_workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
        sampler=sampler,
        worker_init_fn=worker_init_fn_builder(args.seed, rank),
    )
    base_model = model.module if isinstance(model, DDP) else model
    optimizer = build_optimizer(base_model)
    scheduler = build_scheduler(optimizer)

    best_loss = float("inf")
    history = []
    if rank == 0:
        LOGGER.info("Training config:\n%s", cfg.dump())
        LOGGER.info("Dataset length per epoch: %d", len(dataset))
        LOGGER.info("Global batch size: %d | Local batch size: %d | Global workers: %d | Local workers: %d | World size: %d", int(cfg.TRAIN.BATCH_SIZE), local_batch_size, int(cfg.TRAIN.NUM_WORKERS), local_num_workers, world_size)

    for epoch in range(int(cfg.TRAIN.EPOCH)):
        model.train()
        np.random.seed(args.seed + epoch)
        dataset.resample()
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        cls_loss_sum = 0.0
        loc_loss_sum = 0.0
        start = time.time()
        for step, batch in enumerate(loader, start=1):
            batch = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = reduce_loss_tensor(outputs["total_loss"])
            cls_loss = reduce_loss_tensor(outputs["cls_loss"])
            loc_loss = reduce_loss_tensor(outputs["loc_loss"])
            loss.backward()
            clip_grad_norm_(base_model.parameters(), float(cfg.TRAIN.GRAD_CLIP))
            optimizer.step()
            log_loss = reduce_for_logging(loss, distributed)
            log_cls_loss = reduce_for_logging(cls_loss, distributed)
            log_loc_loss = reduce_for_logging(loc_loss, distributed)
            epoch_loss += float(log_loss.item())
            cls_loss_sum += float(log_cls_loss.item())
            loc_loss_sum += float(log_loc_loss.item())
            if rank == 0 and (step % int(cfg.TRAIN.PRINT_FREQ) == 0 or step == len(loader)):
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
        if rank == 0:
            history.append({"epoch": epoch + 1, "loss": avg_loss, "cls_loss": avg_cls, "loc_loss": avg_loc, "seconds": elapsed})
            LOGGER.info("epoch=%d done loss=%.4f cls=%.4f loc=%.4f elapsed=%.1fs", epoch + 1, avg_loss, avg_cls, avg_loc, elapsed)

            checkpoint = {"epoch": epoch + 1, "state_dict": base_model.state_dict(), "optimizer": optimizer.state_dict(), "history": history}
            torch.save(checkpoint, Path(cfg.TRAIN.SNAPSHOT_DIR) / "last.pth")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, Path(cfg.TRAIN.SNAPSHOT_DIR) / "best.pth")
            if (epoch + 1) % max(args.save_every, 1) == 0:
                torch.save(checkpoint, Path(cfg.TRAIN.SNAPSHOT_DIR) / f"epoch_{epoch+1:03d}.pth")
        if distributed:
            dist.barrier()

    if rank == 0:
        history_path = Path(cfg.TRAIN.LOG_DIR) / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        LOGGER.info("Training complete. best_loss=%.4f history=%s", best_loss, history_path)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
