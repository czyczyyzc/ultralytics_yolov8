"""Preserve native exposure and cycle through all newly reviewed negative frames."""

from pathlib import Path

import numpy as np


class NativeExposureSampler:
    """Finite epochs avoid infinite-loader prefetch crossing the sampling boundary."""

    def __init__(self, dataset, config=None, seed=20260915):
        self.seed, self.epoch = seed, 0
        self.negative_order = np.empty(0, dtype=np.int64)
        self.budget = 0
        if config:
            paths = Path(config["negative_pool"]).read_text().splitlines()
            pool = set(paths)
            if len(pool) != len(paths) or len(pool) != config["negative_pool_count"]:
                raise ValueError("Negative pool changed or contains duplicate paths")
            indices = [i for i, row in enumerate(dataset.labels) if row["im_file"] in pool]
            if len(indices) != len(pool) or {dataset.labels[i]["im_file"] for i in indices} != pool:
                raise ValueError("Every new negative must occur exactly once in the candidate dataset")
            if any(len(dataset.labels[i]["cls"]) for i in indices):
                raise ValueError("A positive label was placed in the negative pool")
            self.budget = int(config["negatives_per_epoch"])
            if not 0 < self.budget <= len(indices):
                raise ValueError("Invalid per-epoch negative budget")
            self.negative_order = np.random.default_rng(seed).permutation(indices)
        excluded = set(self.negative_order.tolist())
        self.anchors = np.array([i for i in range(len(dataset)) if i not in excluded], dtype=np.int64)
        if config and len(self.anchors) != config["anchor_slots"]:
            raise ValueError("Native/positive anchor exposure differs from the audited schedule")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.anchors) + self.budget

    def __iter__(self):
        if self.budget:
            offsets = (self.epoch * self.budget + np.arange(self.budget)) % len(self.negative_order)
            selected = np.concatenate((self.anchors, self.negative_order[offsets]))
        else:
            selected = self.anchors.copy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 17]))
        return iter(rng.permutation(selected).tolist())


def set_native_sampler_epoch(trainer):
    trainer.train_loader.sampler.set_epoch(trainer.epoch)
