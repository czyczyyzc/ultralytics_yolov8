"""Preserve audited exposure while building size- and video-balanced epochs."""

from collections import defaultdict, deque
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


def _video_identity(image: str) -> str:
    """Return a stable source-video key without merging every Anti-UAV ``rgb`` folder."""
    path = Path(image)
    if path.parent.name.lower() in {"rgb", "images", "frames", "imgs"}:
        return str(path.parent.parent)
    return str(path.parent)


class TinyAwareSampler:
    """Reorder the audited epoch so every batch sees tiny, normal and negative samples.

    The sampler does not invent extra exposure. It first asks ``NativeExposureSampler``
    for the exact positive/legacy-negative anchors and rotating reviewed-negative slice,
    then interleaves those same indices by target scale and source video.
    """

    def __init__(self, dataset, label_config=None, tiny_config=None, batch_size=64, seed=20260915):
        self.native = NativeExposureSampler(dataset, label_config, seed)
        self.dataset, self.batch_size = dataset, int(batch_size)
        self.seed, self.epoch = int(seed), 0
        config = tiny_config or {}
        input_hw = config.get("input_hw", (544, 960))
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        self.tiny_min = float(config.get("tiny_min_px", 4.0))
        self.tiny_max = float(config.get("tiny_max_px", 8.0))
        if self.batch_size < 1 or not 0 <= self.tiny_min < self.tiny_max:
            raise ValueError("Invalid tiny-aware sampling configuration")

        self.categories = np.empty(len(dataset), dtype=np.int8)
        self.video_keys = []
        for index, row in enumerate(dataset.labels):
            self.video_keys.append(_video_identity(row["im_file"]))
            if not len(row["cls"]):
                self.categories[index] = 2  # negative
                continue
            if row.get("bbox_format", "xywh") != "xywh" or not row.get("normalized", True):
                raise ValueError("Tiny-aware sampling requires normalized xywh labels")
            height, width = row["shape"]
            gain = min(self.input_hw[0] / height, self.input_hw[1] / width)
            boxes = np.asarray(row["bboxes"], dtype=np.float32)
            long_edges = np.maximum(boxes[:, 2] * width, boxes[:, 3] * height) * gain
            self.categories[index] = 0 if np.any(
                (long_edges >= self.tiny_min) & (long_edges <= self.tiny_max)
            ) else 1

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.native.set_epoch(epoch)

    def __len__(self):
        return len(self.native)

    def _video_round_robin(self, indices, category):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, category, 31]))
        groups = defaultdict(list)
        for index in indices:
            groups[self.video_keys[index]].append(index)
        queues = {}
        for key, values in groups.items():
            queues[key] = deque(rng.permutation(values).tolist())
        active = list(queues)
        rng.shuffle(active)
        result = []
        while active:
            next_active = []
            for key in active:
                result.append(queues[key].popleft())
                if queues[key]:
                    next_active.append(key)
            active = next_active
            rng.shuffle(active)
        return result

    def __iter__(self):
        selected = list(self.native)
        pools = [
            self._video_round_robin([i for i in selected if self.categories[i] == category], category)
            for category in range(3)
        ]
        totals = np.asarray([len(pool) for pool in pools], dtype=np.int64)
        if not totals[0] or not totals[1] or not totals[2]:
            raise ValueError(f"Tiny-aware epoch has an empty category: {totals.tolist()}")

        emitted = np.zeros(3, dtype=np.int64)
        output = []
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 43]))
        for start in range(0, len(selected), self.batch_size):
            end = min(start + self.batch_size, len(selected))
            remaining = totals - emitted
            raw = (end - start) * remaining / remaining.sum()
            take = np.floor(raw).astype(np.int64)
            missing = end - start - int(take.sum())
            for category in np.argsort(-(raw - take))[:missing]:
                take[category] += 1
            cumulative = emitted + take
            batch = []
            for category, count in enumerate(take.tolist()):
                batch.extend(pools[category][emitted[category] : emitted[category] + count])
            emitted = cumulative
            rng.shuffle(batch)
            output.extend(batch)
        if emitted.tolist() != totals.tolist() or sorted(output) != sorted(selected):
            raise AssertionError("Tiny-aware interleave changed the audited epoch exposure")
        return iter(output)


def set_native_sampler_epoch(trainer):
    trainer.train_loader.sampler.set_epoch(trainer.epoch)
    dataset = trainer.train_loader.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(trainer.epoch)
