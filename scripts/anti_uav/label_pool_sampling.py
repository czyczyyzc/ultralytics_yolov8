"""Preserve audited exposure while building size- and video-balanced epochs."""

from collections import defaultdict, deque
import math
from pathlib import Path
import re

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


def source_video_identity(image):
    """Resolve flat Anti-UAV filenames instead of merging their rgb directory."""
    path = Path(image)
    if "anti_uav300_yolo" in path.parts:
        match = re.fullmatch(r"(.+)_(?:rgb|ir)_\d+", path.stem, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot identify flat Anti-UAV source video: {image}")
        return str(path.parent / match.group(1))
    return _video_identity(image)


def balanced_video_order(indices, video_keys, rng):
    """Spread each video's exact exposures across the whole epoch, not its prefix."""
    groups = defaultdict(list)
    for index in indices:
        groups[video_keys[index]].append(index)
    values, priorities = [], []
    for key in sorted(groups):
        group = rng.permutation(groups[key])
        values.append(group)
        priorities.append((np.arange(len(group), dtype=float) + rng.random()) / len(group))
    if not values:
        return []
    order = np.argsort(np.concatenate(priorities), kind="stable")
    return np.concatenate(values)[order].tolist()


class BalancedTinyAwareSampler(TinyAwareSampler):
    """Opt-in repair: retain native positives, bound tiny repeats, balance negatives."""

    def __init__(self, dataset, label_config=None, tiny_config=None, batch_size=64, seed=20260915):
        if not label_config:
            raise ValueError("balanced_v2 requires an audited rotating negative pool")
        super().__init__(dataset, label_config, tiny_config, batch_size, seed)
        config = tiny_config or {}
        self.video_keys = [source_video_identity(row["im_file"]) for row in dataset.labels]
        self.tiny_fraction = float(config.get("target_tiny_fraction", .30))
        self.negative_fraction = float(label_config.get("target_negative_fraction", .15))
        self.max_extra_repeat = int(config.get("max_extra_repeat_per_image", 1))
        if not (0 < self.tiny_fraction < 1-self.negative_fraction and 0 < self.negative_fraction < 1):
            raise ValueError("Invalid tiny/negative fractions")
        if not 1 <= self.max_extra_repeat <= 3:
            raise ValueError("Additional per-image repeat bound must be 1..3")
        native_positive = int((self.categories != 2).sum())
        native_tiny = int((self.categories == 0).sum())
        other_positive = native_positive-native_tiny
        target_tiny = max(native_tiny, math.ceil(
            self.tiny_fraction*other_positive/(1-self.negative_fraction-self.tiny_fraction)))
        self.extra_tiny = target_tiny-native_tiny
        by_path = {}
        for i in np.flatnonzero(self.categories == 0):
            by_path.setdefault(dataset.labels[int(i)]["im_file"], int(i))
        self.extra_candidates = list(by_path.values())
        if self.extra_tiny > len(self.extra_candidates)*self.max_extra_repeat:
            raise ValueError("Tiny target cannot be met without exceeding the repeat safety bound")
        total_negative = int((self.categories == 2).sum())
        fixed_negative = total_negative-len(self.native.negative_order)
        desired_negative = round((native_positive+self.extra_tiny)*self.negative_fraction/(1-self.negative_fraction))
        corrected = dict(label_config, negatives_per_epoch=desired_negative-fixed_negative)
        self.native = NativeExposureSampler(dataset, corrected, seed)
        self.audit = dict(policy="balanced_v2", native_positive=native_positive, native_tiny=native_tiny,
            extra_tiny=self.extra_tiny, epoch_positive=native_positive+self.extra_tiny,
            epoch_negative=desired_negative, epoch_total=len(self.native)+self.extra_tiny,
            tiny_fraction=target_tiny/(len(self.native)+self.extra_tiny),
            negative_fraction=desired_negative/(len(self.native)+self.extra_tiny),
            max_extra_repeat_per_image=self.max_extra_repeat,
            exact_native_positive_exposure_preserved=True)

    def __len__(self):
        return len(self.native)+self.extra_tiny

    def _extra_indices(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 97]))
        # Additional exposure is video-balanced, with a hard bound per unique image.
        groups = defaultdict(list)
        for i in self.extra_candidates:
            groups[self.video_keys[i]].extend([i]*self.max_extra_repeat)
        queues = {key: deque(rng.permutation(values).tolist()) for key, values in sorted(groups.items())}
        active, result = list(queues), []
        while active and len(result) < self.extra_tiny:
            rng.shuffle(active)
            following = []
            for key in active:
                if len(result) == self.extra_tiny:
                    break
                result.append(queues[key].popleft())
                if queues[key]:
                    following.append(key)
            active = following
        if len(result) != self.extra_tiny:
            raise AssertionError("Insufficient bounded tiny repeat candidates")
        return result

    def __iter__(self):
        selected = list(self.native)+self._extra_indices()
        pools = []
        for category in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, category, 113]))
            pools.append(balanced_video_order([i for i in selected if self.categories[i] == category],
                                              self.video_keys, rng))
        totals = np.array([len(pool) for pool in pools], dtype=np.int64)
        if not totals.all():
            raise ValueError("balanced_v2 requires nonempty tiny, ordinary and negative groups")
        emitted, output = np.zeros(3, dtype=np.int64), []
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 43]))
        for start in range(0, len(selected), self.batch_size):
            size = min(self.batch_size, len(selected)-start)
            remaining = totals-emitted
            raw = size*remaining/remaining.sum()
            take = np.floor(raw).astype(np.int64)
            for category in np.argsort(-(raw-take))[:size-int(take.sum())]:
                take[category] += 1
            batch = []
            for category in range(3):
                batch.extend(pools[category][emitted[category]:emitted[category]+take[category]])
            emitted += take
            rng.shuffle(batch)
            output.extend(batch)
        if emitted.tolist() != totals.tolist() or sorted(output) != sorted(selected):
            raise AssertionError("balanced_v2 changed intended epoch exposures")
        return iter(output)


def make_exposure_sampler(dataset, label_config=None, tiny_config=None, batch_size=64, seed=20260915):
    if not tiny_config:
        return NativeExposureSampler(dataset, label_config, seed)
    policy = tiny_config.get("policy", "legacy")
    if policy not in ("legacy", "balanced_v2"):
        raise ValueError(f"Unknown tiny sampling policy: {policy}")
    cls = BalancedTinyAwareSampler if policy == "balanced_v2" else TinyAwareSampler
    return cls(dataset, label_config, tiny_config, batch_size, seed)


def set_native_sampler_epoch(trainer):
    trainer.train_loader.sampler.set_epoch(trainer.epoch)
    dataset = trainer.train_loader.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(trainer.epoch)
