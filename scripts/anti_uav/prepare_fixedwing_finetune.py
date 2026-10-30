"""Build a small, immutable adaptation schedule with size-stratified old replay."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from scripts.anti_uav.label_pool_sampling import balanced_video_order, source_video_identity
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def is_tiny(row):
    if not len(row["cls"]):
        return False
    height, width = row["shape"]
    boxes = np.asarray(row["bboxes"])
    edge = np.maximum(boxes[:, 2]*width, boxes[:, 3]*height)*min(544/height, 960/width)
    return bool(((edge >= 4) & (edge <= 8)).any())


def select_replay(old, new, ratio=2., seed=20261030):
    """Use existing exposure multiplicities as weights; never synthesize new labels."""
    if not new or ratio <= 0 or any(not len(x["cls"]) for x in old+new):
        raise ValueError("Replay selection requires positive-only inputs and a positive ratio")
    count = min(len(old), math.ceil(len(new)*ratio))
    pools = [[i for i, row in enumerate(old) if is_tiny(row) == tiny] for tiny in (True, False)]
    target = math.ceil((count+len(new))*.30/.85)-sum(map(is_tiny, new))
    tiny_count = min(len(pools[0]), count, max(0, target, count-len(pools[1])))
    keys = [source_video_identity(x["im_file"]) for x in old]
    selected = []
    for category, (pool, budget) in enumerate(zip(pools, (tiny_count, count-tiny_count))):
        rng = np.random.default_rng(np.random.SeedSequence([seed, category]))
        selected.extend(balanced_video_order(pool, keys, rng)[:budget])
    if len(selected) != count:
        raise ValueError("Insufficient old replay candidates")
    return [old[i] for i in selected]


def fit_monitor(rows, positive_limit=64, negative_limit=32):
    """This subset remains in training: it is an adaptation-fit monitor, not a test."""
    groups = defaultdict(list)
    for row in rows:
        groups[(source_video_identity(row["im_file"]), bool(len(row["cls"])))].append(row)
    result = []
    for (_, positive), values in sorted(groups.items()):
        values.sort(key=lambda x: x["im_file"])
        count = min(len(values), positive_limit if positive else negative_limit)
        result.extend(values[i] for i in np.linspace(0, len(values)-1, count, dtype=int))
    return result


def read_pinned_cache(listing, config):
    listing = Path(listing)
    record = config["trusted_label_caches"][str(listing)]
    if sha256(listing) != record["source_sha256"] or sha256(record["path"]) != record["cache_sha256"]:
        raise ValueError("Frozen source list/cache changed")
    cache = np.load(record["path"], allow_pickle=True).item()
    if (cache["hash"] != record["dataset_hash"] or len(cache["labels"]) != record["entries"]
            or Counter(x["im_file"] for x in cache["labels"]) != Counter(listing.read_text().splitlines())):
        raise ValueError("Frozen cache contents do not match its image schedule")
    return cache


def write_split(path, rows, version, provenance):
    rows = sorted(rows, key=lambda x: x["im_file"])
    path.write_text("\n".join(x["im_file"] for x in rows)+"\n")
    cache_path = path.with_suffix(".cache")
    identity = hashlib.sha256((sha256(path)+provenance).encode()).hexdigest()
    cache = dict(labels=rows, hash=identity, results=(len(rows), 0,
        sum(not len(x["cls"]) for x in rows), 0, len(rows)), msgs=[], version=version)
    with cache_path.open("wb") as stream:
        np.save(stream, cache)
    return dict(path=str(cache_path), source_sha256=sha256(path), cache_sha256=sha256(cache_path),
                dataset_hash=identity, entries=len(rows))


def prepare(source_dir, output, ratio=2., seed=20261030):
    if output.exists():
        raise FileExistsError(output)
    source_path = source_dir/"train_online_gray_monitor.yaml"
    config = yaml.safe_load(source_path.read_text())
    manifest = json.loads((source_dir/"manifest.json").read_text())
    additions = manifest["latest_appended_videos"]
    if len(manifest["train_video_hashes"]) != 73 or len(additions) != 21:
        raise ValueError("Urgent adaptation is scoped to the frozen 52+21 video snapshot")
    blocked = {manifest["test_sha256"], manifest["validation_sha256"]}
    if blocked & set(manifest["train_video_hashes"]):
        raise ValueError("Held-out video hash occurs in training")
    roots = {x["image_directory"]: x for x in additions}
    train = read_pinned_cache(config["train"], config)
    val = read_pinned_cache(config["val"], config)
    old, new, negative = [], [], {}
    for row in train["labels"]:
        path = row["im_file"]
        if {"Video00004", "Video00009", "gray_val", "zoom_val"} & set(Path(path).parts):
            raise ValueError("Held-out image occurs in training")
        added = str(Path(path).parent) in roots
        if added:
            new.append(row)
        if len(row["cls"]):
            if not added:
                old.append(row)
        else:
            negative[path] = row
    expected = sum(x["positive_samples"]+x["negative_samples"] for x in additions)
    if len(new) != expected or len({x["im_file"] for x in new}) != expected:
        raise ValueError("Some newly labeled frames are missing or duplicated")
    new_positive = [x for x in new if len(x["cls"])]
    replay = select_replay(old, new_positive, ratio, seed)
    candidates = replay+new_positive+list(negative.values())
    positives = len(replay)+len(new_positive)
    negative_budget = round(positives*.15/.85)
    if not 0 < negative_budget <= len(negative):
        raise ValueError("Insufficient reviewed negatives")
    monitor = fit_monitor(new)
    validation = val["labels"]+monitor
    if {x["im_file"] for x in val["labels"]} & {x["im_file"] for x in candidates}:
        raise ValueError("Native validation overlaps training")
    output.mkdir(parents=True)
    train_path, val_path, pool_path = (output/x for x in ("train.txt", "val.txt", "negative_pool.txt"))
    provenance = sha256(source_dir/"annotation_snapshot.json")
    pins = {str(train_path): write_split(train_path, candidates, train["version"], provenance),
            str(val_path): write_split(val_path, validation, val["version"], provenance)}
    pool_path.write_text("\n".join(sorted(negative))+"\n")
    data = dict(path=str(output), train=str(train_path), val=str(val_path), names=config["names"],
        label_sampling=dict(negative_pool=str(pool_path), negative_pool_count=len(negative),
            negatives_per_epoch=negative_budget, anchor_slots=positives, target_negative_fraction=.15),
        online_replacement=config["online_replacement"],
        tiny_sampling=dict(config["tiny_sampling"]), trusted_label_caches=pins)
    path = output/"train.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    audit = dict(source=str(source_dir), source_snapshot_sha256=provenance,
        source_manifest_sha256=sha256(source_dir/"manifest.json"), new_video_count=len(additions),
        training_video_hashes=manifest["train_video_hashes"], new_videos=additions,
        new_positive=len(new_positive), new_reviewed_negative=len(new)-len(new_positive),
        replay_positive=len(replay), replay_ratio=ratio, native_epoch_positive=positives,
        native_epoch_negative=negative_budget, negative_pool=len(negative),
        fit_monitor_frames=len(monitor), native_validation_frames=len(val["labels"]),
        adaptation_roots=sorted(roots), seed=seed, excluded_policy="reuse frozen confirmed-only snapshot; uncertain/unlabelled/pending absent excluded",
        note="New-video monitor overlaps training by request; not an independent test. Original images/labels and source dataset unchanged.")
    dump(output/"manifest.json", audit)
    return path, audit
