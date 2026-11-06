#!/usr/bin/env python3
"""Append audited public positives to the unchanged native curriculum for research."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_positive_label(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        values = [float(x) for x in line.split()]
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError(f"Invalid YOLO row: {path}")
        cls, x, y, w, h = values
        if cls != 0 or min(w, h) <= 0 or not all(0 <= v <= 1 for v in values[1:]):
            raise ValueError(f"Unexpected class or box: {path}")
        if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
            raise ValueError(f"Box extends beyond the image: {path}")
        rows.append(values)
    if not rows:
        raise ValueError(f"Public supplement must be an annotated positive: {path}")
    return rows


def sampling_extension(config, base_count, added_positive):
    result = deepcopy(config)
    sample = result["label_sampling"]
    if base_count - sample["negative_pool_count"] != sample["anchor_slots"]:
        raise ValueError("Baseline schedule differs from its audited sampler")
    old_epoch = sample["anchor_slots"] + sample["negatives_per_epoch"]
    fraction = float(sample["target_negative_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("Invalid native negative fraction")
    old_negative = round(old_epoch * fraction)
    old_positive = old_epoch - old_negative
    fixed_negative = old_negative - sample["negatives_per_epoch"]
    desired = round((old_positive + added_positive) * fraction / (1-fraction))
    budget = desired - fixed_negative
    if not 0 < budget <= sample["negative_pool_count"]:
        raise ValueError("Cannot retain negative fraction with the reviewed pool")
    sample.update(anchor_slots=sample["anchor_slots"]+added_positive, negatives_per_epoch=budget)
    return result, dict(native_epoch=old_epoch, native_positive=old_positive,
        native_negative=old_negative, added_positive=added_positive,
        epoch_positive=old_positive+added_positive, epoch_negative=desired,
        epoch_slots=old_positive+added_positive+desired,
        negative_fraction=desired/(old_positive+added_positive+desired))


def inspect_image(path):
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    identity = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    return identity, image.shape[:2]


def check_groups(records):
    groups = {}
    for record in records:
        key = record["dataset"], record["group"]
        split = record["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown public split: {split}")
        if key in groups and groups[key] != split:
            raise ValueError(f"A public source group crosses splits: {key}")
        groups[key] = split


def completed_manifest(path):
    summary = json.loads((path.parent / "summary.json").read_text())
    if summary.get("complete") is not True or summary.get("manifest_sha256") != sha256(path):
        raise ValueError("Public ingestion is incomplete or its manifest changed")
    return summary


def positive_candidates(records):
    candidates, excluded = [], []
    for row in records:
        if row["split"] != "train":
            continue
        if type(row.get("boxes")) is not int or row["boxes"] < 0:
            raise ValueError("Public manifest must declare its valid box count")
        if row["boxes"] == 0:
            excluded.append(dict(image=row["image"], reason="no_positive_gt_not_assumed_negative"))
        else:
            candidates.append(row)
    return candidates, excluded


def reviewed_negative_records(path, training_hashes, snapshot_sha256):
    summary = completed_manifest(path)
    if summary.get("source_snapshot_sha256") != snapshot_sha256:
        raise ValueError("Negative supplement uses a different approved snapshot")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    identities, paths = set(), set()
    for row in rows:
        digest = row.get("raw_video_sha256", row.get("source_sha256"))
        identity = digest, row.get("frame_index")
        if (digest not in training_hashes or row.get("split") != "train" or row.get("boxes") != 0
                or row.get("status") != "approved" or row.get("reviewed_status") != "absent"
                or row.get("revision") is None or row.get("revision") != row.get("approved_revision")
                or row.get("snapshot_sha256") != snapshot_sha256):
            raise ValueError("Unreviewed, heldout, or positive frame in negative supplement")
        if identity in identities or row["image"] in paths or Path(row["label"]).read_text().strip():
            raise ValueError("Duplicate or nonempty negative label")
        identities.add(identity); paths.add(row["image"])
    return rows


def fixedwing_holdouts(records):
    result=[]
    for row in records:
        if row["split"] not in {"val","test"} or not row.get("boxes"):
            continue
        types=row.get("aircraft_types") or row.get("source_classes") or [row.get("source_class","")]
        if types and {str(x).casefold() for x in types}=={"airplane"}:
            result.append(row)
    return result


def build_fixedwing_holdout(records, config, version, output):
    import numpy as np
    import yaml
    rows=fixedwing_holdouts(records)
    if not rows:return
    seen,selected,labels=set(),[],[]
    for row in rows:
        identity,shape=inspect_image(row["image"])
        if identity in seen:continue
        seen.add(identity)
        boxes=np.asarray(read_positive_label(row["label"]),dtype=np.float32)
        if len(boxes)!=row["boxes"]:raise ValueError("Fixed-wing holdout labels changed")
        selected.append(row)
        labels.append(dict(im_file=row["image"],shape=shape,cls=boxes[:,:1],bboxes=boxes[:,1:],
            segments=[],keypoints=None,normalized=True,bbox_format="xywh"))
    listing=output/'fixedwing_public_holdout.txt'
    listing.write_text('\n'.join(row['image'] for row in selected)+'\n')
    labels.sort(key=lambda row:row['im_file'])
    cache=dict(labels=labels,hash=sha256(listing),version=version,results=(len(labels),0,0,0,len(labels)),msgs=[])
    with listing.with_suffix('.cache').open('wb') as stream:np.save(stream,cache,allow_pickle=True)
    evaluation=deepcopy(config);evaluation['val']=str(listing)
    evaluation['trusted_label_caches'][str(listing)]=dict(path=str(listing.with_suffix('.cache')),
        source_sha256=sha256(listing),cache_sha256=sha256(listing.with_suffix('.cache')),
        dataset_hash=cache['hash'],entries=len(labels))
    (output/'fixedwing_public_holdout.yaml').write_text(yaml.safe_dump(evaluation,sort_keys=False))
    dump(output/'fixedwing_public_holdout_protocol.json',dict(frames=len(labels),
        gt=sum(len(row['cls']) for row in labels),selection_used=False,positive_only=True,
        source_types=['airplane'],official_or_whole_video_heldouts=True,
        limitation='Positive-frame subset, not full-video background FP or guaranteed cross-dataset/scene independence',
        listing_sha256=sha256(listing),records=selected))


def bind_control(args):
    import yaml
    control = json.loads((args.baseline_run / "protocol.json").read_text())
    if Path(control["data"]).resolve() != args.baseline_yaml.resolve():
        raise ValueError("Baseline YAML is not the control run's data")
    source = yaml.safe_load(args.baseline_yaml.read_text())
    if source.get("trusted_label_caches") != control["trusted_label_caches"]:
        raise ValueError("Baseline cache pins differ from the control run")
    return control


def replacement_view(source, replacement):
    """An asset-only view may change the catalog, never the reviewed backgrounds."""
    views = []
    for root in (Path(source), Path(replacement)):
        summary = json.loads((root / "summary.json").read_text())
        if summary.get("stage") != "complete" or summary.get("is_smoke_subset"):
            raise ValueError("Replacement background cache is incomplete")
        if sha256(root / "index.json") != summary["index_sha256"]:
            raise ValueError("Replacement index changed")
        index = json.loads((root / "index.json").read_text())
        if sha256(index["catalog"]) != index["catalog_sha256"]:
            raise ValueError("Replacement catalog changed")
        catalog = json.loads(Path(index["catalog"]).read_text())
        ids = [r["id"] for r in sorted(catalog["records"], key=lambda r: int(r["id"])) if r["id"] not in {"24", "25"}]
        if ids != index["asset_ids"] or len(ids) != summary["asset_count"]:
            raise ValueError("Replacement asset identities differ")
        for row in catalog["records"]:
            if row["id"] in {"24", "25"}:
                continue
            if sha256(Path(index["catalog"]).parent / row["cutout"]) != row["cutout_sha256"]:
                raise ValueError("Replacement asset bytes changed")
        views.append((root, index, summary))
    (old_root, old, _), (new_root, new, new_summary) = views
    if old["images"] != new["images"] or old["heldout_sha256"] != new["heldout_sha256"]:
        raise ValueError("An asset-only view changed its reviewed image/background scope")
    for pack in {r["pack"] for r in old["images"].values()}:
        if (old_root / pack).resolve() != (new_root / pack).resolve():
            raise ValueError("An asset-only view changed a background pack")
    return dict(source_cache=str(old_root), cache=str(new_root), asset_count=new_summary["asset_count"],
        catalog=new["catalog"], catalog_sha256=new["catalog_sha256"],
        index_sha256=sha256(new_root / "index.json"), backgrounds_unchanged=True)


def prepare(args):
    import cv2
    import numpy as np
    import yaml
    cv2.setNumThreads(1)
    source = yaml.safe_load(args.baseline_yaml.read_text())
    base_manifest = json.loads((args.baseline_yaml.parent / "manifest.json").read_text())
    if len(base_manifest["train_video_hashes"]) != 69:
        raise ValueError("This research experiment requires the frozen 69-video control")
    if source.get("online_scale"):
        raise ValueError("Do not add scale-view augmentation to this data-only experiment")
    args.data_output.mkdir(parents=True, exist_ok=False)
    base = Path(source["train"]).read_text().splitlines()
    val_text = Path(source["val"]).read_text()
    from scripts.anti_uav.run_native_pool_comparison import HOLDOUT
    protected = set(val_text.splitlines())
    protected.update(Path(yaml.safe_load(HOLDOUT.read_text())["val"]).read_text().splitlines())
    records, public_sources = [], []
    negative_manifest = getattr(args, "reviewed_negative_manifest", None)
    negative_identity = sha256(negative_manifest) if negative_manifest else None
    negative_rows = []
    negative_cache = []
    if negative_manifest:
        audit = json.loads(args.approved_audit.read_text())
        negative_rows = reviewed_negative_records(negative_manifest,
            set(base_manifest["train_video_hashes"]), audit["snapshot_sha256"])
        if set(row["image"] for row in negative_rows) & (set(base) | protected):
            raise ValueError("New reviewed negatives overlap native data or holdouts")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row, (_, shape) in zip(negative_rows, pool.map(inspect_image, [r["image"] for r in negative_rows])):
                if tuple(shape) != (row["height"], row["width"]):
                    raise ValueError("Reviewed negative image geometry changed")
                negative_cache.append(dict(im_file=row["image"], shape=shape,
                    cls=np.empty((0,1),dtype=np.float32), bboxes=np.empty((0,4),dtype=np.float32),
                    segments=[], keypoints=None, normalized=True, bbox_format="xywh"))
    for manifest in args.public_manifest:
        completed_manifest(manifest)
        entries = [json.loads(line) for line in manifest.read_text().splitlines()]
        records.extend(entries)
        public_sources.append(dict(path=str(manifest), sha256=sha256(manifest), entries=len(entries)))
    check_groups(records)
    heldout = [x["image"] for x in records if x["split"] != "train"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        protected_hashes = {x[0] for x in pool.map(inspect_image, sorted(protected | set(heldout)))}
    candidates, excluded = positive_candidates(records)
    images = [x["image"] for x in candidates]
    if len(set(images)) != len(images) or set(images) & set(base):
        raise ValueError("Public manifest contains repeated or native image paths")
    accepted, labels, seen = [], [], set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, (identity, shape) in zip(candidates, pool.map(inspect_image, images)):
            if identity in protected_hashes or identity in seen:
                excluded.append(dict(image=row["image"], reason="exact_pixel_duplicate_heldout_or_public"))
                continue
            seen.add(identity)
            boxes = np.asarray(read_positive_label(row["label"]), dtype=np.float32)
            if len(boxes) != row["boxes"]:
                raise ValueError("Public label count differs from its manifest")
            accepted.append(dict(row, pixel_sha256=identity))
            labels.append(dict(im_file=row["image"], shape=shape, cls=boxes[:, :1],
                bboxes=boxes[:, 1:], segments=[], keypoints=None, normalized=True, bbox_format="xywh"))
    if not accepted:
        raise ValueError("No safe public training positives remain")
    extended = deepcopy(source)
    if negative_rows:
        old_pool = Path(source["label_sampling"]["negative_pool"]).read_text().splitlines()
        if len(set(old_pool)) != len(old_pool) or len(old_pool) != source["label_sampling"]["negative_pool_count"]:
            raise ValueError("Reviewed negative pool changed")
        pool = args.data_output/"negative_pool.txt"
        pool.write_text("\n".join(old_pool+[r["image"] for r in negative_rows])+"\n")
        extended["label_sampling"].update(negative_pool=str(pool), negative_pool_count=len(old_pool)+len(negative_rows))
    config, exposure = sampling_extension(extended, len(base)+len(negative_rows), len(accepted))
    if (exposure["native_positive"], exposure["native_negative"]) != (
            base_manifest["epoch_positive"], base_manifest["epoch_negative"]):
        raise ValueError("Native exposure counts differ from the frozen manifest")
    # Keep the exact repeated native prefix, not merely the same set of files.
    train = args.data_output / "train_public.txt"
    val = args.data_output / "val_monitor.txt"
    train.write_text("\n".join(base + [row["image"] for row in negative_rows] + [row["image"] for row in accepted]) + "\n")
    val.write_text(val_text)
    base_cache_path = Path(source["train"]).with_suffix(".cache")
    pinned = source["trusted_label_caches"][str(Path(source["train"]))]
    if sha256(base_cache_path) != pinned["cache_sha256"] or sha256(source["train"]) != pinned["source_sha256"]:
        raise ValueError("Native cache/list differs from the baseline protocol")
    cache = np.load(base_cache_path, allow_pickle=True).item()
    if Counter(x["im_file"] for x in cache["labels"]) != Counter(base):
        raise ValueError("Native cache does not preserve the repeated image schedule")
    cache["labels"] = sorted(cache["labels"] + negative_cache + labels, key=lambda row: row["im_file"])
    count = len(cache["labels"])
    cache.update(hash=hashlib.sha256((sha256(base_cache_path)+sha256(train)+json.dumps(public_sources)).encode()).hexdigest(),
        results=(count, 0, sum(not len(x["cls"]) for x in cache["labels"]), 0, count), msgs=[])
    with train.with_suffix(".cache").open("wb") as stream:
        np.save(stream, cache, allow_pickle=True)
    val_source = Path(source["val"])
    val_pin = source["trusted_label_caches"][str(val_source)]
    if (sha256(val_source) != val_pin["source_sha256"] or
            sha256(val_source.with_suffix(".cache")) != val_pin["cache_sha256"]):
        raise ValueError("Native validation cache/list changed")
    if any(sha256(row["path"]) != row["sha256"] for row in public_sources):
        raise ValueError("A public manifest changed during preparation")
    if negative_manifest and sha256(negative_manifest) != negative_identity:
        raise ValueError("Reviewed negative manifest changed during preparation")
    shutil.copy2(val_source.with_suffix(".cache"), val.with_suffix(".cache"))
    config.update(path=str(args.data_output), train=str(train), val=str(val))
    asset_view = None
    if getattr(args, "background_cache", None):
        asset_view = replacement_view(source["online_replacement"]["cache"], args.background_cache)
        config["online_replacement"] = dict(config["online_replacement"], cache=str(args.background_cache))
    pins = {}
    for listing in (train, val):
        cached = np.load(listing.with_suffix(".cache"), allow_pickle=True).item()
        pins[str(listing)] = dict(path=str(listing.with_suffix(".cache")), source_sha256=sha256(listing),
            cache_sha256=sha256(listing.with_suffix(".cache")), dataset_hash=str(cached["hash"]), entries=len(cached["labels"]))
    config["trusted_label_caches"] = pins
    build_fixedwing_holdout(records,config,cache['version'],args.data_output)
    output = args.data_output / "train_public_gray_monitor.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = dict(base_manifest, schema="anti_uav.public_research.v1", source_dataset=str(args.baseline_yaml.parent),
        public_sources=public_sources, public_training_images=len(accepted), **exposure,
        append_only_entries=count, candidate_entries=count, append_only_positive=exposure["epoch_positive"],
        negative=sum(not len(row["cls"]) for row in cache["labels"]), final_entries=exposure["epoch_slots"],
        final_negative_fraction=exposure["negative_fraction"],
        additional_negatives_per_epoch=config["label_sampling"]["negatives_per_epoch"],
        native_schedule_sha256=sha256(train), validation_schedule_sha256=sha256(val),
        research_only=True, commercial_clearance=False, native_prefix_preserved=True,
        source_train_sha256=sha256(source["train"]), source_validation_sha256=sha256(source["val"]),
        external_splits_not_used_for_selection=True,
        augmentation_assets=asset_view,
        additional_reviewed_negative_images=len(negative_rows),
        reviewed_negative_manifest=str(negative_manifest) if negative_manifest else None,
        reviewed_negative_manifest_sha256=negative_identity,
        deduplication="Exact decoded pixels against public heldouts, native validation/test, and public train; not perceptual/near-duplicate deduplication")
    if getattr(args,"approved_audit",None):
        approval=json.loads(args.approved_audit.read_text())
        manifest["current_approved_scope"]=dict(snapshot=approval["snapshot_path"],sha256=approval["snapshot_sha256"],
            captured_at=approval["captured_at"],approved_training_videos=approval["approved_training_videos"],
            historical_sources=approval["legacy_training_videos"])
    dump(args.data_output / "manifest.json", manifest)
    dump(args.data_output / "public_accepted.json", accepted)
    dump(args.data_output / "public_excluded.json", excluded)
    dump(args.data_output / "summary.json", dict(stage="complete", **exposure,
        dataset_counts=dict(Counter(x["dataset"] for x in accepted)), excluded=len(excluded),
        additional_reviewed_negative_images=len(negative_rows),
        research_only=True, config=str(output)))
    if asset_view:
        dump(args.data_output / "augmentation_assets.json", asset_view)
    return output


def compare(args, device):
    import numpy as np
    import torch
    import yaml
    from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator, pool_native
    from scripts.anti_uav.run_native_pool_comparison import HOLDOUT, clean
    from scripts.anti_uav.run_approved_data_only_refresh import recipe_difference
    from ultralytics import YOLO
    deadline = time.monotonic() + 24*3600
    while True:
        state = args.baseline_run / "refresh_status.json"
        value = json.loads(state.read_text()) if state.exists() else {}
        if value.get("stage") == "complete":
            break
        if value.get("stage") == "failed" or time.monotonic() > deadline:
            raise RuntimeError("The native control did not complete; do not compare partial checkpoints")
        dump(args.run_dir / "research_status.json", dict(stage="waiting_for_control_completion"))
        time.sleep(120)
    old = json.loads((args.baseline_run / "protocol.json").read_text())
    new = json.loads((args.run_dir / "protocol.json").read_text())
    if recipe_difference(old, new):
        raise ValueError("Training recipe changed beyond its public-data supplement")
    output = args.run_dir / "comparison"
    output.mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    splits = {"Video00004": HOLDOUT, "Video00009": args.baseline_yaml}
    counts = {"Video00004": (2359, 448), "Video00009": (1421, 797)}
    result = {}
    arms = [("native69", args.baseline_run)]
    if getattr(args, "previous_public_run", None):
        arms.append(("public_drone_328assets", args.previous_public_run))
    arms.append(("all_aircraft_342assets" if getattr(args, "background_cache", None) else "native69_public", args.run_dir))
    for arm, folder in arms:
        for stage, relative in (("p3_initial", "foundation/p3/weights/best.pt"),
                                ("teacher", "teacher/p2p3/weights/best.pt"),
                                ("student", "p3_student_distilled_best_fp32.pt")):
            weights = folder / relative
            identity = sha256(weights)
            name = arm + "_" + stage
            entries, result[name] = [], dict(weights=str(weights), sha256=identity, splits={})
            for split, data in splits.items():
                dump(args.run_dir / "research_status.json", dict(stage="evaluating", model=name, split=split))
                model = YOLO(str(weights))
                metrics = model.val(data=str(data), validator=PairValidator, imgsz=[544, 960],
                    rect=False, device=str(device), batch=32, workers=4, conf=.001, iou=.45,
                    max_det=100, half=False, plots=False, verbose=False,
                    project=str(output / "val"), name=name+"_"+split)
                native = {k: v for k, v in clean(metrics.gray_selection).items() if k.startswith("native/")}
                if (native["native/c0.03/FRAMES"], native["native/c0.03/TP"]+native["native/c0.03/FN"]) != counts[split]:
                    raise ValueError("Holdout coverage changed")
                result[name]["splits"][split] = native
                entries.append(dict(metrics=native, arrays=metrics.native_pr_arrays, scales=metrics.native_scale_counts))
                np.savez_compressed(output / (name+"_"+split+".npz"), **metrics.native_pr_arrays)
                del model
                torch.cuda.empty_cache()
            pooled, arrays = pool_native(entries)
            result[name]["splits"]["pooled"] = pooled
            np.savez_compressed(output / (name+"_pooled.npz"), **arrays)
            external=args.data_output/'fixedwing_public_holdout.yaml'
            if external.exists():
                expected=json.loads((args.data_output/'fixedwing_public_holdout_protocol.json').read_text())
                model=YOLO(str(weights))
                metrics=model.val(data=str(external),validator=PairValidator,imgsz=[544,960],rect=False,
                    device=str(device),batch=32,workers=4,conf=.001,iou=.45,max_det=100,half=False,
                    plots=False,verbose=False,project=str(output/'val'),name=name+'_fixedwing_public')
                native={k:v for k,v in clean(metrics.gray_selection).items() if k.startswith('native/')}
                if (native['native/c0.03/FRAMES'],native['native/c0.03/TP']+native['native/c0.03/FN']) != (expected['frames'],expected['gt']):
                    raise ValueError('External fixed-wing holdout coverage changed')
                result[name]['splits']['external_fixedwing_positive_holdout']=native
                del model;torch.cuda.empty_cache()
            if sha256(weights) != identity:
                raise ValueError("A checkpoint changed while evaluating")
            dump(output / "results.json", result)
    dump(output / "protocol.json", dict(research_only=True, conf=.03, iou=.5, nms_iou=.45,
        precision="FP32", input_hw=[544, 960], tracker=False,
        pooled_contains_selection_validation=True, core_training_recipe_changes={},
        intentional_changes=["public_training_data", "augmentation_asset_catalog"] if getattr(args,"background_cache",None) else ["public_training_data"],
        splits={k: dict(yaml=str(v), yaml_sha256=sha256(v),
                        list_sha256=sha256(yaml.safe_load(v.read_text())["val"])) for k, v in splits.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-yaml", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--public-manifest", required=True, nargs="+", type=Path)
    parser.add_argument("--data-output", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpus", type=int, nargs="+", default=[6])
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--background-cache", type=Path,
                        help="Audited asset-only view; original image/background scope must be unchanged")
    parser.add_argument("--previous-public-run", type=Path,
                        help="Completed prior public-data experiment for a three-arm comparison")
    parser.add_argument("--approved-audit", type=Path,
                        help="Completed current-approval/cache audit; pins the approved snapshot and reused labels")
    parser.add_argument("--reviewed-negative-manifest",type=Path,
                        help="Append missing reviewed-absent frames to the rotating pool, not fixed anchors")
    args = parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    args.public_manifest = [x.resolve() for x in args.public_manifest]
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.reviewed_negative_manifest and not args.approved_audit:
        parser.error("Reviewed negative supplementation requires --approved-audit")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    lock = (args.run_dir / "research.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        dump(args.run_dir / "research_status.json", dict(stage="preparing", research_only=True, pid=os.getpid()))
        control = bind_control(args)
        if args.approved_audit:
            approval = json.loads(args.approved_audit.read_text())
            if approval.get("complete") is not True or approval.get("all_current_approved_training_sources_covered") is not True:
                raise ValueError("Current approved data audit is incomplete")
            import yaml
            native_source = yaml.safe_load(args.baseline_yaml.read_text())
            if (approval["source_train_sha256"] != sha256(native_source["train"]) or
                    approval["source_cache_sha256"] != sha256(Path(native_source["train"]).with_suffix(".cache")) or
                    approval["snapshot_sha256"] != sha256(approval["snapshot_path"])):
                raise ValueError("Approved snapshot or audited native data changed")
            dump(args.run_dir / "approved_audit.json", dict(path=str(args.approved_audit),sha256=sha256(args.approved_audit),audit=approval))
        data = prepare(args)
        from scripts.anti_uav.run_approved_data_only_refresh import choose_idle_gpu
        if sha256(control["foundation"]) != control["foundation_sha256"]:
            raise ValueError("Public initialization checkpoint changed")
        usage = ""
        while True:
            usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            device = choose_idle_gpu(usage, args.gpus, args.allow_shared_gpu)
            if device is not None:
                break
            dump(args.run_dir / "research_status.json", dict(stage="waiting_for_allowed_gpu", usage=usage))
            time.sleep(60)
        command = [sys.executable, "-u", str(ROOT / "scripts/anti_uav/run_p3_independent_curriculum_training.py"),
            "--data", str(data), "--foundation", control["foundation"], "--run-dir", str(args.run_dir),
            "--device", str(device), "--batch", "64", "--workers", "16", "--foundation-epochs", "30",
            "--teacher-epochs", "12", "--student-epochs", "15", "--localization-epochs", "0",
            "--replacement-probability", "0.5", "--expected-training-videos", "69"]
        dump(args.run_dir / "launch.json", dict(command=command, gpu=device, usage_before_launch=usage,
            allow_shared_gpu=args.allow_shared_gpu, research_only=True, commercial_clearance=False,
            control_unchanged=str(args.baseline_run), public_sources=[str(x) for x in args.public_manifest]))
        dump(args.run_dir / "research_status.json", dict(stage="training", device=device, pid=os.getpid()))
        with (args.run_dir / "training.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        compare(args, device)
        dump(args.run_dir / "research_status.json", dict(stage="complete", research_only=True))
    except BaseException as error:
        dump(args.run_dir / "research_status.json", dict(stage="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
