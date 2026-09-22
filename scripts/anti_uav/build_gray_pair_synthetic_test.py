#!/usr/bin/env python3
"""Export an evaluation-only, deterministic appearance intervention on the gray pair.

This deliberately does not create a training cache or bypass any training guard.
Unsafe positives retain their original pixels and labels, with explicit provenance.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.gray_temporal_background import TemporalBackgrounds
from scripts.anti_uav.online_gray_replacement import compose_cached
from scripts.anti_uav.prepare_online_gray_replacement import atomic_json
from scripts.anti_uav.synthesize_gray_drone_replacements import (
    SkipSample, normalize_box, parse_boxes, prepare_target, preview_card,
    preview_overview, safe_path, sha256, stable_seed,
)


def label_for(image):
    parts = list(Path(image).parts)
    if parts.count("images") != 1:
        raise ValueError(f"Ambiguous label path: {image}")
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def assign_assets(rows, ids, seed):
    """Plan without inspecting detector predictions or replacement success."""
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Asset catalog must be nonempty with unique identities")
    rng = np.random.default_rng(seed)
    positive = [r for r in rows if r["box"]]
    order = rng.permutation(len(positive))
    choices = []
    while len(choices) < len(positive):
        choices.extend(ids[int(i)] for i in rng.permutation(len(ids)))
    for j, i in enumerate(order):
        positive[int(i)]["asset_id"] = choices[j]
    return rows


def checked_json(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError(f"Protected metadata changed: {path}")
    return json.loads(path.read_text())


def test_annotations(cache):
    meta = json.loads((cache/"manifest.json").read_text())
    parent = json.loads((Path(meta["approved_data"]).parent/"manifest.json").read_text())
    legacy_path = Path(parent["old_root"])/"manifest.json"
    legacy = json.loads(legacy_path.read_text())
    contract = legacy["annotation_contract"]
    if contract["bbox_format"] != "xywh" or contract["frame_index_base"] != 0:
        raise ValueError("Unsupported original test annotation contract")
    records = [v for v in legacy["videos"] if v["video_sha256"] == meta["test_sha256"]]
    if len(records) != 1:
        raise ValueError("Ambiguous test video identity")
    record = records[0]
    annotation = safe_path(legacy_path.parent, record["annotation_path"])
    data = checked_json(annotation, record["annotation_sha256"])
    if len(data["exist"]) != record["frame_count"] or len(data["gt_rect"]) != record["frame_count"]:
        raise ValueError("Legacy annotation length changed")
    boxes = {}
    for frame, (present, box) in enumerate(zip(data["exist"], data["gt_rect"])):
        if present not in (0, 1) or (present and (len(box) != 4 or not np.isfinite(box).all() or min(box[2:]) <= 0)):
            raise ValueError("Invalid or uncertain legacy label")
        boxes[frame] = [box] if present else []
    videos = {"Video00004": dict(video=record["video_path"], video_sha256=record["video_sha256"],
        boxes=boxes, annotation=str(annotation), annotation_sha256=record["annotation_sha256"],
        manifest=str(legacy_path), manifest_sha256=sha256(legacy_path))}
    val = meta["validation_video"]
    manifest = Path(val["task"])/"manifest.json"
    approved = checked_json(manifest, val["manifest_sha256"])
    if approved["frames"]["indexBase"] != 0 or approved["video"]["sha256"] != meta["validation_sha256"]:
        raise ValueError("Validation source identity changed")
    annotation = Path(val["task"])/"coco/annotations.json"
    digest = next(f["sha256"] for f in approved["files"] if f["path"] == "coco/annotations.json")
    data = checked_json(annotation, digest)
    frames = approved["frames"]
    eligible = set(frames["includedFrameIndices"])
    eligible -= set(frames.get("excludedUncertainFrameIndices", []))
    eligible -= set(frames.get("excludedUnreviewedFrameIndices", []))
    images = {im["id"]: im for im in data["images"]}
    if len(images) != len(data["images"]) or not eligible <= {i["frame_index"] for i in images.values()}:
        raise ValueError("Missing or duplicate validation annotations")
    boxes = {f: [] for f in eligible}
    for ann in data["annotations"]:
        frame = images[ann["image_id"]]["frame_index"]
        if frame in boxes:
            boxes[frame].append(ann["bbox"])
    videos["Video00009"] = dict(video=val["video"], video_sha256=val["sha256"], boxes=boxes,
        annotation=str(annotation), annotation_sha256=digest,
        manifest=str(manifest), manifest_sha256=val["manifest_sha256"])
    if {v["video_sha256"] for v in videos.values()} & set(meta["train_video_hashes"]):
        raise ValueError("Evaluation source is in training videos")
    for v in videos.values():
        if sha256(Path(v["video"])) != v["video_sha256"]:
            raise ValueError("Raw video checksum mismatch")
    return videos


def plan_dataset(reference, cache, seed, smoke):
    protocol = json.loads((reference/"protocol.json").read_text())
    videos = test_annotations(cache)
    index = json.loads((cache/"index.json").read_text())
    catalog = Path(index["catalog"])
    assets = checked_json(catalog, index["catalog_sha256"])
    assets = {r["id"]: r for r in assets["records"] if r["id"] in index["asset_ids"]}
    if len(assets) != len(index["asset_ids"]):
        raise ValueError("Missing asset identities")
    for a in assets.values():
        a["path"] = str(safe_path(catalog.parent, a["cutout"]))
        if sha256(Path(a["path"])) != a["cutout_sha256"]:
            raise ValueError("Cutout pixels changed")
    protected = {str(reference/"protocol.json"): sha256(reference/"protocol.json"),
                 str(catalog): index["catalog_sha256"]}
    for name in ("manifest.json", "index.json", "train_online_gray_monitor.yaml"):
        protected[str(cache/name)] = sha256(cache/name)
    train = yaml.safe_load((cache/"train_online_gray_monitor.yaml").read_text())
    for key in ("train", "val"):
        protected[train[key]] = sha256(Path(train[key]))
    rows = []
    for video, split in (("Video00004", "Video00004_test"), ("Video00009", "Video00009_selection_subset")):
        info = protocol["datasets"][split]
        listing = Path(info["list"])
        if sha256(listing) != info["list_sha256"]:
            raise ValueError("Reference evaluation frames changed")
        protected[str(listing)] = info["list_sha256"]
        selected = [Path(s) for s in listing.read_text().splitlines() if s and "/zoom_val/" not in s]
        if len(selected) != info["native_frames"] or len(set(selected)) != len(selected):
            raise ValueError("Reference frame coverage changed")
        group = []
        for image in selected:
            frame = int(image.stem)
            labels = videos[video]["boxes"].get(frame)
            if labels is None or len(labels) > 1:
                raise ValueError("Unreviewed/multiple-target frame not supported")
            label = label_for(image)
            with Image.open(image) as im:
                w, h = im.size
            actual = parse_boxes(label.read_text(), w, h)
            if len(labels) != len(actual) or (labels and not np.allclose(labels, actual, atol=.25, rtol=0)):
                raise ValueError(f"Image label differs from source annotations: {image}")
            group.append(dict(video=video, frame=frame, source=str(image), label=str(label),
                source_sha256=sha256(image), label_sha256=sha256(label), width=w, height=h,
                box=actual[0] if actual else [], asset_id=None))
        if smoke:
            rng = np.random.default_rng(seed)
            positive = [r for r in group if r["box"]]
            group = [positive[int(i)] for i in rng.permutation(len(positive))[:smoke]] + [r for r in group if not r["box"]][:2]
        rows.extend(group)
    for v in videos.values():
        protected[v["manifest"]] = v["manifest_sha256"]
        protected[v["annotation"]] = v["annotation_sha256"]
    return dict(schema="evaluation_only_gray_replacement.v1", training_allowed=False,
        reference=str(reference), reference_protocol=protocol, seed=seed, smoke_per_video=smoke,
        assets_seen_during_training=True, detector_evaluation_only=True, catalog=str(catalog),
        assets=assets, videos=videos, protected=protected,
        rows=assign_assets(rows, list(assets), seed))


class EvaluationTemporalBackgrounds(TemporalBackgrounds):
    def __init__(self, plan):
        if plan.get("schema") != "evaluation_only_gray_replacement.v1" or plan.get("training_allowed") is not False:
            raise ValueError("Explicit evaluation-only provenance required")
        self.records, self.cache = {}, {}
        self.offsets = (10, -10, 20, -20, 40, -40, 80, -80)
        for name, v in plan["videos"].items():
            capture = cv2.VideoCapture(v["video"])
            if not capture.isOpened():
                self.close()
                raise ValueError("Cannot open evaluation video")
            boxes = {int(k): b for k, b in v["boxes"].items()}
            self.records[name] = dict(capture=capture, boxes=boxes, eligible=set(boxes),
                video_sha256=v["video_sha256"], coco_sha256=v["annotation_sha256"],
                approved_manifest_sha256=v["manifest_sha256"],
                annotation_provenance="evaluation_only_original_reviewed_annotations")


@lru_cache(maxsize=8)
def asset_pixels(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))


def init_worker(plan, root):
    global PLAN, OUTPUT, TEMPORAL
    cv2.setNumThreads(1)
    PLAN, OUTPUT = plan, Path(root)
    TEMPORAL = EvaluationTemporalBackgrounds(plan)


def export_frame(row):
    row = dict(row)
    source, label = Path(row["source"]), Path(row["label"])
    if sha256(source) != row["source_sha256"] or sha256(label) != row["label_sha256"]:
        raise ValueError("Source changed after planning")
    stem = f"{row['frame']:09d}"
    original = OUTPUT/f"original/images/{row['video']}/{stem}{source.suffix}"
    original_label = OUTPUT/f"original/labels/{row['video']}/{stem}.txt"
    shutil.copy2(source, original)
    shutil.copy2(label, original_label)
    row.update(original=str(original.relative_to(OUTPUT)), original_label=str(original_label.relative_to(OUTPUT)),
        state="negative_unchanged" if not row["box"] else "original_retained", reason=None)
    if row["box"]:
        bgr = cv2.imread(str(source))
        if bgr is None or bgr.shape[:2] != (row["height"], row["width"]):
            raise ValueError("Source decode/shape failure")
        try:
            prepared = prepare_target(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), row["box"],
                stable_seed(PLAN["seed"], row["video"], row["frame"]), temporal=TEMPORAL,
                video_id=row["video"], frame_index=row["frame"])
            result, normalized, metrics = compose_cached(bgr, prepared, asset_pixels(PLAN["assets"][row["asset_id"]]["path"]))
            output = OUTPUT/f"synthetic/images/{row['video']}/{stem}.png"
            if not cv2.imwrite(str(output), result, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise IOError("PNG write failed")
            if not np.array_equal(cv2.imread(str(output)), result):
                raise AssertionError("Lossless PNG roundtrip failed")
            new_label = normalize_box(metrics["new_box_xywh"], row["width"], row["height"])
            if not np.allclose(normalized, [float(x) for x in new_label.split()[1:]], atol=1e-7):
                raise AssertionError("Updated bbox mismatch")
            row.update(state="replaced", metrics=metrics)
        except SkipSample as error:
            row["reason"] = str(error)
        finally:
            TEMPORAL.cache.clear()
    if row["state"] != "replaced":
        output = OUTPUT/f"synthetic/images/{row['video']}/{stem}{source.suffix}"
        output.symlink_to(Path("../../../original/images")/row["video"]/original.name)
        new_label = label.read_text()
    out_label = OUTPUT/f"synthetic/labels/{row['video']}/{stem}.txt"
    out_label.write_text(new_label)
    row.update(replacement=str(output.relative_to(OUTPUT)), output_label=str(out_label.relative_to(OUTPUT)),
        output_sha256=sha256(output), output_label_sha256=sha256(out_label))
    if row["state"] != "replaced" and row["output_sha256"] != row["source_sha256"]:
        raise AssertionError("Unchanged sample was modified")
    if sha256(source) != row["source_sha256"] or sha256(label) != row["label_sha256"]:
        raise ValueError("Source mutated during export")
    return row


def finalize(plan, rows, root):
    rows.sort(key=lambda r: (r["video"], r["frame"]))
    for path, expected in plan["protected"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Protected input changed: {path}")
    for kind, key in (("original", "original"), ("synthetic", "replacement")):
        for video in ("combined", "Video00004", "Video00009"):
            selected = [r for r in rows if video == "combined" or r["video"] == video]
            listing = root/f"{kind}_{video}.txt"
            listing.write_text("\n".join(str(root/r[key]) for r in selected)+"\n")
            config = dict(path=str(root), train=None, val=str(listing), test=str(listing), names={0: "drone"})
            (root/f"{kind}_{video}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        matched = [r for r in rows if r["state"] == "replaced"]
        (root/f"{kind}_replaced_positive_pairs.txt").write_text("\n".join(str(root/r[key]) for r in matched)+"\n")
    previews = []
    rng = np.random.default_rng(plan["seed"])
    for video in plan["videos"]:
        candidates = [r for r in rows if r["video"] == video and r["state"] == "replaced"]
        for i in rng.permutation(len(candidates))[:12]:
            row = candidates[int(i)]
            a = plan["assets"][row["asset_id"]]
            info = dict(row, asset_model=a["model"])
            card = preview_card(cv2.imread(str(root/row["original"]), 0),
                cv2.imread(str(root/row["replacement"]), 0), row["box"], info, Image.open(a["path"]))
            name = f"previews/{video}_{row['frame']:09d}_a{row['asset_id']}.png"
            card.save(root/name)
            previews.append(name)
    preview_overview(root, previews[:8])
    counts = Counter(r["state"] for r in rows)
    summary = dict(stage="complete", total_frames=len(rows), positive_frames=sum(bool(r["box"]) for r in rows),
        states=dict(counts), by_video={v: dict(Counter(r["state"] for r in rows if r["video"] == v)) for v in plan["videos"]},
        available_assets=len(plan["assets"]), assigned_assets=len({r["asset_id"] for r in rows if r["box"]}),
        successful_assets=len({r["asset_id"] for r in rows if r["state"] == "replaced"}),
        rejection_reasons=dict(Counter(r["reason"].split(":")[0] for r in rows if r["reason"])),
        seed=plan["seed"], bbox_updated=True, unsafe_positives_retained=True, training_modified=False,
        manual_visual_review_pending=True, assets_seen_during_training=True, smoke_per_video=plan["smoke_per_video"],
        previews=previews, manifest_sha256=None)
    atomic_json(root/"manifest.json", dict(schema=plan["schema"], training_allowed=False, rows=rows))
    summary["manifest_sha256"] = sha256(root/"manifest.json")
    atomic_json(root/"summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--smoke-per-video", type=int, default=0)
    a = p.parse_args()
    if a.workers < 1 or a.smoke_per_video < 0:
        raise ValueError("Invalid worker/smoke count")
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root/"status.json", dict(stage="planning", pid=os.getpid()))
    try:
        plan = plan_dataset(a.reference.resolve(), a.cache.resolve(), a.seed, a.smoke_per_video)
        atomic_json(root/"plan.json", plan)
        for kind in ("original", "synthetic"):
            for data in ("images", "labels"):
                for video in plan["videos"]:
                    (root/kind/data/video).mkdir(parents=True)
        (root/"previews").mkdir()
        rows, start = [], time.monotonic()
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=init_worker, initargs=(plan, str(root))) as pool:
            futures = [pool.submit(export_frame, row) for row in plan["rows"]]
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows) % 25 == 0 or len(rows) == len(futures):
                    status = dict(stage="exporting", processed=len(rows), total=len(futures),
                        states=dict(Counter(r["state"] for r in rows)), seconds=round(time.monotonic()-start, 2))
                    atomic_json(root/"status.json", status)
                    print(json.dumps(status), flush=True)
        summary = finalize(plan, rows, root)
        atomic_json(root/"status.json", dict(stage="complete", summary=str(root/"summary.json")))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        atomic_json(root/"status.json", dict(stage="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
