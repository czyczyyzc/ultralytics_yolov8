#!/usr/bin/env python3
"""Build an isolated, training-only batch of reviewed-frame replacement candidates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.gray_temporal_background import TemporalBackgrounds, legacy_annotations
from scripts.anti_uav.synthesize_gray_drone_replacements import (
    SkipSample, dump, fresh_directory, normalize_box, parse_boxes, preview_card,
    preview_overview, replace_target, safe_path, sha256, stable_seed,
)


def size_bin(box, width, height):
    short = min(box[2:]) * min(960 / width, 544 / height)
    return next((name for limit, name in ((8, "le8"), (16, "8to16"), (32, "16to32"))
                 if short <= limit), "gt32")


def select_frames(candidates, count, seed, min_gap=10):
    """Round-robin size strata; prefer separated frames without duplicating any."""
    pools = defaultdict(list)
    for item in candidates:
        pools[item["size_bin"]].append(item)
    rng = np.random.default_rng(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    selected, deferred = [], []
    while pools and len(selected) < count:
        for key in sorted(list(pools)):
            item = pools[key].pop()
            if not pools[key]:
                del pools[key]
            if all(abs(item["frame"] - old["frame"]) >= min_gap for old in selected):
                selected.append(item)
            else:
                deferred.append(item)
            if len(selected) == count:
                break
    for item in deferred:
        if len(selected) >= count:
            break
        selected.append(item)
    return sorted(selected, key=lambda item: item["frame"])


def training_records(dataset, include_legacy=False):
    meta = json.loads((dataset / "manifest.json").read_text())
    old = json.loads((Path(meta["approved_data"]).parent / "manifest.json").read_text())
    allowed = set(meta["train_video_hashes"])
    blocked = {meta["test_sha256"], meta["validation_sha256"]}
    if allowed & blocked:
        raise ValueError("Training/held-out video hash overlap")
    records = {}
    for row in old["approved_tasks"] + meta["appended_videos"]:
        digest = row["sha256"]
        if digest in blocked or digest not in allowed:
            continue
        if digest in records:
            raise ValueError("Duplicate approved training video")
        records[digest] = row
    schedule = dataset / "train_hardneg.txt"
    groups = defaultdict(dict)
    for text in schedule.read_text().splitlines():
        if text.strip():
            path = Path(text.strip())
            groups[str(path.parent)][path.stem] = path
    if include_legacy:
        legacy_path = Path(old["old_root"]) / "manifest.json"
        legacy = json.loads(legacy_path.read_text())
        for row in legacy["videos"]:
            digest = row["video_sha256"]
            if digest in blocked or digest not in allowed or digest in records:
                continue
            name = Path(row["video_name"]).stem
            dirs = [p for p in groups if Path(p).name == name]
            if len(dirs) != 1:
                raise ValueError(f"Ambiguous/missing legacy training image directory: {name}")
            parts = list(Path(dirs[0]).parts)
            if parts.count("images") != 1:
                raise ValueError("Cannot resolve legacy labels directory")
            parts[parts.index("images")] = "labels"
            records[digest] = dict(sha256=digest, video=row["video_path"],
                legacy_manifest=str(legacy_path), legacy_manifest_sha256=sha256(legacy_path),
                training_manifest=str(dataset / "manifest.json"),
                training_manifest_sha256=sha256(dataset / "manifest.json"),
                image_directory=dirs[0], label_directory=str(Path(*parts)),
                annotation_source="legacy_human_adjudicated_visible_json_explicit_training_split")
    return meta, sorted(records.values(), key=lambda row: row["video"]), groups


def candidate_frames(row, groups):
    if "legacy_manifest" in row:
        _, annotations = legacy_annotations(row)
        result, counts = [], Counter()
        images = groups.get(row["image_directory"], {})
        if not images:
            return [], {"eligible_positive_frames": 0}
        with Image.open(next(iter(images.values()))) as im:
            width, height = im.size
        for stem, path in sorted(images.items()):
            if not stem.isdecimal() or int(stem) not in annotations:
                raise ValueError("Training image does not match legacy frame indices")
            boxes = annotations[int(stem)]
            if len(boxes) != 1:
                counts["negative_or_multiple_targets"] += 1
                continue
            box = boxes[0]
            if min(box[2:]) < 3 or max(box[2:]) > 160:
                counts["outside_synthesis_size_range_original_retained"] += 1
                continue
            result.append(dict(frame=int(stem), image=str(path),
                label=str(Path(row["label_directory"]) / (stem + ".txt")),
                box=box, width=width, height=height, size_bin=size_bin(box, width, height)))
        counts["eligible_positive_frames"] = len(result)
        return result, dict(counts)
    task = Path(row["task"])
    approved = task / "manifest.json"
    if sha256(approved) != row["manifest_sha256"]:
        raise ValueError(f"Approved manifest changed: {approved}")
    meta = json.loads(approved.read_text())
    if meta["video"]["sha256"] != row["sha256"] or meta["frames"]["indexBase"] != 0:
        raise ValueError("Video identity or frame index mismatch")
    coco = task / "coco/annotations.json"
    expected = next(f["sha256"] for f in meta["files"] if f["path"] == "coco/annotations.json")
    if sha256(coco) != expected:
        raise ValueError(f"COCO changed: {coco}")
    data = json.loads(coco.read_text())
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation["bbox"])
    frames = meta["frames"]
    excluded = set(frames.get("excludedUncertainFrameIndices", []))
    excluded.update(frames.get("excludedUnreviewedFrameIndices", []))
    included = set(frames["includedFrameIndices"]) - excluded
    images = groups.get(row["image_directory"], {})
    by_frame = {int(stem): path for stem, path in images.items() if stem.isdecimal()}
    result, counts = [], Counter()
    for image in data["images"]:
        frame = int(image["frame_index"])
        if frame not in included or frame not in by_frame:
            continue
        boxes = annotations[image["id"]]
        if len(boxes) != 1:
            counts["negative_or_multiple_targets"] += 1
            continue
        box = boxes[0]
        if min(box[2:]) < 3 or max(box[2:]) > 160:
            counts["outside_synthesis_size_range_original_retained"] += 1
            continue
        path = by_frame[frame]
        result.append(dict(frame=frame, image=str(path),
            label=str(Path(row["label_directory"]) / (path.stem + ".txt")),
            box=box, width=image["width"], height=image["height"],
            size_bin=size_bin(box, image["width"], image["height"])))
    counts["eligible_positive_frames"] = len(result)
    counts["excluded_uncertain_or_unreviewed_frames"] = len(excluded)
    return result, dict(counts)


def load_assets(catalog, excluded):
    data = json.loads(catalog.read_text())
    result = []
    for item in sorted(data["records"], key=lambda item: int(item["id"])):
        if not item.get("cutout") or item["id"] in excluded:
            continue
        path = safe_path(catalog.parent, item["cutout"])
        if sha256(path) != item["cutout_sha256"]:
            raise ValueError(f"Asset hash mismatch: {path}")
        with Image.open(path) as im:
            result.append((item, np.asarray(im.convert("RGBA"))))
    if not result:
        raise ValueError("No eligible assets")
    return result


def prior_outputs(batches):
    sources, hashes, provenance = set(), set(), {}
    for root in batches:
        manifest = root / "manifest.json"
        if json.loads((root / "status.json").read_text())["stage"] != "complete":
            raise ValueError(f"Cannot exclude an incomplete batch: {root}")
        data = json.loads(manifest.read_text())
        for row in data["samples"]:
            sources.add((row["video_sha256"], int(row["frame"])))
            hashes.add(row["output_sha256"])
        provenance[str(manifest)] = sha256(manifest)
    return sources, hashes, provenance


def build(args):
    if args.per_video < 1 or args.variants < 1 or args.previews < 0:
        raise ValueError("Invalid batch size")
    cv2.setNumThreads(2)
    meta, records, groups = training_records(args.dataset, getattr(args, "include_legacy", False))
    excluded_sources, prior_hashes, prior_manifests = prior_outputs(getattr(args, "exclude_batch", []))
    assets = load_assets(args.catalog, set(args.exclude_assets.split(",")))
    if args.variants > len(assets):
        raise ValueError("Variants would repeat assets within a source frame")
    if getattr(args, "plan_only", False):
        plans = []
        for row in records:
            candidates, counts = candidate_frames(row, groups)
            remaining = [r for r in candidates if (row["sha256"], r["frame"]) not in excluded_sources]
            plans.append(dict(video=row["video"], sha256=row["sha256"],
                source_type="legacy" if "legacy_manifest" in row else "approved",
                **counts, previously_used_frames=len(candidates)-len(remaining),
                selected=min(args.per_video, len(remaining))))
        fresh_directory(args.output)
        dump(args.output / "plan.json", dict(videos=plans, registered_training_videos=len(records),
            expected_training_videos=len(meta["train_video_hashes"]),
            missing_training_hashes=sorted(set(meta["train_video_hashes"])-{r["sha256"] for r in records}),
            selected_frames=sum(r["selected"] for r in plans), variants=args.variants,
            maximum_outputs=sum(r["selected"] for r in plans)*args.variants,
            excluded_prior_manifests=prior_manifests, ground_truth_holdouts_excluded=True))
        return
    fresh_directory(args.output)
    for name in ("images", "labels", "masks", "previews", "provenance"):
        (args.output / name).mkdir()
    protected = [args.dataset / name for name in
                 ("manifest.json", "train_hardneg.txt", "val_monitor.txt", "train_hardneg_gray_monitor.yaml")]
    before = {str(path): sha256(path) for path in protected}
    for row in records:
        if "legacy_manifest" in row:
            before[row["legacy_manifest"]] = row["legacy_manifest_sha256"]
    before.update(prior_manifests)
    protected = [Path(p) for p in before]
    protocol = dict(schema="gray_replacement_batch.v1", synthetic=True,
        dataset=str(args.dataset), protected_input_sha256=before,
        catalog=str(args.catalog), catalog_sha256=sha256(args.catalog),
        asset_ids=[item["id"] for item, _ in assets], excluded_asset_ids=args.exclude_assets.split(","),
        blocked_video_sha256=[meta["test_sha256"], meta["validation_sha256"]],
        eligible_video_count=len(records), per_video=args.per_video, variants=args.variants,
        include_legacy=getattr(args, "include_legacy", False), excluded_prior_manifests=prior_manifests,
        excluded_previously_replaced_source_frames=len(excluded_sources),
        seed=args.seed, augmentation="registered_real_neighbor_plus_foreground_contour",
        training_modified=False, negatives_added=0, review_status="visual_review_candidates",
        limitations=["Not a continuous tracking sequence or a 3D pose simulation.",
            "Original target center and geometric long edge are preserved, not bbox area or aspect ratio.",
            "Approved or explicitly included legacy human-adjudicated sources; exact training-list membership required.",
            "Original-frame synthesis range: short edge >=3px, long edge <=160px; no originals removed.",
            "Pixel/geometry checks cannot guarantee visual realism or training improvement.",
            "Asset licensing and task-specific suitability require review before redistribution."])
    dump(args.output / "protocol.json", protocol)
    samples, skipped, previews, video_stats, selected_sources = [], [], [], [], []
    output_hashes, preview_videos, preview_assets = set(prior_hashes), Counter(), set()
    started = time.monotonic()
    dump(args.output / "status.json", dict(stage="running", pid=os.getpid(), accepted=0))
    try:
        with (args.output / "accepted.jsonl").open("w") as accepted_log:
            for index, row in enumerate(records):
                name = Path(row["video"]).stem
                candidates, counts = candidate_frames(row, groups)
                remaining = [r for r in candidates if (row["sha256"], r["frame"]) not in excluded_sources]
                counts["previously_replaced_source_frames_excluded"] = len(candidates)-len(remaining)
                candidates = remaining
                chosen = select_frames(candidates, args.per_video, stable_seed(args.seed, name))
                selected_sources.extend(dict(video=name, video_sha256=row["sha256"], **item) for item in chosen)
                registry = args.output / "provenance" / f"{row['sha256'][:12]}_registry.json"
                entry = row if "legacy_manifest" in row else dict(video=row["video"],
                    approved_manifest=str(Path(row["task"]) / "manifest.json"),
                    coco=str(Path(row["task"]) / "coco/annotations.json"))
                dump(registry, dict(videos=[entry]))
                accepted_before = len(samples)
                temporal = TemporalBackgrounds(registry, blocked=(),
                    blocked_hashes=protocol["blocked_video_sha256"]) if chosen else None
                try:
                    for source_index, source in enumerate(chosen):
                        image, lab = Path(source["image"]), Path(source["label"])
                        image_hash, label_hash = sha256(image), sha256(lab)
                        with Image.open(image) as im:
                            gray = np.asarray(im.convert("L"))
                        h, w = gray.shape
                        boxes = parse_boxes(lab.read_text(), w, h)
                        if (w, h) != (source["width"], source["height"]) or len(boxes) != 1 or not np.allclose(
                                boxes[0], source["box"], atol=0.25, rtol=0):
                            raise ValueError(f"Training label differs from approved annotation: {lab}")
                        offset = stable_seed(args.seed, name) + source_index * args.variants
                        source_success = 0
                        for variant in range(args.variants):
                            asset, rgba = assets[(offset + variant) % len(assets)]
                            stem = f"{row['sha256'][:12]}_{source['frame']:09d}_a{asset['id']}"
                            info = dict(video=name, frame=str(source["frame"]), variant=variant,
                                asset_id=asset["id"], asset_model=asset["model"],
                                source_image=str(image), source_label=str(lab), source_sha256=image_hash,
                                source_label_sha256=label_hash, video_sha256=row["sha256"],
                                size_bin=source["size_bin"], synthetic=True, status="replaced",
                                image=f"images/{stem}.png", label=f"labels/{stem}.txt", edit_mask=f"masks/{stem}.png")
                            try:
                                result, mask, box, metrics = replace_target(gray, boxes[0], rgba,
                                    stable_seed(args.seed, name, source["frame"], asset["id"]),
                                    temporal=temporal, video_id=name, frame_index=source["frame"])
                            except SkipSample as error:
                                skipped.append(dict(video=name, frame=source["frame"], asset_id=asset["id"], reason=str(error)))
                                # Background failures are independent of the selected cutout.
                                if source_success == 0 and not temporal.cache and not str(error).startswith((
                                        "insufficient_solid_asset", "asset_disappeared", "contrast_gain", "contrast_matching")):
                                    skipped[-1]["remaining_variants_not_attempted"] = args.variants - variant - 1
                                    break
                                continue
                            payload = normalize_box(box, w, h)
                            parsed = parse_boxes(payload, w, h)[0]
                            if not np.allclose(parsed, box, atol=1e-4, rtol=0):
                                raise AssertionError("Normalized bbox roundtrip failed")
                            out = args.output / info["image"]
                            Image.fromarray(result).save(out)
                            digest = sha256(out)
                            if digest in output_hashes:
                                out.unlink()
                                skipped.append(dict(video=name, frame=source["frame"], asset_id=asset["id"], reason="duplicate_output_pixels"))
                                continue
                            with Image.open(out) as saved:
                                decoded = np.asarray(saved)
                                if saved.mode != "L" or not np.array_equal(decoded, result):
                                    raise AssertionError("Lossless grayscale PNG verification failed")
                            if not np.array_equal(decoded[mask == 0], gray[mask == 0]):
                                raise AssertionError("Pixels outside edit mask changed")
                            (args.output / info["label"]).write_text(payload)
                            Image.fromarray(mask).save(args.output / info["edit_mask"])
                            output_hashes.add(digest)
                            info.update(metrics=metrics, output_sha256=digest,
                                output_label_sha256=sha256(args.output / info["label"]),
                                saved_png_outside_mask_changed_pixels=0)
                            if len(previews) < args.previews and preview_videos[name] < 2 and asset["id"] not in preview_assets:
                                pp = f"previews/{len(previews)+1:02d}_{stem}.png"
                                preview_card(gray, decoded, boxes[0], info, Image.fromarray(rgba), parsed).save(args.output / pp)
                                previews.append(pp)
                                preview_videos[name] += 1
                                preview_assets.add(asset["id"])
                                info["preview"] = pp
                            samples.append(info)
                            source_success += 1
                            accepted_log.write(json.dumps(info) + "\n")
                            accepted_log.flush()
                        temporal.cache.clear()
                        if sha256(image) != image_hash or sha256(lab) != label_hash:
                            raise AssertionError("Source image or label changed during synthesis")
                        progress = dict(stage="running", pid=os.getpid(), videos_finished=index,
                            videos_total=len(records), current_video=name, current_frame=source["frame"],
                            accepted=len(samples), source_frames_processed=len(selected_sources)-len(chosen)+source_index+1,
                            seconds=round(time.monotonic()-started, 2))
                        dump(args.output / "status.json", progress)
                        print(json.dumps(progress), flush=True)
                finally:
                    if temporal:
                        temporal.close()
                video_stats.append(dict(video=name, selected=len(chosen), accepted=len(samples)-accepted_before, **counts))
        if {str(path): sha256(path) for path in protected} != before:
            raise AssertionError("Protected training configuration changed")
        (args.output / "train_synthetic.txt").write_text("".join(str(args.output / s["image"]) + "\n" for s in samples))
        summary = dict(accepted=len(samples), source_frames_selected=len(selected_sources),
            source_frames_replaced=len({s["source_image"] for s in samples}),
            source_videos_replaced=len({s["video"] for s in samples}),
            asset_ids_used=sorted({s["asset_id"] for s in samples}, key=int),
            counts_by_source_size_bin=dict(Counter(s["size_bin"] for s in samples)),
            rejected_reasons=dict(Counter(s["reason"].split(":")[0] for s in skipped)),
            seconds=round(time.monotonic()-started, 2), protected_inputs_unchanged=True)
        dump(args.output / "manifest.json", dict(protocol=protocol, summary=summary,
            videos=video_stats, samples=samples, skipped=skipped, selected_sources=selected_sources, previews=previews))
        dump(args.output / "summary.json", summary)
        preview_overview(args.output, previews)
        (args.output / "README.md").write_text(
            "# Synthetic Grayscale Drone Replacement Candidates\n\n"
            f"Accepted: {len(samples)} lossless grayscale images with updated YOLO labels.\n\n"
            "images/ contains unannotated PNGs; labels/ contains normalized class-0 boxes; "
            "masks/ records the edit region. previews/ is for review only.\n\n"
            "train_synthetic.txt lists successful replacements only. Original negatives, failed "
            "replacements and held-out videos are not duplicated here. These are positive-only "
            "candidates, NOT a standalone balanced training set.\n\n"
            "No running training list or original image/label was modified. Review candidates "
            "before a separate controlled training experiment; maintain the chosen negative ratio. "
            "Do not use these as validation/test data or real camera captures.\n\n"
            "The current repair method supports original-frame targets with short edge >=3 px "
            "and long edge <=160 px. Larger original training targets are retained untouched. "
            "Shape, aspect ratio and updated bbox can change; center and geometric long edge are matched. "
            "No claim of perfect blending, new real viewpoints or measured accuracy improvement.\n")
        dump(args.output / "status.json", dict(stage="complete", pid=os.getpid(), **summary))
    except BaseException as error:
        dump(args.output / "status.json", dict(stage="failed", pid=os.getpid(), accepted=len(samples), error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-video", type=int, default=20)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--previews", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--exclude-assets", default="24,25", help="Catalog images containing remote controls")
    parser.add_argument("--include-legacy", action="store_true", help="Include old human-adjudicated videos explicitly allowed by this training split")
    parser.add_argument("--exclude-batch", type=Path, action="append", default=[], help="Completed batch whose successful source frames and output hashes must not repeat")
    parser.add_argument("--plan-only", action="store_true", help="Audit full source coverage and quotas without synthesis")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
