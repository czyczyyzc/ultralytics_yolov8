#!/usr/bin/env python3
"""Read-only integrity audit for successful outputs of build_gray_replacement_batch.py."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.synthesize_gray_drone_replacements import parse_boxes, safe_path, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    started = time.monotonic()
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    require(json.loads((root / "status.json").read_text())["stage"] == "complete", "Batch is not complete")
    require(manifest["protocol"] == protocol and manifest["summary"] == summary, "Metadata disagrees")
    require(protocol["schema"] == "gray_replacement_batch.v1", "Unsupported batch schema")
    require(not protocol["training_modified"], "Batch reports training modifications")
    require(protocol["augmentation"] == "registered_real_neighbor_plus_foreground_contour", "Unexpected background method")
    for path, digest in protocol["protected_input_sha256"].items():
        require(sha256(Path(path)) == digest, f"Protected training input changed: {path}")
    dataset = Path(protocol["dataset"])
    data = json.loads((dataset / "manifest.json").read_text())
    blocked = {data["test_sha256"], data["validation_sha256"]}
    require(set(protocol["blocked_video_sha256"]) == blocked, "Held-out definitions changed")
    training = set(data["train_video_hashes"])
    require(not (training & blocked), "Training/held-out video overlap")
    train_images = set((dataset / "train_hardneg.txt").read_text().splitlines())
    catalog = Path(protocol["catalog"])
    require(sha256(catalog) == protocol["catalog_sha256"], "Catalog hash changed")
    assets = {r["id"]: r for r in json.loads(catalog.read_text())["records"] if r.get("cutout")}
    for ident in protocol["asset_ids"]:
        asset = assets[ident]
        require(sha256(safe_path(catalog.parent, asset["cutout"])) == asset["cutout_sha256"],
                f"Cutout hash changed: {ident}")
    samples = manifest["samples"]
    require(len(samples) == summary["accepted"] and len(samples) > 0, "Invalid/empty accepted count")
    output_paths, seen_hashes = [], set()
    source_paths, video_hashes, used_assets = set(), set(), set()
    for index, row in enumerate(samples):
        require(row["synthetic"] and row["status"] == "replaced", "Non-replacement published")
        require(row["video_sha256"] in training and row["video_sha256"] not in blocked, "Nontraining video")
        require(row["source_image"] in train_images, "Source outside training list")
        require(row["asset_id"] in protocol["asset_ids"] and row["asset_id"] not in protocol["excluded_asset_ids"],
                "Unapproved asset ID")
        image_path = safe_path(root, row["image"])
        label_path = safe_path(root, row["label"])
        mask_path = safe_path(root, row["edit_mask"])
        for path, digest in ((Path(row["source_image"]), row["source_sha256"]),
                             (Path(row["source_label"]), row["source_label_sha256"]),
                             (image_path, row["output_sha256"]),
                             (label_path, row["output_label_sha256"])):
            require(sha256(path) == digest, f"File hash mismatch: {path}")
        require(row["output_sha256"] not in seen_hashes, "Duplicate output image")
        seen_hashes.add(row["output_sha256"])
        with Image.open(row["source_image"]) as im:
            original = np.asarray(im.convert("L"))
        with Image.open(image_path) as im:
            require(im.format == "PNG" and im.mode == "L", "Expected unannotated grayscale PNG")
            rendered = np.asarray(im)
        with Image.open(mask_path) as im:
            require(im.mode == "L", "Expected grayscale edit mask")
            mask = np.asarray(im)
        require(rendered.shape == original.shape == mask.shape, "Image/mask shape mismatch")
        require(np.all((mask == 0) | (mask == 255)), "Nonbinary edit mask")
        require(np.array_equal(rendered[mask == 0], original[mask == 0]), "Pixels changed outside mask")
        require(np.any(rendered != original), "No replacement pixels")
        height, width = original.shape
        new_boxes = parse_boxes(label_path.read_text(), width, height)
        old_boxes = parse_boxes(Path(row["source_label"]).read_text(), width, height)
        metrics = row["metrics"]
        require(len(new_boxes) == len(old_boxes) == 1, "Expected one target per sample")
        require(np.allclose(new_boxes[0], metrics["new_box_xywh"], atol=1e-4, rtol=0), "New bbox differs from record")
        require(np.allclose(old_boxes[0], metrics["original_box_xywh"], atol=.25, rtol=0), "Original bbox differs")
        require(metrics["background_method"] == protocol["augmentation"], "Unexpected sample background method")
        require(metrics["donor_video_sha256"] == row["video_sha256"], "Donor is not the same training video")
        output_paths.append(str(image_path))
        source_paths.add(row["source_image"])
        video_hashes.add(row["video_sha256"])
        used_assets.add(row["asset_id"])
        if (index + 1) % 100 == 0:
            print(f"Verified {index + 1}/{len(samples)}", file=sys.stderr, flush=True)
    scheduled = (root / "train_synthetic.txt").read_text().splitlines()
    require(scheduled == output_paths, "Synthetic list differs from accepted outputs or batch was relocated")
    require(len(source_paths) == summary["source_frames_replaced"], "Source count mismatch")
    require(len(video_hashes) == summary["source_videos_replaced"], "Video count mismatch")
    require(used_assets == set(summary["asset_ids_used"]), "Asset count mismatch")
    for folder, field in (("images", "image"), ("labels", "label"), ("masks", "edit_mask")):
        require({p.resolve() for p in (root / folder).iterdir() if p.is_file()} ==
                {safe_path(root, r[field]) for r in samples}, f"Extra or missing files: {folder}")
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), batch=str(root),
                verified_images=len(samples), source_frames=len(source_paths),
                source_videos=len(video_hashes), assets=len(used_assets),
                file_hashes_valid=True, label_boxes_match_recorded_geometry=True,
                pixels_outside_edit_mask_unchanged=True, heldout_video_overlap=False,
                protected_training_inputs_unchanged=True, seconds=round(time.monotonic()-started, 2),
                limitation="Integrity audit only; not visual realism, asset licensing, independent silhouette accuracy or training-benefit validation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional new JSON file; never overwritten")
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    result = verify(args.batch.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        with args.report.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
