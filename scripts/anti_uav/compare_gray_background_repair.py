#!/usr/bin/env python3
"""Matched old/new synthesis crops and edit-mask diagnostics, without training edits."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.synthesize_gray_drone_replacements import (
    draw_corner_box, dump, fresh_directory, label, parse_boxes, safe_path, sha256,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-manifest", required=True, type=Path)
    p.add_argument("--new-manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    old = json.loads(args.old_manifest.read_text())
    new = json.loads(args.new_manifest.read_text())
    old_by_key = {s["image"]:s for s in old["samples"]}
    fresh_directory(args.output)
    records = []
    for info in new["samples"]:
        if not info.get("preview") or info["status"] != "replaced":
            continue
        previous = old_by_key[info["image"]]
        if previous["source_sha256"] != info["source_sha256"] or previous["asset_id"] != info["asset_id"]:
            raise ValueError("Comparison must use identical original frame and asset")
        source_path = Path(info["source_image"])
        if sha256(source_path) != info["source_sha256"]:
            raise ValueError("Original frame hash mismatch")
        original = Image.open(source_path).convert("L")
        x, y, w, h = info["metrics"]["original_box_xywh"]
        span = max(64, int(math.ceil(max(w, h)*2.2)))
        left = max(0, min(original.width-span, int(x+w/2-span/2)))
        top = max(0, min(original.height-span, int(y+h/2-span/2)))
        rect = (left, top, left+span, top+span)
        view = Image.new("RGB", (1320, 574), (25, 29, 35))
        mask_view = Image.new("RGB", (880, 504), (25, 29, 35))
        label(view, (14, 10), f"Frame {info['frame']} / Asset {info['asset_id']} / Same crop, same scale / SYNTHETIC", 20)
        label(view, (14, 39), "Background comparison: original | previous rectangle repair | registered real neighboring frame", 17)
        source = original.crop(rect).convert("RGB").resize((420, 420), Image.Resampling.NEAREST)
        draw_corner_box(source, [x,y,w,h], (420/span,)*2, (left, top), (85,220,135))
        view.paste(source, (10, 104))
        label(view, (14, 77), "ORIGINAL / original bbox", 20, (85,220,135))
        for index, (item, root, title, color) in enumerate((
                (previous,args.old_manifest.parent,"OLD / rectangle repair",(255,190,90)),
                (info,args.new_manifest.parent,"NEW / real neighbor",(40,210,255)))):
            image_path = safe_path(root, item["image"])
            if sha256(image_path) != item["output_sha256"]:
                raise ValueError("Synthesis image hash mismatch")
            image = Image.open(image_path).convert("RGB")
            bbox = parse_boxes(safe_path(root,item["label"]).read_text(), image.width, image.height)[0]
            crop = image.crop(rect).resize((420,420), Image.Resampling.NEAREST)
            draw_corner_box(crop, bbox, (420/span,)*2, (left,top), color)
            view.paste(crop, ((index+1)*440+10,104))
            label(view, ((index+1)*440+14,77), title,20,color)
            mask = Image.open(safe_path(root,item["edit_mask"])).crop(rect)
            mask_view.paste(mask.convert("RGB").resize((420,420), Image.Resampling.NEAREST),(index*440+10,64))
            label(mask_view,(index*440+14,20),title+" / edit mask",18,color)
        label(view,(14,541),f"Display zoom: {420/span:.2f}x / no cross / 1px label corners",16)
        label(view,(650,541),f"New donor offset: {info['metrics']['donor_offset']:+d} frames / no independent noise fill",16)
        stem = Path(info["image"]).stem
        file = f"{len(records)+1:02d}_{stem}_old_vs_new.png"
        view.save(args.output/file)
        mask_file = f"{len(records)+1:02d}_{stem}_edit_masks.png"
        mask_view.save(args.output/mask_file)
        records.append(dict(comparison=file, edit_masks=mask_file, source_sha256=info["source_sha256"],
                            asset_id=info["asset_id"], donor_frame=info["metrics"]["donor_frame"]))
    dump(args.output/"comparison_manifest.json", dict(old_manifest_sha256=sha256(args.old_manifest),
         new_manifest_sha256=sha256(args.new_manifest), training_files_modified=False, comparisons=records))
    print(json.dumps({"comparisons":len(records),"output":str(args.output)}))


if __name__ == "__main__":
    main()
