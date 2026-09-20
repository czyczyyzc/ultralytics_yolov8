#!/usr/bin/env python3
"""Publish visually screened cutouts without updating any training configuration."""
import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.matte_drone_catalog import card, digest, dump, gallery, pack_alpha


def main(root):
    initial = json.loads((root / "catalog.json").read_text())
    refinement = json.loads((root / "refined/catalog.json").read_text())
    if digest(root / "catalog.json") != refinement["source_catalog_sha256"]:
        raise ValueError("Parent catalog changed")
    revised = {r["id"]: r for r in refinement["records"]}
    exclusions = refinement["rules"]["exclude"]
    out = root / "screened"
    out.mkdir(exist_ok=False)
    for name in ("cutouts", "previews", "contact_sheets", "rgba"):
        (out / name).mkdir()
    rows = []
    for old in initial["records"]:
        aid = old["id"]
        if aid in exclusions:
            continue
        base = root / "refined" if aid in revised else root
        chosen = revised.get(aid, old)
        for key in ("raw", "rgba", "cutout"):
            if digest(base / chosen[key]) != chosen[key + "_sha256"]:
                raise ValueError(f"Input hash mismatch: {aid}/{key}")
        row = dict(chosen, raw="../" + old["raw"],
            rgba=os.path.relpath(base / chosen["rgba"], out),
            mask=os.path.relpath(base / chosen["mask"], out),
            cutout=f"cutouts/{aid}.png", preview=f"previews/{aid}.jpg",
            status="visually_screened_compositing_candidate")
        if aid == "295":
            # Explicit visual review: two components are drone and detached battery.
            # This selection must NOT be applied to other aircraft automatically.
            rgba = np.array(Image.open(base / chosen["rgba"]).convert("RGBA"))
            count, labels, stats, _ = cv2.connectedComponentsWithStats((rgba[..., 3] >= 4).astype(np.uint8), 8)
            if count != 3:
                raise ValueError("Reviewed drone/battery component structure changed")
            main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            alpha = rgba[..., 3].copy()
            alpha[labels != main] = 0
            rgba, bounds, quality = pack_alpha(rgba[..., :3], alpha)
            full = Image.fromarray(rgba)
            row.update(rgba=f"rgba/{aid}.png", bounds=bounds, quality=quality,
                method="roi_segmentation_then_visually_verified_detached_battery_removal")
            full.save(out / row["rgba"])
            full.crop(bounds).save(out / row["cutout"])
            Image.fromarray(alpha).save(out / "rgba/295_alpha.png")
            row.update(mask="rgba/295_alpha.png")
            for key in ("rgba", "mask"):
                row[key + "_sha256"] = digest(out / row[key])
            source_rgb = np.array(Image.open(root / old["raw"]).convert("RGB"))
            if not np.array_equal(rgba[..., :3], source_rgb):
                raise ValueError("Source pixels changed")
        else:
            os.link(base / chosen["cutout"], out / row["cutout"])
        row["cutout_sha256"] = digest(out / row["cutout"])
        if aid == "315":
            row["review_note"] = "Small drone in kit image is intentional; inspect at intended synthesis scale."
        source = Image.open(root / old["raw"]).convert("RGB")
        with Image.open(out / row["cutout"]) as cutout:
            card(source, cutout.convert("RGBA"), row).save(out / row["preview"], quality=95)
        rows.append(row)
    summary = dict(source_entries_processed=len(initial["records"]), screened_candidates=len(rows),
        unique_source_images=len({r["embedded_sha256"] for r in rows}), excluded=exclusions,
        kit_images_refined=sorted(revised, key=int), detached_battery_removed=["295"],
        review_scope=refinement["rules"]["visual_review_scope"], training_configuration_changed=False,
        note="Screened for compositing validation, not a pixel-perfect segmentation guarantee.")
    dump(out / "catalog.json", dict(records=rows, summary=summary,
        parent_catalog_sha256=digest(root / "catalog.json"),
        refinement_catalog_sha256=digest(root / "refined/catalog.json"),
        script_sha256=digest(Path(__file__)), training_approved=False))
    dump(out / "summary.json", summary)
    gallery(out, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    main(parser.parse_args().library)
