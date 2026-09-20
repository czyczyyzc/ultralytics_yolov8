#!/usr/bin/env python3
"""Publish visually screened cutouts without updating any training configuration."""
import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.matte_drone_catalog import card, digest, dump, gallery


def main(root):
    initial = json.loads((root / "catalog.json").read_text())
    refinement = json.loads((root / "refined/catalog.json").read_text())
    if digest(root / "catalog.json") != refinement["source_catalog_sha256"]:
        raise ValueError("Parent catalog changed")
    revised = {r["id"]: r for r in refinement["records"]}
    exclusions = dict(refinement["rules"]["exclude"])
    exclusions["295"] = "Close-up review: translucent rotor overlaps charger; accessory face remains in alpha. Preserve RGB; do not invent hidden foreground."
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
        kit_images_refined=sorted(revised, key=int),
        kit_refinements_accepted=sorted(set(revised) - set(exclusions), key=int),
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
