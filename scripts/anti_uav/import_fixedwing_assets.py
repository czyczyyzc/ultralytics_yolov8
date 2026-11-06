#!/usr/bin/env python3
"""Versioned alpha-only folder import and reviewed union with an existing catalog."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.matte_drone_catalog import card, digest, dump, gallery, pack_alpha

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def inventory(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in SUFFIXES:
            continue
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original)
            alpha = image.getchannel("A").getextrema() if "A" in image.getbands() else None
            rows.append(dict(input_id=str(len(rows)+1), model=path.parent.name,
                relative=str(path.relative_to(root)), sha256=digest(path), size=list(image.size),
                native_transparency=bool(alpha and alpha[0] < 255 and alpha[1] > 0)))
    return rows


def alpha_eligible(path):
    with Image.open(path) as image:
        if image.mode != "RGBA":
            raise ValueError(f"RGBA required: {path}")
        alpha = np.asarray(image)[:, :, 3]
        if not np.any(alpha > 0) or not np.any(alpha == 0):
            raise ValueError(f"Empty or opaque asset: {path}")


def segment(args):
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    if args.output.exists():
        raise FileExistsError(args.output)
    rules = json.loads(args.rules.read_text())
    base = json.loads(args.base_catalog.read_text())
    offset = max(int(r["id"]) for r in base["records"])
    source = inventory(args.input)
    if len(source) != rules["expected_inputs"]:
        raise ValueError("Input count changed")
    selected = set(rules["segment_input_ids"])
    if not selected <= {r["input_id"] for r in source}:
        raise ValueError("Unknown selected input")
    model_files = {p.name: digest(p) for p in args.model.iterdir() if p.is_file()}
    reference = json.loads(args.model_provenance.read_text())
    if model_files != reference["model_files"]:
        raise ValueError("Pinned segmentation model changed")
    args.output.mkdir(parents=True)
    for name in ("raw", "rgba", "masks", "cutouts", "previews", "contact_sheets"):
        (args.output/name).mkdir()
    dump(args.output/"input_inventory.json", source)
    dump(args.output/"run_config.json", dict(model_files=model_files,
        model_revision=reference["model_revision"], rules=rules, rules_sha256=digest(args.rules),
        base_catalog=str(args.base_catalog), base_catalog_sha256=digest(args.base_catalog),
        device=args.device, input_root=str(args.input), source_rgb_regenerated=False,
        orientation="EXIF transpose to displayed orientation before lossless RGB preservation"))
    torch.set_num_threads(4)
    torch.manual_seed(20261106)
    model = AutoModelForImageSegmentation.from_pretrained(str(args.model), trust_remote_code=True,
        local_files_only=True).eval().to(args.device)
    transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    rows = []
    for item in source:
        if item["input_id"] not in selected:
            continue
        path = args.input/item["relative"]
        if digest(path) != item["sha256"]:
            raise ValueError("Source changed")
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        roi = rules.get("roi", {}).get(item["input_id"], [0, 0, 1, 1])
        if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
            raise ValueError("Invalid ROI")
        x0,y0,x1,y1 = [round(v*s) for v,s in zip(roi, [image.width,image.height]*2)]
        crop = image.crop((x0,y0,x1,y1))
        with torch.inference_mode():
            pred = model(transform(crop).unsqueeze(0).to(args.device))[-1].sigmoid()
        small = Image.fromarray((pred[0,0].cpu().numpy()*255).clip(0,255).astype(np.uint8))
        alpha = np.zeros((image.height,image.width),np.uint8)
        alpha[y0:y1,x0:x1] = np.asarray(small.resize(crop.size, Image.Resampling.BILINEAR))
        rgba, bounds, quality = pack_alpha(np.asarray(image),alpha)
        aid = str(offset+int(item["input_id"]))
        row = dict(item, id=aid, category="fixedwing_rc", company=None,
            raw=f"raw/{aid}{path.suffix.lower()}", rgba=f"rgba/{aid}.png",
            cutout=f"cutouts/{aid}.png", mask=f"masks/{aid}.png", preview=f"previews/{aid}.jpg",
            bounds=bounds, quality=quality, roi=roi, source_relative=item["relative"],
            source_sha256=item["sha256"], status="candidate_requires_visual_review")
        shutil.copy2(path,args.output/row["raw"])
        full = Image.fromarray(rgba)
        full.save(args.output/row["rgba"])
        full.crop(bounds).save(args.output/row["cutout"]) if bounds else full.save(args.output/row["cutout"])
        Image.fromarray(rgba[:,:,3]).save(args.output/row["mask"])
        with Image.open(args.output/row["rgba"]) as check:
            if not np.array_equal(np.asarray(check)[:,:,:3],np.asarray(image)):
                raise ValueError("Source RGB changed")
        for field in ("raw","rgba","cutout","mask"):
            row[field+"_sha256"] = digest(args.output/row[field])
        card(image,full,row).save(args.output/row["preview"],quality=95)
        rows.append(row)
        dump(args.output/"catalog.json",dict(records=rows,complete=False))
        print(json.dumps(dict(input_id=item["input_id"],id=aid,quality=quality)),flush=True)
    dump(args.output/"catalog.json",dict(records=rows,complete=True,training_configuration_changed=False))
    gallery(args.output,rows)


def publish(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    original_bytes = args.base_catalog.read_bytes()
    old = json.loads(original_bytes)
    candidates = json.loads((args.candidates/"catalog.json").read_text())
    config = json.loads((args.candidates/"run_config.json").read_text())
    if not candidates.get("complete") or digest(args.base_catalog) != config["base_catalog_sha256"]:
        raise ValueError("Incomplete candidates or changed base catalog")
    review = json.loads(args.review.read_text())
    accepted = set(review["accepted_input_ids"])
    available = {r["input_id"]:r for r in candidates["records"]}
    if not accepted <= set(available):
        raise ValueError("Unsegmented accepted ID")
    args.output.mkdir(parents=True)
    (args.output/"cutouts").mkdir()
    rows, seen_ids, old_hashes = [], set(), set()
    source_catalogs, old_source_hashes = {}, set()
    for row in old["records"]:
        source = args.base_catalog.parent/row["cutout"]
        if digest(source) != row["cutout_sha256"]:
            raise ValueError("Existing asset changed")
        alpha_eligible(source)
        if row["id"] in seen_ids:
            raise ValueError("Duplicate old ID")
        seen_ids.add(row["id"]);old_hashes.add(row["cutout_sha256"])
        if row.get("source_catalog"):
            catalog = row["source_catalog"]
            if catalog not in source_catalogs:
                source_catalogs[catalog] = {r["id"]:r for r in json.loads(Path(catalog).read_text())["records"]}
            original = source_catalogs[catalog][row["id"]]
            if original.get("embedded_sha256"):
                old_source_hashes.add(original["embedded_sha256"])
        rel = f"cutouts/{row['id']}.png"
        shutil.copy2(source,args.output/rel)
        rows.append(dict(row,cutout=rel))
    added, duplicate = [], []
    hashes = set(old_hashes)
    for key in sorted(accepted,key=int):
        row = available[key];source=args.candidates/row["cutout"]
        if digest(source) != row["cutout_sha256"]:
            raise ValueError("Candidate changed")
        alpha_eligible(source)
        if row["cutout_sha256"] in hashes or row.get("source_sha256") in old_source_hashes:
            duplicate.append(key);continue
        if row["id"] in seen_ids:
            raise ValueError("New asset ID overlaps old catalog")
        hashes.add(row["cutout_sha256"]);seen_ids.add(row["id"])
        rel = f"cutouts/{row['id']}.png";shutil.copy2(source,args.output/rel)
        published=dict(row,cutout=rel,status="visually_screened_compositing_candidate",
            source_candidate_catalog=str(args.candidates/"catalog.json"),
            review_note=review["notes"].get(key,"Full-aircraft silhouette accepted"),
            commercial_rights_review_pending=True)
        rows.append(published);added.append(published)
    rows.sort(key=lambda r:int(r["id"]))
    unique, aliases = {}, {}
    for row in rows:
        identity = row["cutout_sha256"]
        if identity not in unique:
            unique[identity] = row
            aliases[row["id"]] = []
        aliases[unique[identity]["id"]].append(row["id"])
    summary=dict(original_entries=len(old["records"]),original_unique_png_sha256=len(old_hashes),
        accepted_new_entries=len(added),merged_entries=len(rows),merged_unique_png_sha256=len(hashes),
        new_model_count=len({r["model"] for r in added}),new_models=dict(Counter(r["model"] for r in added)),
        duplicate_new_pngs_not_added=duplicate,base_catalog=str(args.base_catalog),
        base_catalog_sha256=config["base_catalog_sha256"],base_catalog_unchanged=True,
        existing_training_configuration_changed=False,review=review)
    dump(args.output/"catalog.json",dict(records=rows,summary=summary,production_qualification=False))
    dump(args.output/"catalog_unique.json",dict(records=list(unique.values()),
        duplicate_id_aliases=aliases,source_catalog="catalog.json",production_qualification=False))
    dump(args.output/"summary.json",summary)
    dump(args.output/"new_assets.json",dict(records=added))
    shutil.copy2(args.review,args.output/"visual_review.json")
    if args.base_catalog.read_bytes() != original_bytes:
        raise ValueError("Original catalog modified")
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("segment");a.add_argument("--input",type=Path,required=True)
    a.add_argument("--rules",type=Path,required=True);a.add_argument("--model",type=Path,required=True)
    a.add_argument("--model-provenance",type=Path,required=True);a.add_argument("--device",default="cuda:0")
    b=sub.add_parser("publish");b.add_argument("--candidates",type=Path,required=True)
    b.add_argument("--review",type=Path,required=True)
    for parser in (a,b):
        parser.add_argument("--base-catalog",type=Path,required=True)
        parser.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    segment(args) if args.action=="segment" else publish(args)


if __name__=="__main__":main()
