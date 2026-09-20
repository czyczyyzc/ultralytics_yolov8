#!/usr/bin/env python3
"""Re-segment visually selected drone ROIs without altering source RGB."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.matte_drone_catalog import card, digest, dump, gallery, pack_alpha


def main(args):
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    torch.set_num_threads(4)
    root = args.library
    catalog = json.loads((root / "catalog.json").read_text())
    rules = json.loads(args.rules.read_text())
    actual_model = {p.name: digest(p) for p in args.model.iterdir() if p.is_file()}
    if actual_model != catalog["config"]["model_files"]:
        raise ValueError("Model files changed")
    target = root / "refined"
    target.mkdir(exist_ok=False)
    for name in ("rgba", "cutouts", "masks", "previews", "contact_sheets"):
        (target / name).mkdir()
    model = AutoModelForImageSegmentation.from_pretrained(str(args.model),
        trust_remote_code=True, local_files_only=True).eval().cuda()
    transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    results = []
    for old in catalog["records"]:
        aid = old["id"]
        if aid not in rules["roi"]:
            continue
        source_path = root / old["raw"]
        if digest(source_path) != old["raw_sha256"]:
            raise ValueError("Source changed")
        source = Image.open(source_path).convert("RGB")
        x0, y0, x1, y1 = rules["roi"][aid]
        w, h = source.size
        bounds = [round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)]
        x0, y0, x1, y1 = bounds
        roi = source.crop(bounds)
        with torch.inference_mode():
            pred = model(transform(roi).unsqueeze(0).cuda())[-1].sigmoid()
        small = Image.fromarray((pred[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
        alpha = np.zeros((h, w), np.uint8)
        alpha[y0:y1, x0:x1] = np.array(small.resize(roi.size, Image.Resampling.BILINEAR))
        rgba, crop, quality = pack_alpha(np.array(source), alpha)
        full = Image.fromarray(rgba)
        row = dict(old, bounds=crop, quality=quality, roi_source_pixels=bounds,
            method="visually_selected_roi_then_birefnet", raw="../" + old["raw"],
            rgba=f"rgba/{aid}.png", cutout=f"cutouts/{aid}.png", mask=f"masks/{aid}.png",
            preview=f"previews/{aid}.jpg", parent_rgba_sha256=old["rgba_sha256"])
        full.save(target / row["rgba"])
        full.crop(crop).save(target / row["cutout"])
        Image.fromarray(rgba[..., 3]).save(target / row["mask"])
        assert np.array_equal(np.array(Image.open(target / row["rgba"]))[..., :3], np.array(source))
        for field in ("rgba", "cutout", "mask"):
            row[field + "_sha256"] = digest(target / row[field])
        card(source, full, row).save(target / row["preview"], quality=95)
        results.append(row)
        print(aid, quality, flush=True)
    dump(target / "catalog.json", dict(records=results, rules=rules,
        rules_sha256=digest(args.rules), source_catalog_sha256=digest(root / "catalog.json"),
        script_sha256=digest(Path(__file__)), model_files=catalog["config"]["model_files"]))
    gallery(target, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    main(parser.parse_args())
