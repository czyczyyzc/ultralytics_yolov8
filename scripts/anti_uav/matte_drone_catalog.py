#!/usr/bin/env python3
"""Create isolated alpha-only cutout candidates from verified workbook media.

No generated RGB, geometric edits, component deletion, or training-catalog edits.
Model code must be downloaded/reviewed separately at a pinned revision.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import io
import json
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def pack_alpha(rgb, alpha):
    """Keep source RGB verbatim; only suppress negligible (< 4/255) alpha haze."""
    if rgb.dtype != np.uint8 or alpha.dtype != np.uint8 or rgb.shape[:2] != alpha.shape:
        raise ValueError("Expected source-resolution uint8 RGB and alpha")
    alpha = alpha.copy()
    alpha[alpha < 4] = 0
    rgba = np.dstack((rgb, alpha))
    ys, xs = np.where(alpha > 0)
    bounds = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    mask = (alpha >= 128).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)
    fraction = float(mask.mean())
    large = [a for a in areas if a >= max(16, sum(areas) * .01)]
    edge = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    flags = []
    if fraction < .005:
        flags.append("very_small_or_empty_mask")
    if fraction > .90:
        flags.append("nearly_full_image_mask")
    if len(large) > 1:
        flags.append("multiple_components_check_accessories")
    if float(edge.mean()) > .01:
        flags.append("foreground_touches_source_border")
    return rgba, bounds, dict(foreground_fraction=fraction, significant_components=len(large),
        component_count=count - 1, main_component_fraction=areas[0] / sum(areas) if areas else 0.,
        border_foreground_fraction=float(edge.mean()), flags=flags)


def font(size=16):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def checker(size):
    w, h = size
    y, x = np.indices((h, w))
    a = np.where((x // 16 + y // 16) % 2, 190, 230).astype(np.uint8)
    return Image.fromarray(np.repeat(a[..., None], 3, axis=2)).convert("RGBA")


def card(source, rgba, row):
    out = Image.new("RGB", (900, 360), (30, 34, 40))
    draw = ImageDraw.Draw(out)
    draw.text((10, 5), f"ID {row['id']} | {row['model'][:85]}", font=font(17), fill="white")
    for i, (label, bg) in enumerate((("ORIGINAL", None), ("ALPHA / CHECKER", checker((294, 265))),
                                    ("ALPHA / DARK", Image.new("RGBA", (294, 265), (35, 35, 35, 255))))):
        if bg is None:
            bg = Image.new("RGBA", (294, 265), (225, 225, 225, 255))
            im = source.convert("RGBA")
        else:
            im = rgba
        im = ImageOps.contain(im, (294, 265), Image.Resampling.LANCZOS)
        bg.alpha_composite(im, ((294 - im.width) // 2, (265 - im.height) // 2))
        out.paste(bg.convert("RGB"), (i * 300 + 3, 57))
        draw.text((i * 300 + 8, 33), label, font=font(14), fill=(190, 225, 240))
    note = ", ".join(row["quality"]["flags"]) or "No automatic flag; visual review still required"
    draw.text((8, 330), note[:110], font=font(13), fill=(255, 220, 130))
    return out


def gallery(output, rows):
    links = []
    for offset in range(0, len(rows), 15):
        sheet = Image.new("RGB", (1800, 1200), (30, 34, 40))
        for index, row in enumerate(rows[offset:offset + 15]):
            with Image.open(output / row["preview"]) as im:
                sheet.paste(im.resize((600, 240), Image.Resampling.LANCZOS),
                            ((index % 3) * 600, (index // 3) * 240))
        name = f"contact_sheets/page_{offset // 15 + 1:02d}.jpg"
        sheet.save(output / name, quality=95)
        links.append(f'<a href="{name}">Page {offset // 15 + 1}</a>')
    sections = ["<html><meta charset='utf-8'><title>Drone cutout candidates</title>",
        "<style>body{background:#20242a;color:#eee;font:16px sans-serif}img{max-width:100%}a{color:#7ddcff}</style>",
        "<h1>Alpha-only cutout candidates</h1><p>RGB preserved. Not automatically approved for training.</p>",
        "<p>" + " | ".join(links) + "</p>"]
    for row in rows:
        sections.append(f'<h3>{html.escape(row["id"] + " / " + row["model"])}</h3>'
            f'<a href="{row["rgba"]}">Full-resolution RGBA</a> | '
            f'<a href="{row["cutout"]}">Tight cutout</a><br>'
            f'<img loading="lazy" src="{row["preview"]}">')
    (output / "index.html").write_text("\n".join(sections) + "</html>")


def run(args):
    cv2.setNumThreads(1)
    catalog = json.loads(args.catalog.read_text())
    if digest(args.workbook) != catalog["xlsx_sha256"]:
        raise ValueError("Workbook identity mismatch")
    records = [r for r in catalog["records"] if r["status"] == "opaque_requires_segmentation"]
    if args.limit:
        records = [records[i] for i in np.linspace(0, len(records) - 1, min(args.limit, len(records)), dtype=int)]
    model_files = {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()}
    config = dict(workbook_sha256=catalog["xlsx_sha256"], catalog_sha256=digest(args.catalog),
        model_files=model_files, model_revision=args.revision, resolution=1024,
        alpha_floor=4, source_rgb_unchanged=True, script_sha256=digest(Path(__file__)))
    if args.output.exists():
        if not args.resume or json.loads((args.output / "run_config.json").read_text()) != config:
            raise ValueError("Output exists; resume requires identical inputs, model and script")
    else:
        args.output.mkdir(parents=True)
        dump(args.output / "run_config.json", config)
    for name in ("raw", "rgba", "masks", "cutouts", "records", "previews", "contact_sheets"):
        (args.output / name).mkdir(exist_ok=True)

    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    torch.set_num_threads(4)
    torch.manual_seed(20260920)
    model = AutoModelForImageSegmentation.from_pretrained(str(args.model), trust_remote_code=True,
        local_files_only=True).eval().to(args.device)
    transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    rows, memo, started = [], {}, time.time()
    with zipfile.ZipFile(args.workbook) as archive:
        for n, source_row in enumerate(records, 1):
            aid = source_row["id"]
            if not aid.isdecimal():
                raise ValueError("Non-numeric asset ID")
            record_path = args.output / "records" / (aid + ".json")
            if record_path.exists():
                row = json.loads(record_path.read_text())
                for key in ("raw", "rgba", "cutout", "mask"):
                    if digest(args.output / row[key]) != row[key + "_sha256"]:
                        raise ValueError(f"Resume file changed: {aid}/{key}")
                rows.append(row)
                memo[row["embedded_sha256"]] = row
                continue
            data = archive.read(source_row["media"])
            if hashlib.sha256(data).hexdigest() != source_row["embedded_sha256"]:
                raise ValueError(f"Embedded media mismatch: {aid}")
            source = Image.open(io.BytesIO(data)).convert("RGB")
            rgb = np.asarray(source)
            original = dict(source_row)
            key = source_row["embedded_sha256"]
            if key in memo:
                with Image.open(args.output / memo[key]["mask"]) as im:
                    alpha = np.array(im)
                duplicate = memo[key]["id"]
            else:
                with torch.inference_mode():
                    pred = model(transform(source).unsqueeze(0).to(args.device))[-1].sigmoid()
                alpha = (pred[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                alpha = np.array(Image.fromarray(alpha).resize(source.size, Image.Resampling.BILINEAR))
                duplicate = None
            rgba, bounds, quality = pack_alpha(rgb, alpha)
            full = Image.fromarray(rgba)
            crop = full.crop(bounds) if bounds else full
            row = dict(original, status="segmented_candidate_not_training_approved", bounds=bounds,
                width=source.width, height=source.height, quality=quality, duplicate_of=duplicate,
                raw=f"raw/{aid}{Path(original['media']).suffix}", rgba=f"rgba/{aid}.png",
                mask=f"masks/{aid}.png", cutout=f"cutouts/{aid}.png", preview=f"previews/{aid}.jpg")
            (args.output / row["raw"]).write_bytes(data)
            full.save(args.output / row["rgba"])
            crop.save(args.output / row["cutout"])
            Image.fromarray(rgba[..., 3]).save(args.output / row["mask"])
            with Image.open(args.output / row["rgba"]) as check:
                if not np.array_equal(np.array(check)[..., :3], rgb):
                    raise ValueError("RGB changed during PNG roundtrip")
            for field in ("raw", "rgba", "cutout", "mask"):
                row[field + "_sha256"] = digest(args.output / row[field])
            card(source, full, row).save(args.output / row["preview"], quality=95)
            dump(record_path, row)
            rows.append(row)
            memo[key] = row
            print(json.dumps(dict(done=n, total=len(records), id=aid, flags=quality["flags"],
                                  elapsed=round(time.time() - started, 1))), flush=True)
    rows.sort(key=lambda r: int(r["id"]))
    dump(args.output / "catalog.json", dict(records=rows, source_catalog=str(args.catalog),
        workbook=str(args.workbook), config=config, training_approved=False))
    summary = dict(processed=len(rows), unique_source_images=len(memo),
        full_rgba_files=len(rows), automatic_flagged=sum(bool(r["quality"]["flags"]) for r in rows),
        flag_counts=dict(Counter(f for r in rows for f in r["quality"]["flags"])),
        training_approved=0, source_rgb_verified=len(rows), wall_seconds=round(time.time() - started, 1))
    dump(args.output / "summary.json", summary)
    gallery(args.output, rows)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "workbook", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
