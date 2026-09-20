#!/usr/bin/env python3
"""Export seeded random online replacements and browsable original/bbox comparisons."""
import argparse
from collections import Counter, OrderedDict
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.build_gray_replacement_batch import load_assets
from scripts.anti_uav.online_gray_replacement import compose_cached, decode_prepared
from scripts.anti_uav.synthesize_gray_drone_replacements import (
    SkipSample, normalize_box, parse_boxes, preview_card, render_prepared, sha256,
)


def random_schedule(images, asset_count, count, seed):
    if not 1 <= count <= len(images) or asset_count < 1:
        raise ValueError("Request must fit the unique source-frame pool")
    rng = np.random.default_rng(seed)
    sources = [images[i] for i in rng.permutation(len(images))]
    assets = []
    while len(assets) < count:
        assets.extend(int(i) for i in rng.permutation(asset_count))
    return sources, assets[:count]


def write_gallery(root, samples, count):
    sheets = []
    for page, start in enumerate(range(0, len(samples), 10), 1):
        canvas = Image.new("RGB", (1200, 2000), (25, 29, 35))
        ImageDraw.Draw(canvas).text((18, 12), f"ONLINE SYNTHESIS REVIEW | {page:02d} | ORIGINAL LEFT / REPLACED RIGHT", fill="white")
        for i, row in enumerate(samples[start:start+10]):
            with Image.open(root/row["comparison"]) as card:
                canvas.paste(card.resize((600, 390), Image.Resampling.LANCZOS), ((i % 2)*600, 40+(i//2)*390))
        relative = f"contact_sheets/page_{page:02d}.jpg"
        canvas.save(root/relative, quality=94)
        sheets.append(relative)
    cards = []
    for row in samples:
        title = html.escape(f"{row['number']:03d} | {row['video']} | frame {row['frame']} | asset {row['asset_id']} | {row['asset_model']}")
        links = " | ".join(f'<a href="{row[key]}">{label}</a>' for key, label in (
            ("original", "Original"), ("replacement", "Replaced"), ("label", "Updated YOLO bbox")))
        cards.append(f'<article><h2>{title}</h2><a href="{row["comparison"]}"><img loading="lazy" src="{row["comparison"]}" alt="{title}"></a><p>{links}</p></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grayscale Drone Replacement Review</title><style>
body{margin:0;background:#171d23;color:#e8ebef;font:16px Georgia,serif}main{max-width:1260px;margin:auto;padding:24px}
h1{font-size:34px}h2{font-size:16px;font-weight:normal;overflow-wrap:anywhere}article{margin:32px 0;border-top:1px solid #56616c;padding-top:16px}
img{width:100%;height:auto}a{color:#56d2eb}p{line-height:1.6}
</style><main><h1>Grayscale Drone Replacement Review</h1>'''
    page += f'<p>{count} random unique positive frames; balanced shuffled cutout coverage. Synthetic augmentation, not real captures. '
    page += 'Green: original box. Cyan: updated box. Click an image for full size. Originals and replacements are unannotated PNGs.</p>'
    page += '<p><a href="summary.json">Summary and sampling protocol</a> | <a href="manifest.json">Per-sample provenance</a></p>'
    page += "\n".join(cards)+"</main></html>"
    (root/"index.html").write_text(page)
    return sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cache, root = args.cache.resolve(), args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    summary = json.loads((cache/"summary.json").read_text())
    if summary["stage"] != "complete" or summary["is_smoke_subset"]:
        raise ValueError("Use a completed full cache")
    if sha256(cache/"index.json") != summary["index_sha256"]:
        raise ValueError("Cache index changed")
    index = json.loads((cache/"index.json").read_text())
    catalog = Path(index["catalog"])
    if sha256(catalog) != index["catalog_sha256"]:
        raise ValueError("Asset catalog changed")
    assets = load_assets(catalog, {"24", "25"})
    if [a["id"] for a, _ in assets] != index["asset_ids"]:
        raise ValueError("Asset identity mismatch")
    paths = sorted(index["images"])
    identities = [(r["video_sha256"], r["frame"]) for r in index["images"].values()]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate source frames in cache index")
    if {r["video_sha256"] for r in index["images"].values()} & set(index["heldout_sha256"]):
        raise ValueError("Held-out source in cache")
    sources, choices = random_schedule(paths, len(assets), args.count, args.seed)
    root.mkdir(parents=True)
    for name in ("originals", "replacements", "labels", "original_labels", "comparisons", "contact_sheets"):
        (root/name).mkdir()
    samples, skipped, connections = [], [], OrderedDict()
    started = time.monotonic()
    try:
        for source in sources:
            if len(samples) == args.count:
                break
            info = index["images"][source]
            asset, rgba = assets[choices[len(samples)]]
            path = Path(source)
            if sha256(path) != info["source_sha256"]:
                raise ValueError(f"Changed source: {path}")
            pack = info["pack"]
            if pack not in connections:
                if len(connections) >= 4:
                    connections.popitem(last=False)[1].close()
                connections[pack] = sqlite3.connect(f"file:{cache/pack}?mode=ro", uri=True)
            connections.move_to_end(pack)
            record = connections[pack].execute(
                "SELECT payload,payload_sha,label,label_sha FROM backgrounds WHERE frame=? AND state='ready'",
                (info["frame"],)).fetchone()
            if record is None:
                raise ValueError("Cached background missing")
            payload, payload_hash, source_label, label_hash = record
            if hashlib.sha256(payload).hexdigest() != payload_hash or payload_hash != info["payload_sha"]:
                raise ValueError("Background checksum mismatch")
            if sha256(Path(source_label)) != label_hash or label_hash != info["label_sha256"]:
                raise ValueError("Original label changed")
            prepared = decode_prepared(payload)
            original_bgr = cv2.imread(source)
            if original_bgr is None:
                raise FileNotFoundError(source)
            h, w = original_bgr.shape[:2]
            original_boxes = parse_boxes(Path(source_label).read_text(), w, h)
            if len(original_boxes) != 1 or not np.allclose(original_boxes[0], info["box"], atol=.25, rtol=0):
                raise ValueError("Original label/cache geometry mismatch")
            try:
                replaced_bgr, normalized, metrics = compose_cached(original_bgr, prepared, rgba)
                _, mask, box, _ = render_prepared(prepared, rgba)
            except SkipSample as error:
                skipped.append(dict(source=source, frame=info["frame"], asset_id=asset["id"], reason=str(error)))
                continue
            x, y, side = (prepared["meta"][k] for k in ("left", "top", "side"))
            full_mask = np.zeros((h, w), np.uint8)
            full_mask[y:y+side, x:x+side] = mask
            if not np.array_equal(original_bgr[full_mask == 0], replaced_bgr[full_mask == 0]):
                raise AssertionError("Pixels outside edit mask changed")
            label = normalize_box(box, w, h)
            if not np.allclose(normalized, [float(v) for v in label.split()[1:]], atol=1e-7, rtol=0):
                raise AssertionError("Online normalized box differs from exported label")
            n = len(samples)+1
            stem = f"{n:03d}_{info['video_sha256'][:12]}_f{info['frame']:06d}_a{asset['id']}"
            original, replaced = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in (original_bgr, replaced_bgr))
            row = dict(number=n, source=source, source_sha256=info["source_sha256"],
                video=path.parent.name, video_sha256=info["video_sha256"], frame=info["frame"],
                asset_id=asset["id"], asset_model=asset["model"], synthetic=True, metrics=metrics,
                original=f"originals/{stem}.png", replacement=f"replacements/{stem}.png",
                comparison=f"comparisons/{stem}.png", label=f"labels/{stem}.txt",
                original_label=f"original_labels/{stem}.txt")
            Image.fromarray(original).save(root/row["original"])
            Image.fromarray(replaced).save(root/row["replacement"])
            for key, expected in (("original", original), ("replacement", replaced)):
                with Image.open(root/row[key]) as saved:
                    if not np.array_equal(np.asarray(saved), expected):
                        raise AssertionError("PNG roundtrip failed")
            (root/row["label"]).write_text(label)
            (root/row["original_label"]).write_bytes(Path(source_label).read_bytes())
            preview_card(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(replaced_bgr, cv2.COLOR_BGR2GRAY), info["box"], row, Image.fromarray(rgba), box).save(root/row["comparison"])
            row["replacement_sha256"] = sha256(root/row["replacement"])
            if sha256(path) != info["source_sha256"]:
                raise ValueError("Original changed during export")
            samples.append(row)
            if n % 10 == 0:
                print(json.dumps(dict(accepted=n, attempted=n+len(skipped), seconds=round(time.monotonic()-started, 2))), flush=True)
        if len(samples) != args.count:
            raise RuntimeError("Exhausted safe source candidates before reaching requested count")
        sheets = write_gallery(root, samples, args.count)
        result = dict(accepted=len(samples), attempted=len(samples)+len(skipped), rejected=len(skipped), seed=args.seed,
            source_frames=len({(s["video_sha256"], s["frame"]) for s in samples}),
            source_videos=len({s["video_sha256"] for s in samples}), available_cutouts=len(assets),
            asset_ids_used=sorted({s["asset_id"] for s in samples}, key=int),
            counts_by_asset=dict(Counter(s["asset_id"] for s in samples)), source_pool=len(paths),
            cache=str(cache), cache_index_sha256=summary["index_sha256"],
            source_sampling="Seeded random unique cached positive frames; next random source on cutout-quality rejection",
            asset_sampling="Balanced independently shuffled asset cycles, not independent uniform draws",
            bbox_updated=True, outside_edit_mask_unchanged=True, lossless_png_roundtrip_verified=True,
            training_modified=False, visual_review_candidates=True, contact_sheets=sheets,
            seconds=round(time.monotonic()-started, 2))
        (root/"manifest.json").write_text(json.dumps(dict(summary=result, samples=samples, rejected=skipped), indent=2)+"\n")
        (root/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
    finally:
        for db in connections.values():
            db.close()


if __name__ == "__main__":
    main()
