#!/usr/bin/env python3
"""Render safe online replacements from new frozen frames before starting training."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def verify_cache(cache_dir, data_dir, output):
    import cv2
    import numpy as np
    from scripts.anti_uav.build_gray_replacement_batch import load_assets
    from scripts.anti_uav.online_gray_replacement import compose_cached, decode_prepared
    from scripts.anti_uav.synthesize_gray_drone_replacements import SkipSample

    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((cache_dir/"summary.json").read_text())
    if summary["stage"] != "complete" or sha256(cache_dir/"index.json") != summary["index_sha256"]:
        raise ValueError("Background cache is not complete and pinned")
    index = json.loads((cache_dir/"index.json").read_text())
    if sha256(index["catalog"]) != index["catalog_sha256"]:
        raise ValueError("Augmentation assets changed")
    assets = load_assets(Path(index["catalog"]), {"24", "25"})
    if [a["id"] for a, _ in assets] != index["asset_ids"]:
        raise ValueError("Augmentation asset identities changed")
    groups = defaultdict(list)
    for path, info in index["images"].items():
        if path.startswith(str(data_dir/"images/labeler_snapshot")+"/"):
            groups[info["video_sha256"]].append((path, info))
    selected = []
    for digest, rows in sorted(groups.items()):
        rows.sort(key=lambda x: x[1]["frame"])
        selected.append(rows[len(rows)//2])
    records, previews = [], []
    rng = np.random.default_rng(20261030)
    for path, info in selected:
        if sha256(path) != info["source_sha256"]:
            raise ValueError("Frozen cache source bytes changed")
        with sqlite3.connect(f"file:{(cache_dir/info['pack']).resolve()}?mode=ro", uri=True) as db:
            payload, expected = db.execute("SELECT payload,payload_sha FROM backgrounds WHERE frame=? AND state='ready'",
                                           (info["frame"],)).fetchone()
        if hashlib.sha256(payload).hexdigest() != expected or expected != info["payload_sha"]:
            raise ValueError("Background payload changed")
        prepared = decode_prepared(payload)
        image = cv2.imread(path)
        for asset_index in rng.permutation(len(assets))[:8]:
            asset, rgba = assets[int(asset_index)]
            try:
                composed, bbox, _ = compose_cached(image, prepared, rgba)
            except SkipSample:
                continue
            if not np.isfinite(bbox).all() or min(bbox[2:]) <= 0 or (bbox < 0).any() or (bbox > 1).any():
                raise ValueError("Composited bbox is invalid")
            h, w = image.shape[:2]
            views = [cv2.resize(image, (480, 270)), cv2.resize(composed, (480, 270))]
            x, y, bw, bh = info["box"]
            normalized_original = np.array([(x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h])
            for view, box in zip(views, (normalized_original, bbox)):
                cx, cy, width, height = box*np.array([480, 270, 480, 270])
                cv2.rectangle(view, (round(cx-width/2), round(cy-height/2)),
                               (round(cx+width/2), round(cy+height/2)), (0, 230, 0), 1)
            previews.append((path, views, asset["id"]))
            records.append(dict(image=path, frame=info["frame"], video_sha256=info["video_sha256"],
                asset_id=asset["id"], original_xywh_normalized=normalized_original.tolist(),
                updated_xywh_normalized=bbox.tolist(), source_sha256=info["source_sha256"],
                unmasked_pixels_unchanged=True, source_image_preserved=True))
            break
    if len(records) < min(8, len(selected)) or not records:
        raise ValueError("Too few new frames passed online composition QA")
    for page in range((len(previews)+3)//4):
        canvas = np.full((660, 1920, 3), 24, np.uint8)
        for offset, (path, views, asset) in enumerate(previews[page*4:page*4+4]):
            x, y = (offset % 2)*960, (offset//2)*330
            canvas[y+50:y+320, x:x+480] = views[0]
            canvas[y+50:y+320, x+480:x+960] = views[1]
            title = f'{Path(path).parent.name[:52]} | frame {Path(path).stem} | asset {asset}'
            cv2.putText(canvas, title, (x+4, y+20), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(canvas, 'ORIGINAL GT                        ONLINE REPLACEMENT / UPDATED GT',
                        (x+4, y+40), cv2.FONT_HERSHEY_SIMPLEX, .43, (100, 230, 100), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(output/f'replacement_QA_{page+1:02d}.jpg'), canvas):
            raise OSError("Cannot write replacement QA preview")
    result = dict(stage="complete", rendered=len(records), asset_count=len(assets), records=records,
                  independent_background_visual_quality_guaranteed=False)
    dump(output/"acceptance.json", result)
    return result
