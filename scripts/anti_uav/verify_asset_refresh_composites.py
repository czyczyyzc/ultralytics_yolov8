#!/usr/bin/env python3
"""Require every new asset to compose safely at three source-pixel target scales."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image

from scripts.anti_uav.build_gray_replacement_batch import load_assets
from scripts.anti_uav.online_gray_replacement import compose_cached, decode_prepared
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256
from scripts.anti_uav.synthesize_gray_drone_replacements import SkipSample, preview_card


def verify(cache, asset_ids, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((cache / 'summary.json').read_text())
    if summary['stage'] != 'complete' or sha256(cache / 'index.json') != summary['index_sha256']:
        raise ValueError('Incomplete or changed background cache')
    index = json.loads((cache / 'index.json').read_text())
    if sha256(index['catalog']) != index['catalog_sha256']:
        raise ValueError('Asset catalog changed')
    assets = [(r, a) for r, a in load_assets(Path(index['catalog']), {'24', '25'}) if r['id'] in asset_ids]
    if len(assets) != len(set(asset_ids)):
        raise ValueError('Missing or duplicated requested asset')
    bins = dict(tiny=(8, 16), medium=(32, 64), large=(96, 160))
    records, skipped = [], Counter()
    cards = []
    pools = {name: [(p, r) for p, r in index['images'].items() if lo <= max(r['box'][2:]) <= hi]
             for name, (lo, hi) in bins.items()}
    for row, rgba in assets:
        source_catalog = row.get('source_candidate_catalog')
        if source_catalog:
            with Image.open(Path(source_catalog).parent / row['preview']) as image:
                cards.append(image.convert('RGB'))
        for name in bins:
            for path, info in sorted(pools[name], key=lambda p: (p[1]['video_sha256'], p[1]['frame']))[:256]:
                if info['video_sha256'] in index['heldout_sha256'] or sha256(path) != info['source_sha256']:
                    raise ValueError('Heldout or changed source image')
                with sqlite3.connect(f"file:{(cache / info['pack']).resolve()}?mode=ro", uri=True) as db:
                    payload, expected = db.execute("SELECT payload,payload_sha FROM backgrounds WHERE frame=? AND state='ready'",
                                                   (info['frame'],)).fetchone()
                if hashlib.sha256(payload).hexdigest() != expected or expected != info['payload_sha']:
                    raise ValueError('Background payload changed')
                original = cv2.imread(path)
                if original is None:
                    raise FileNotFoundError(path)
                before = original.copy()
                try:
                    composed, bbox, metrics = compose_cached(original, decode_prepared(payload), rgba)
                except SkipSample as error:
                    skipped[str(error)] += 1
                    continue
                if not np.array_equal(original, before):
                    raise ValueError('Composition modified source image')
                if not np.isfinite(bbox).all() or min(bbox[2:]) <= 0 or (bbox < 0).any() or (bbox > 1).any():
                    raise ValueError('Invalid updated bbox')
                height, width = composed.shape[:2]
                new_box = [(bbox[0] - bbox[2] / 2) * width, (bbox[1] - bbox[3] / 2) * height,
                           bbox[2] * width, bbox[3] * height]
                preview = preview_card(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(composed, cv2.COLOR_BGR2GRAY), info['box'],
                    dict(video=Path(path).parent.name, frame=info['frame'], asset_id=row['id'],
                         asset_model=row['model'], metrics=metrics), Image.fromarray(rgba), new_box=new_box)
                preview.save(output / f"asset_{row['id']}_{name}.jpg", quality=95)
                records.append(dict(asset_id=row['id'], scale_bin_source_pixels=name,
                    source_long_edge_px=max(info['box'][2:]), source_image=path, frame=info['frame'],
                    updated_xywh_normalized=bbox.tolist(), metrics=metrics,
                    source_pixels_preserved=True, outside_edit_mask_unchanged=True))
                break
            else:
                raise ValueError(f"No safe composite for asset {row['id']} at {name}")
    result = dict(complete=True, catalog=index['catalog'], catalog_sha256=index['catalog_sha256'],
        asset_ids=sorted(asset_ids, key=int), rendered=len(records), records=records,
        bins_source_pixels=bins, skipped_attempts=dict(skipped),
        perfect_background_fusion_guaranteed=False, detector_accuracy_evaluated=False)
    dump(output / 'composition_QA.json', result)
    if cards:
        sheet = Image.new('RGB', (1800, ((len(cards) + 1) // 2) * 360), '#20242a')
        for position, card in enumerate(cards):
            sheet.paste(card, (position % 2 * 900, position // 2 * 360))
        sheet.save(output / 'new_asset_contact_sheet.jpg', quality=95)
    sheet = Image.new('RGB', (1200, ((len(assets) + 1) // 2) * 390), '#20242a')
    for position, (row, _) in enumerate(assets):
        with Image.open(output / f"asset_{row['id']}_medium.jpg") as image:
            sheet.paste(image.resize((600, 390), Image.Resampling.LANCZOS),
                        (position % 2 * 600, position // 2 * 390))
    sheet.save(output / 'composition_contact_sheet.jpg', quality=95)
    print(json.dumps(dict(rendered=len(records), assets=result['asset_ids'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--asset-ids', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    verify(a.cache.resolve(), a.asset_ids, a.output.resolve())


if __name__ == '__main__':
    main()
