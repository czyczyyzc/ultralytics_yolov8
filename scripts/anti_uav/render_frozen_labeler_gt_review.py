#!/usr/bin/env python3
"""Render one unmodified positive frame per newly appended training video."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.prepare_labeler_training_expansion import dump, frame_plan, sha256


BG = (25, 22, 19)
TEXT = (239, 239, 234)
MUTED = (181, 179, 171)
GREEN = (105, 235, 95)


def label(image, message, x, y, size=.5, color=TEXT):
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def corners(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.rint(np.clip(box, [0, 0, 0, 0], [w-1, h-1, w-1, h-1])).astype(int)
    dx, dy = min(9, (x2-x1)//2), min(9, (y2-y1)//2)
    for x, sx in ((x1, 1), (x2, -1)):
        for y, sy in ((y1, 1), (y2, -1)):
            cv2.line(image, (x, y), (x+sx*dx, y), GREEN, 1, cv2.LINE_8)
            cv2.line(image, (x, y), (x, y+sy*dy), GREEN, 1, cv2.LINE_8)


def render_card(item, source, number, output):
    w, h = item['frame_width'], item['frame_height']
    if (item['status'] != 'approved' or item['revision'] != item['approved_revision']
            or source['revision'] != item['revision']):
        raise ValueError('Expected the same frozen approved revision in both inputs')
    plan, _ = frame_plan(item['annotations'], item['frame_count'], w, h, True)
    positive = sorted(i for i, boxes in plan.items() if boxes)
    if len(positive) != source['positive_samples']:
        raise ValueError('Frozen positive frame count does not match the training manifest')
    frame = positive[len(positive)//2]
    name = f'{frame:09d}'
    image_path = Path(source['image_directory']) / f'{name}.jpg'
    label_path = Path(source['label_directory']) / f'{name}.txt'
    image = cv2.imread(str(image_path))
    rows = np.asarray([list(map(float, line.split())) for line in label_path.read_text().splitlines()])
    expected = np.asarray(plan[frame])
    if (image is None or image.shape[:2] != (h, w) or rows.shape != (len(expected), 5)
            or np.any(rows[:, 0] != 0) or not np.allclose(rows[:, 1:], expected, rtol=0, atol=1e-9)):
        raise ValueError(f'Frozen frame or training label mismatch: {image_path}')

    cx, cy, bw, bh = (expected[0]*[w, h, w, h]).tolist()
    box = np.array([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2])
    side = min(w, h, max(40, int(np.ceil(max(bw, bh)*1.65))))
    left = int(np.clip(cx-side/2, 0, w-side))
    top = int(np.clip(cy-side/2, 0, h-side))
    zoom = 300/side
    crop = cv2.resize(image[top:top+side, left:left+side], (300, 300),
                      interpolation=cv2.INTER_NEAREST if zoom >= 1 else cv2.INTER_AREA)
    marked = crop.copy()
    corners(marked, (box-[left, top, left, top])*zoom)
    full = cv2.resize(image, (768, 432), interpolation=cv2.INTER_AREA)
    locator = box*[768/w, 432/h, 768/w, 432/h]
    if min(locator[2:]-locator[:2]) < 4:
        locator += [-2, -2, 2, 2]
    corners(full, locator)

    canvas = np.full((530, 1416, 3), BG, np.uint8)
    label(canvas, f'VIDEO {number:02d}  /  FROZEN TRAINING FRAME', 14, 27, .74)
    label(canvas, f'SHA {item["source_sha256"][:12]}    |    GT rev {item["revision"]}    |    '
          f'Frame {frame:04d} (zero-based)    |    {frame/item["fps"]:.2f}s', 14, 52, .5, MUTED)
    canvas[76:508, 12:780] = full
    canvas[103:403, 792:1092] = crop
    canvas[103:403, 1104:1404] = marked
    label(canvas, 'TARGET CROP / NO OVERLAY', 792, 91, .53)
    label(canvas, 'SAME CROP / APPROVED GT', 1104, 91, .53)
    label(canvas, f'Original bbox: {bw:.1f} x {bh:.1f} px   |   Display crop {zoom:.1f}x', 792, 434, .55)
    gain = min(960/w, 544/h)
    label(canvas, f'At 960 x 544: {bw*gain:.1f} x {bh*gain:.1f} px', 792, 461, .55, GREEN)
    label(canvas, 'Middle positive frame. No prediction, sharpening or synthetic replacement.', 792, 489, .42, MUTED)

    path = output / f'{number:02d}_{item["source_sha256"][:12]}_frame_{frame:06d}.png'
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f'Cannot write {path}')
    return dict(number=number, original_path=item.get('original_path'), source_sha256=item['source_sha256'],
                frame_zero_based=frame, seconds_at_nominal_fps=frame/item['fps'], image=path.name,
                source_image=str(image_path), source_label=str(label_path), annotation_revision=item['revision'],
                source_bbox_wh=[bw, bh], bbox_wh_at_960x544=[bw*gain, bh*gain], card_sha256=sha256(path))


def write_overview(output, records):
    sheets = []
    for offset in range(0, len(records), 5):
        subset = records[offset:offset+5]
        canvas = np.full((48+272*((len(subset)+1)//2), 1432, 3), BG, np.uint8)
        label(canvas, f'FIXED-WING GT REVIEW    |    VIDEO {subset[0]["number"]:02d} - '
              f'{subset[-1]["number"]:02d}', 16, 33, .76)
        for i, row in enumerate(subset):
            x, y = 8+(i%2)*708, 48+(i//2)*272
            card = cv2.imread(str(output / row['image']))
            canvas[y:y+265, x:x+708] = cv2.resize(card, (708, 265), interpolation=cv2.INTER_AREA)
        name = f'overview_{len(sheets)+1:02d}.png'
        if not cv2.imwrite(str(output / name), canvas):
            raise OSError(f'Cannot write {name}')
        sheets.append(name)
    return sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    snapshot = args.data / 'annotation_snapshot.json'
    digest = sha256(snapshot)
    videos = json.loads(snapshot.read_text())['videos']
    sources = {v['sha256']: v for v in manifest['latest_appended_videos']}
    ids = {v['source_sha256'] for v in videos}
    if (digest != manifest['snapshot_sha256'] or not videos or len(videos) != len(ids) or ids != set(sources)
            or ids & {manifest['validation_sha256'], manifest['test_sha256']}
            or not ids <= set(manifest['train_video_hashes'])):
        raise ValueError('Snapshot does not match the latest training-only append')
    args.output.mkdir(parents=True, exist_ok=False)
    records = [render_card(v, sources[v['source_sha256']], i, args.output) for i, v in enumerate(videos, 1)]
    sheets = write_overview(args.output, records)
    dump(args.output / 'manifest.json', dict(snapshot=str(snapshot), snapshot_sha256=digest,
         selection='Median of each video\'s positive frame indices', frame_index_base=0,
         overviews=sheets, frames=records))
    print(json.dumps(dict(videos=len(records), overviews=sheets, output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
