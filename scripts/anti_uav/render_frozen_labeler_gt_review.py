#!/usr/bin/env python3
"""Render one unmodified positive frame per approved training candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
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

    return render_frame(item, number, output, frame, image, expected[0],
                        source_image=str(image_path), source_label=str(label_path))


def render_frame(item, number, output, frame, image, box_xywh, **provenance):
    w, h = item['frame_width'], item['frame_height']
    cx, cy, bw, bh = (np.asarray(box_xywh)*[w, h, w, h]).tolist()
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
    source_title = item.get('preview_title', 'FROZEN TRAINING FRAME')
    seconds = item.get('frame_pts_s')
    if seconds is None:
        seconds = frame/item['fps']
    label(canvas, f'VIDEO {number:02d}  /  {source_title}', 14, 27, .74)
    label(canvas, f'SHA {item["source_sha256"][:12]}    |    GT rev {item["revision"]}    |    '
          f'Frame {frame:04d} (zero-based)    |    {seconds:.2f}s', 14, 52, .5, MUTED)
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
                annotation_revision=item['revision'], display_seconds=seconds, source_bbox_wh=[bw, bh],
                bbox_wh_at_960x544=[bw*gain, bh*gain], card_sha256=sha256(path), **provenance)


def pinned_asset(root, relative, digest):
    path = (root / relative).resolve()
    path.relative_to(root)
    if sha256(path) != digest:
        raise ValueError(f'Published frame asset checksum changed: {path}')
    return path


def render_untrained_approvals(database, assets, manifest, output):
    blocked = set(manifest['train_video_hashes']) | {manifest['validation_sha256'], manifest['test_sha256']}
    assets = assets.resolve()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        rows = db.execute("""
            SELECT t.id AS task_id,t.revision,t.approved_revision,v.source_sha256,v.name,
              v.frame_count,v.frame_width,v.frame_height,v.fps,v.dataset_category,
              (SELECT mi.display_path FROM media_imports mi WHERE mi.task_id=t.id AND mi.status='succeeded'
                ORDER BY mi.created_at DESC LIMIT 1) AS original_path,
              a.id AS snapshot_id,a.annotations_json,a.content_sha256,a.created_at AS approved_at
            FROM tasks t JOIN videos v ON v.id=t.video_id
            JOIN approval_snapshots a ON a.task_id=t.id AND a.revision=t.approved_revision
            WHERE t.is_published=1 AND t.status='approved' AND v.validation_status='ready'
            ORDER BY a.created_at,v.source_sha256
        """).fetchall()
        candidates = [dict(row) for row in rows if row['source_sha256'] not in blocked]
        if len(candidates) != len({row['source_sha256'] for row in candidates}):
            raise ValueError('More than one current approval for a candidate video')
        records = []
        for number, item in enumerate(candidates, 1):
            if item['revision'] != item['approved_revision']:
                raise ValueError('Live revision differs from approved revision')
            snapshot = item.pop('annotations_json')
            if hashlib.sha256(snapshot.encode()).hexdigest() != item['content_sha256']:
                raise ValueError('Approval snapshot checksum changed')
            annotations = []
            for row in json.loads(snapshot)['annotations']:
                box = row['bboxXyxy'] or [None]*4
                annotations.append(dict(frame_index=row['frameIndex'], status=row['status'],
                                        **dict(zip(('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'), box))))
            w, h = item['frame_width'], item['frame_height']
            plan, excluded = frame_plan(annotations, item['frame_count'], w, h, True)
            positive = sorted(i for i, boxes in plan.items() if boxes)
            if not positive:
                raise ValueError(f'No approved positive in {item["name"]}')
            frame = positive[len(positive)//2]
            matches = db.execute("""
                SELECT s.id AS frame_set_id,s.manifest_relpath,s.manifest_sha256,s.source_sha256,s.mode,
                  f.image_id,f.image_relpath,f.image_sha256,f.width,f.height,f.pts_s,f.source_ordinals_json
                FROM approval_asset_bindings b JOIN frame_sets s ON s.id=b.frame_set_id
                JOIN frame_images f ON f.frame_set_id=s.id AND f.ordinal=?
                WHERE b.snapshot_id=?
            """, (frame, item['snapshot_id'])).fetchall()
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous exact frame binding')
            bound, = matches
            if (bound['mode'] != 'sequential' or bound['source_sha256'] != item['source_sha256']
                    or (bound['width'], bound['height']) != (w, h)
                    or json.loads(bound['source_ordinals_json']) != [frame]):
                raise ValueError('Bound image differs from the approved source video/frame')
            path = pinned_asset(assets, bound['manifest_relpath'], bound['manifest_sha256'])
            frame_manifest = json.loads(path.read_text())
            entry = frame_manifest['frames'][frame]
            if (frame_manifest['sourceSha256'] != item['source_sha256'] or entry['ordinal'] != frame
                    or entry['imageId'] != bound['image_id'] or entry['imageSha256'] != bound['image_sha256']
                    or entry['imagePath'] != bound['image_relpath']):
                raise ValueError('Frame-set manifest differs from the approved binding')
            path = pinned_asset(assets, bound['image_relpath'], bound['image_sha256'])
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (h, w):
                raise ValueError('Published frame dimensions changed')
            item['preview_title'] = 'APPROVED FRAME / NOT IN TRAINING'
            item['frame_pts_s'] = bound['pts_s']
            records.append(render_frame(item, number, output, frame, image, plan[frame][0],
                source_image=str(path), source_image_sha256=bound['image_sha256'], frame_set_id=bound['frame_set_id'],
                approval_snapshot_id=item['snapshot_id'], approval_snapshot_sha256=item['content_sha256'],
                approved_at=item['approved_at'], positive_frames=len(positive),
                reviewed_absent_frames=len(plan)-len(positive), excluded_frames=excluded,
                dataset_category=item['dataset_category']))
    return records


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
    parser.add_argument('--database', type=Path, help='Preview new approvals outside the baseline training scope')
    parser.add_argument('--frame-assets', type=Path, help='Root of the approval-bound frame assets')
    args = parser.parse_args()
    if bool(args.database) != bool(args.frame_assets):
        parser.error('--database and --frame-assets must be supplied together')
    cv2.setNumThreads(1)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if args.database:
        args.output.mkdir(parents=True, exist_ok=False)
        records = render_untrained_approvals(args.database, args.frame_assets, manifest, args.output)
        dump(args.output / 'manifest.json', dict(baseline_manifest=str(args.data / 'manifest.json'),
            baseline_training_videos=len(manifest['train_video_hashes']), source='approval-bound frames and snapshot GT',
            frame_index_base=0, overviews=write_overview(args.output, records), frames=records))
        print(json.dumps(dict(videos=len(records), output=str(args.output)), ensure_ascii=False))
        return
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
