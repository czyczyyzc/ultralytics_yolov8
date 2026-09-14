#!/usr/bin/env python3
"""Merge approved video annotations with an audited six-video/RGB training schedule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import yaml

from scripts.anti_uav.build_manual_gray_video_rehearsal import (
    label_path, replace_duplicate_class_slots, sha256_file,
)


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset: {relative}")
    return path


def parse_label(text: str) -> list[list[float]]:
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [float(x) for x in line.split()]
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError(f"Invalid label: {line}")
        cls, x, y, w, h = values
        if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized box: {line}")
        if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
            raise ValueError(f"Box outside frame: {line}")
        result.append(values)
    return result


def validate_frame_sets(manifest: dict) -> tuple[set[int], set[int]]:
    f, v = manifest['frames'], manifest['video']
    included = set(f['includedFrameIndices'])
    negative = set(f['negativeFrameIndices'])
    uncertain = set(f.get('excludedUncertainFrameIndices', []))
    unreviewed = set(f.get('excludedUnreviewedFrameIndices', []))
    if f['indexBase'] != 0 or len(included) != len(f['includedFrameIndices']):
        raise ValueError('Invalid or duplicate frame indices')
    if not negative <= included or included & (uncertain | unreviewed):
        raise ValueError('Conflicting frame inclusion policy')
    if included | uncertain | unreviewed != set(range(v['frameCount'])):
        raise ValueError('Missing or out-of-range frame classification')
    return included, negative


def audit_base(source_paths: list[Path], old_root: Path, holdout: str, rgb_root: Path) -> dict:
    old_manifest = json.loads((old_root / 'manifest.json').read_text())
    records = {Path(v['video_name']).stem: v for v in old_manifest['videos']}
    holdout_hash = records[holdout]['video_sha256']
    annotations, hashes = {}, {}
    for name, record in records.items():
        video = Path(record['video_path'])
        if sha256_file(video) != record['video_sha256']:
            raise ValueError(f'Old video checksum failed: {video}')
        annotation = checked_path(old_root, record['annotation_path'])
        if sha256_file(annotation) != record['annotation_sha256']:
            raise ValueError(f'Old annotation checksum failed: {annotation}')
        annotations[name] = json.loads(annotation.read_text())
        hashes[name] = record['video_sha256']
    seen = set()
    for image in set(source_paths):
        if not image.is_file():
            raise FileNotFoundError(image)
        text = label_path(image).read_text()
        boxes = parse_label(text)
        if image.is_relative_to(rgb_root):
            continue
        name = image.parent.name
        if name not in records or name == holdout:
            raise ValueError(f'Unexpected/held-out base training image: {image}')
        frame = int(image.stem)
        annotation = annotations[name]
        if len(boxes) != int(bool(annotation['exist'][frame])):
            raise ValueError(f'Base label presence differs from old GT: {image}')
        if boxes:
            image_data = cv2.imread(str(image))
            if image_data is None:
                raise ValueError(f'Cannot decode {image}')
            height, width = image_data.shape[:2]
            _, cx, cy, bw, bh = boxes[0]
            actual = [(cx-bw/2)*width, (cy-bh/2)*height, bw*width, bh*height]
            expected = annotation['gt_rect'][frame]
            if max(abs(a-b) for a, b in zip(actual, expected)) > 0.1:
                raise ValueError(f'Base box differs from old GT: {image}: {actual} != {expected}')
        seen.add(name)
    if seen != set(records) - {holdout}:
        raise ValueError(f'Expected six non-held-out videos, found {seen}')
    return dict(training_video_hashes={k:hashes[k] for k in sorted(seen)},
                holdout=holdout, holdout_sha256=holdout_hash,
                base_unique_samples=len(set(source_paths)), old_labels_match_source=True)


def extract_task(task: Path, video_root: Path, output: Path, blocked_hashes: set[str],
                 positive_stride: int, negative_stride: int) -> tuple[list[Path], list[Path], dict]:
    manifest_path = task / 'manifest.json'
    m = json.loads(manifest_path.read_text())
    v = m['video']
    video = checked_path(video_root, v['name'])
    actual_hash = sha256_file(video)
    if actual_hash != v['sha256'] or actual_hash in blocked_hashes:
        raise ValueError(f'Video checksum/duplicate/holdout conflict: {video}')
    blocked_hashes.add(actual_hash)
    included, negative = validate_frame_sets(m)
    expected_label_ids = {int(f.stem) for f in (task / 'yolo/labels').glob('*.txt')}
    if expected_label_ids != included:
        raise ValueError(f'Label frame set mismatch: {task}')
    for entry in m['files']:
        f = checked_path(task, entry['path'])
        if f.stat().st_size != entry['size'] or sha256_file(f) != entry['sha256']:
            raise ValueError(f'Annotation checksum failed: {f}')
    labels = {}
    for frame in sorted(included):
        text = (task / 'yolo/labels' / f'{frame:09d}.txt').read_text()
        boxes = parse_label(text)
        if bool(boxes) != (frame not in negative):
            raise ValueError(f'Presence mismatch: {task} frame {frame}')
        if frame % (negative_stride if frame in negative else positive_stride) == 0:
            labels[frame] = text
    key = f'{video.stem}_{actual_hash[:12]}'
    images = output / 'images' / 'approved_gray' / key
    targets = output / 'labels' / 'approved_gray' / key
    images.mkdir(parents=True)
    targets.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {video}')
    positives, negatives = [], []
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape[:2] != (v['frameHeight'], v['frameWidth']):
                raise ValueError(f'Video dimensions mismatch: {video}')
            if index in labels:
                path = images / f'{index:09d}.jpg'
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(path)
                (targets / f'{index:09d}.txt').write_text(labels[index])
                (negatives if index in negative else positives).append(path)
            index += 1
    finally:
        capture.release()
    if index != v['frameCount'] or len(positives)+len(negatives) != len(labels):
        raise ValueError(f'Frame count mismatch: {video}: {index}')
    record = dict(task=str(task), manifest_sha256=sha256_file(manifest_path),
                  video=str(video), sha256=actual_hash, decoded_frames=index,
                  included_frames=len(included), positive_frames=len(included-negative),
                  negative_frames=len(negative), excluded_frames=index-len(included),
                  positive_samples=len(positives), negative_samples=len(negatives),
                  image_directory=str(images), label_directory=str(targets))
    print(json.dumps(record), flush=True)
    return positives, negatives, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--approved-root', type=Path, required=True)
    p.add_argument('--video-root', type=Path, required=True)
    p.add_argument('--old-root', type=Path, required=True)
    p.add_argument('--source-data', type=Path, required=True)
    p.add_argument('--rgb-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--holdout', default='Video00004')
    p.add_argument('--positive-stride', type=int, default=3)
    p.add_argument('--negative-stride', type=int, default=20)
    p.add_argument('--seed', type=int, default=20260904)
    a = p.parse_args()
    if min(a.positive_stride, a.negative_stride) < 1:
        raise ValueError('Invalid stride')
    if a.output.exists():
        raise FileExistsError(f'Use a new output directory: {a.output}')
    cv2.setNumThreads(1)
    data = yaml.safe_load(a.source_data.read_text())
    source_list = Path(data['train'])
    paths = [Path(line) for line in source_list.read_text().splitlines() if line.strip()]
    audit = audit_base(paths, a.old_root, a.holdout, a.rgb_root)
    print(json.dumps({'base_audit':audit}), flush=True)
    tasks = sorted(a.approved_root.glob('*/manifest.json'))
    if not tasks:
        raise ValueError('No approved manifests')
    blocked = set(audit['training_video_hashes'].values()) | {audit['holdout_sha256']}
    positives, negatives, records = {}, {}, []
    for manifest in tasks:
        pos, neg, record = extract_task(manifest.parent, a.video_root, a.output, blocked,
                                       a.positive_stride, a.negative_stride)
        positives[record['sha256']] = pos
        negatives[record['sha256']] = neg
        records.append(record)
    source_positive = sum(bool(label_path(path).read_text().strip()) for path in paths)
    # Every video must cycle through all its samples, including the longest clip.
    quota = max(round(source_positive * 0.10), max(map(len, positives.values())) * len(positives))
    fraction = quota / source_positive
    schedule, sampling = replace_duplicate_class_slots(paths, positives, negatives, fraction, a.seed)
    train_list = a.output / 'train_manual_gray_rehearsal.txt'
    train_list.write_text(''.join(f'{path}\n' for path in schedule))
    data.update(path=str(a.output), train=str(train_list))
    (a.output / 'train_rgb_monitor.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    report = dict(schema_version='anti_uav.approved_gray_rehearsal.v1',
                  old_root=str(a.old_root), source_data=str(a.source_data),
                  source_train_sha256=sha256_file(source_list), base_audit=audit,
                  approved_tasks=records, training_gray_videos=len(records)+6,
                  positive_stride=a.positive_stride, negative_stride=a.negative_stride,
                  seed=a.seed, positive_rehearsal_fraction=fraction, sampling=sampling,
                  train_list=str(train_list), train_list_sha256=sha256_file(train_list),
                  selection='RGB validation only; Video00004 is final test only',
                  limitation='SHA256 identity check excludes exact duplicate videos, not reencoded/overlapping clips')
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
