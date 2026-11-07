#!/usr/bin/env python3
"""Export reviewable video ROIs without seeking, resizing or regenerating pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def frame_indices(windows, step):
    if step < 1 or not windows:
        raise ValueError('Positive step and nonempty windows required')
    result = set()
    for start, stop in windows:
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop:
            raise ValueError('Expected zero-based, half-open frame intervals')
        result.update(range(start, stop, step))
    return result


def export(video_root, plan, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / 'inputs').mkdir()
    (output / 'contact_sheets').mkdir()
    cv2.setNumThreads(1)
    rows, videos, names = [], [], set()
    for item in plan['videos']:
        source = (video_root / item['file']).resolve()
        if not source.is_relative_to(video_root.resolve()):
            raise ValueError('Video path escapes root')
        name = item['name']
        if not name or Path(name).name != name or name in names:
            raise ValueError('Unique single-component video name required')
        names.add(name)
        wanted = frame_indices(item['windows'], item.get('step', 3))
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f'Cannot open {source}')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError('Positive finite source FPS required')
        source_kind = item.get('source_kind', 'display_recording_candidate')
        if source_kind not in {'display_recording_candidate', 'direct_aircraft_capture'}:
            capture.release()
            raise ValueError('Unknown video source kind')
        x0, y0, x1, y1 = item['roi_xyxy']
        if not all(isinstance(x, int) for x in [x0, y0, x1, y1]) or not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError('Integer source-pixel ROI required')
        source_sha = digest(source)
        video_rows = []
        try:
            for index in range(max(wanted) + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Video ended before requested frame {index}: {source}')
                if index not in wanted:
                    continue
                height, width = frame.shape[:2]
                if x1 > width or y1 > height:
                    raise ValueError('ROI outside source frame')
                relative = f'{name}/frame_{index:06d}.png'
                target = output / 'inputs' / relative
                target.parent.mkdir(exist_ok=True)
                crop = frame[y0:y1, x0:x1]
                if not cv2.imwrite(str(target), crop):
                    raise OSError('PNG write failed')
                if not np.array_equal(cv2.imread(str(target)), crop):
                    raise ValueError('PNG changed decoded source pixels')
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                have_pts = np.isfinite(timestamp) and timestamp >= 0 and (index == 0 or timestamp > 0)
                row = dict(relative=relative, source_video=str(source), source_video_sha256=source_sha,
                           source_frame_zero_based=index, source_time_seconds=timestamp if have_pts else index / fps,
                           source_nominal_time_seconds=index / fps,
                           timestamp_basis='decoder_pts' if have_pts else 'nominal_frame_rate_fallback',
                           source_fps=fps, source_frame_wh=[width, height], roi_xyxy=[x0, y0, x1, y1],
                           sha256=digest(target), laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                           source_kind=source_kind, original_decoded_pixels_preserved=True,
                           requires_review_for_hud_occlusion=source_kind == 'display_recording_candidate',
                           requires_review_for_support_lines=bool(item.get('suspension_lines_present', False)),
                           visible_surface=item.get('visible_surface', 'unverified'))
                rows.append(row)
                video_rows.append(row)
        finally:
            capture.release()
        if len(video_rows) != len(wanted) or digest(source) != source_sha:
            raise ValueError('Incomplete extraction or changed video')
        videos.append(dict(file=item['file'], name=name, sha256=source_sha, exported=len(video_rows)))
    rows.sort(key=lambda row: row['relative'])
    for index, row in enumerate(rows, 1):
        row['input_id'] = str(index)
    dump(output / 'provenance.json', dict(records=rows, videos=videos, plan=plan,
        source_rgb_regenerated=False, no_resize=True, accepted_for_training=False))
    font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    for offset in range(0, len(rows), 24):
        sheet = Image.new('RGB', (1600, 1440), '#20242a')
        draw = ImageDraw.Draw(sheet)
        for position, row in enumerate(rows[offset:offset + 24]):
            x, y = position % 4 * 400, position // 4 * 240
            with Image.open(output / 'inputs' / row['relative']) as image:
                image.thumbnail((396, 210), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + (400 - image.width) // 2, y + 27))
            label = f"ID {row['input_id']} | {row['relative'].split('/')[0]} | frame {row['source_frame_zero_based']}"
            draw.text((x + 5, y + 5), label, font=font, fill='white')
        sheet.save(output / 'contact_sheets' / f'page_{offset // 24 + 1:02d}.jpg', quality=95)
    print(json.dumps(dict(candidates=len(rows), videos=videos), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.video_root, json.loads(args.plan.read_text()), args.output)


if __name__ == '__main__':
    main()
