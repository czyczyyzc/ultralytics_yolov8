#!/usr/bin/env python3
"""Freeze supplied photo bytes and view notes for the alpha-only asset importer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageOps


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def stage(plan, output):
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for number, item in enumerate(plan['photos'], 1):
        source = Path(item['source']).resolve()
        if source.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            raise ValueError('Unsupported source image')
        with Image.open(source) as original:
            displayed = ImageOps.exif_transpose(original)
            size = list(displayed.size)
        rows.append(dict(item, input_id=str(number), relative=f'beluga/{number:02d}{source.suffix.lower()}',
                         sha256=digest(source), source_photo=str(source), source_photo_sha256=digest(source),
                         displayed_wh=size, source_kind='supplied_photo', source_rgb_regenerated=False))
    output.mkdir(parents=True)
    (output / 'inputs/beluga').mkdir(parents=True)
    for row in rows:
        target = output / 'inputs' / row['relative']
        shutil.copy2(row['source_photo'], target)
        if digest(target) != row['sha256']:
            raise ValueError('Photo copy checksum differs')
    (output / 'provenance.json').write_text(json.dumps(dict(records=rows, accepted_for_training=False),
                                                    indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(photos=len(rows), displayed_dimensions=[r['displayed_wh'] for r in rows])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage(json.loads(args.plan.read_text()), args.output)


if __name__ == '__main__':
    main()
