#!/usr/bin/env python3
"""Convert official UAVSwarm human MOT GT to audited positive-only YOLO data.

Example: python scripts/anti_uav/prepare_uavswarm_research.py --root RAW --output OUT
The input must be the extracted repository root containing train/ and test/.
No downloads, extraction, training, or changes to source data are performed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile

from PIL import Image


OFFICIAL_SOURCE = "https://github.com/UAVSwarm/UAVSwarm-dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
QUARANTINED_GROUPS = {
    "uavswarm-18": "V1 GT alignment issue #8; no verified fix in the pinned official release"
}


def mot_box_to_yolo(x, y, width, height, image_width, image_height, xy_origin=0):
    """Preserve official numeric x/y by default; shifting requires an explicit override.

    Official train.json preserves sampled gt.txt x/y/w/h unchanged. The README
    does not specify a pixel origin; the MOT name alone is not evidence for -1.
    """
    if xy_origin not in (0, 1):
        raise ValueError("xy_origin must be 0 (preserve official) or 1 (explicit -1 override)")
    if not all(math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError("Non-finite MOT box")
    if min(width, height) <= 0:
        return None
    left, top = max(0.0, x - xy_origin), max(0.0, y - xy_origin)
    right = min(float(image_width), x - xy_origin + width)
    bottom = min(float(image_height), y - xy_origin + height)
    if right <= left or bottom <= top:
        return None
    return ((left + right) / (2 * image_width), (top + bottom) / (2 * image_height),
            (right - left) / image_width, (bottom - top) / image_height)


def read_gt(path):
    """Read only gt/gt.txt; never substitute detector output or COCO test.json."""
    content = path.read_bytes()
    frames, seen = defaultdict(list), set()
    for number, row in enumerate(csv.reader(io.StringIO(content.decode("utf-8-sig"))), 1):
        if not row or not any(value.strip() for value in row):
            continue
        where = f"{path}:{number}"
        try:
            values = [float(value) for value in row]
        except ValueError as exc:
            raise ValueError(f"Non-numeric MOT row at {where}") from exc
        if len(values) != 9 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Expected nine finite MOT values at {where}")
        frame, identity, x, y, width, height, active, category, visibility = values
        if frame < 1 or frame != int(frame):
            raise ValueError(f"Expected one-based integer frame at {where}")
        if identity < 1 or identity != int(identity):
            raise ValueError(f"Invalid GT track ID at {where}")
        if active not in (0, 1) or category != int(category) or not 0 <= visibility <= 1:
            raise ValueError(f"Invalid GT flags at {where}")
        key = int(frame), int(identity)
        if key in seen:
            raise ValueError(f"Duplicate frame/track GT at {where}")
        seen.add(key)
        frames[int(frame)].append((x, y, width, height, active, category, visibility))
    return frames, hashlib.sha256(content).hexdigest()


def image_frames(sequence):
    frames = {}
    directory = sequence / "img1"
    if not directory.is_dir():
        raise ValueError(f"Missing image directory: {directory}")
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not path.is_file() or not path.stem.isdecimal() or int(path.stem) < 1:
            raise ValueError(f"Expected readable one-based numeric image filename: {path}")
        frame = int(path.stem)
        if frame in frames:
            raise ValueError(f"Duplicate image frame {frame} in {sequence}")
        frames[frame] = path.resolve()
    if not frames:
        raise ValueError(f"No images in {directory}")
    return frames


def discover_sequences(root):
    result, groups, aliases = [], {}, {}
    for split in ("train", "test"):
        directory = root / split
        if not directory.is_dir():
            raise ValueError(f"Missing official split: {directory}")
        sequences = sorted(path for path in directory.iterdir() if path.is_dir())
        if not sequences:
            raise ValueError(f"Empty official split: {directory}")
        for sequence in sequences:
            key = sequence.name.casefold()
            if key in groups:
                raise ValueError(f"Duplicate group/cross-split group: {sequence.name}")
            if sequence.resolve() in aliases:
                raise ValueError(f"Duplicate sequence directory alias: {sequence}")
            groups[key], aliases[sequence.resolve()] = split, split
            if not (sequence / "gt" / "gt.txt").is_file():
                raise ValueError(f"Missing human GT; det files cannot substitute: {sequence}")
            result.append((split, sequence))
    return result


def write_notebook(path, output):
    code = (
        "from pathlib import Path\nimport json\nfrom collections import Counter\n"
        f"root = Path({str(output)!r})\n"
        "audit = json.loads((root / 'audit.json').read_text())\n"
        "rows = [json.loads(x) for x in (root / 'manifest.jsonl').read_text().splitlines()]\n"
        "groups = {}\nfor row in rows:\n"
        "    key = row['dataset'], row['group']\n"
        "    assert groups.setdefault(key, row['split']) == row['split']\n"
        "    assert Path(row['image']).is_file() and Path(row['label']).is_file()\n"
        "    assert row['source'].endswith('/gt/gt.txt')\n"
        "    assert len(Path(row['label']).read_text().splitlines()) == row['boxes'] > 0\n"
        "print(Counter(row['split'] for row in rows))\naudit\n"
    )
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}}, cells=[
        dict(cell_type="markdown", id="policy", metadata={}, source=[
            "# UAVSwarm Agent Audit\n",
            "Generated audit, not a claim of visual annotation correctness. Only human gt/gt.txt is used. "
            "Official train/test groups are retained. No validation split is invented. "
            "Frame IDs map exactly to numeric filenames. By default numeric x/y are preserved, "
            "consistent with sampled official COCO JSON, not shifted merely because GT is MOT-style. "
            "Upstream pixel origin is not explicitly documented; inspect xy_translation in audit.json "
            "and visually check tiny/boundary boxes before training. "
            "Any frame with inactive or unsupported-class GT, or no valid positive GT, is excluded. "
            "Tiny and visibility-zero positive boxes are retained. Sequence 18 is quarantined, "
            "with no exported frames or labels.\n",
            "Dataset license is unspecified. Community issue #8 reports V1 sequence-18 alignment problems. "
            "Test JSON has no GT; the JSON category name pedestrian is not used. "
            "Sequence-ID separation does not guarantee scene or near-duplicate independence.\n"]),
        dict(cell_type="code", id="verify", metadata={}, execution_count=None, outputs=[], source=code.splitlines(True))])
    path.write_text(json.dumps(notebook, indent=2) + "\n")


def prepare(root, output, xy_origin=0):
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if root == output or root in output.parents:
        raise ValueError("Output must be outside the raw dataset")
    if xy_origin not in (0, 1):
        raise ValueError("xy_origin must be 0 or 1")
    sequences = discover_sequences(root)
    records, labels, sequence_audits = [], [], []
    hashes, realpaths = {}, {}
    counts, skipped, dimensions = Counter(), Counter(), Counter()
    duplicate_same_split = 0
    for split, sequence in sequences:
        gt_path = (sequence / "gt" / "gt.txt").resolve()
        if sequence.name.casefold() in QUARANTINED_GROUPS:
            frame_count = len(image_frames(sequence))
            skipped["quarantined_sequence_frames"] += frame_count
            sequence_audits.append(dict(split=split, group=sequence.name, gt=str(gt_path),
                                       gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                                       raw_images=frame_count, quarantined=True,
                                       reason=QUARANTINED_GROUPS[sequence.name.casefold()],
                                       exported={"images": 0, "boxes": 0}))
            continue
        gt, gt_hash = read_gt(gt_path)
        frames = image_frames(sequence)
        missing = sorted(set(gt) - set(frames))
        if missing:
            raise ValueError(f"GT frames have no matching image in {sequence}: {missing[:10]}")
        sequence_counts, sequence_skips = Counter(), Counter()
        for frame, source_image in sorted(frames.items()):
            # Read actual image headers and verify data, not seqinfo.ini or inferred frame offsets.
            with Image.open(source_image) as image:
                width, height = image.size
                image.load()
            if min(width, height) <= 0:
                raise ValueError(f"Invalid image dimensions: {source_image}")
            dimensions[f"{width}x{height}"] += 1
            rows = gt.get(frame, [])
            reason = None
            if not rows:
                reason = "no_gt"
            elif any(row[4] == 0 for row in rows):
                reason = "inactive_gt_frame"
            elif any(row[5] != 1 for row in rows):
                reason = "unsupported_class_frame"
            boxes = []
            if reason is None:
                boxes = [box for row in rows
                         if (box := mot_box_to_yolo(*row[:4], width, height, xy_origin)) is not None]
                sequence_counts["invalid_or_outside_boxes"] += len(rows) - len(boxes)
                if not boxes:
                    reason = "no_valid_positive"
            if reason:
                skipped[reason] += 1
                sequence_skips[reason] += 1
                continue
            digest = hashlib.sha256(source_image.read_bytes()).hexdigest()
            repeated = digest in hashes
            for key, seen in ((source_image, realpaths), (digest, hashes)):
                if key in seen and seen[key] != split:
                    raise ValueError(f"Cross-split duplicate image: {source_image}")
                seen[key] = split
            if repeated:
                duplicate_same_split += 1
            image_path = output / "images" / split / sequence.name / source_image.name
            label_path = output / "labels" / split / sequence.name / (source_image.stem + ".txt")
            records.append(dict(dataset="uavswarm", split=split, group=sequence.name,
                                image=str(image_path), label=str(label_path), boxes=len(boxes),
                                source=str(gt_path), source_image=str(source_image),
                                source_sha256=digest, frame=frame, width=width, height=height))
            labels.append("".join("0 " + " ".join(format(value, ".12g") for value in box) + "\n"
                                  for box in boxes))
            counts[split] += 1
            sequence_counts["images"] += 1
            sequence_counts["boxes"] += len(boxes)
        sequence_audits.append(dict(split=split, group=sequence.name, gt=str(gt_path),
                                   gt_sha256=gt_hash, raw_images=len(frames),
                                   gt_rows=sum(map(len, gt.values())), exported=dict(sequence_counts),
                                   skipped=dict(sequence_skips)))
    if not records:
        raise ValueError("No valid positive human-GT frames remain")
    audit = dict(schema="anti_uav.uavswarm.v1", source=OFFICIAL_SOURCE, raw_root=str(root),
                 output=str(output), annotation_source="gt/gt.txt only; never det or COCO JSON",
                 mot_frame_origin=1, xy_translation=-xy_origin,
                 coordinate_policy="Preserve official numeric x/y unless explicitly overridden; pixel origin unspecified upstream",
                 coordinate_reference=OFFICIAL_SOURCE + "/blob/master/annotations/train.json",
                 coordinate_visual_qa="not performed; tiny and boundary boxes require visual review",
                 visual_qa_complete=False, training_approved=False,
                 split_counts=dict(counts),
                 dimensions=dict(dimensions), skipped=dict(skipped), sequences=sequence_audits,
                 cross_split_groups=[], cross_split_exact_file_duplicates=[],
                 same_split_exact_file_duplicate_images=duplicate_same_split,
                 quarantined_groups=[row for row in sequence_audits if row.get("quarantined")],
                 license="Unspecified; no commercial clearance inferred", research_only=True,
                 readiness="not approved for training: visual coordinate/annotation QA still required",
                 warnings=["V1 sequence 18: quarantined by default, no frames exported",
                           "Sequence groups disjoint; scene/near-duplicate independence not established",
                           "test.json has no annotations; JSON category pedestrian is a placeholder"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        with (staging / "manifest.jsonl").open("w") as manifest:
            for row, label in zip(records, labels):
                image_path = staging / Path(row["image"]).relative_to(output)
                label_path = staging / Path(row["label"]).relative_to(output)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.symlink_to(row["source_image"])
                label_path.write_text(label)
                manifest.write(json.dumps(row, allow_nan=False) + "\n")
        for split in ("train", "test"):
            (staging / f"{split}.txt").write_text("".join(
                row["image"] + "\n" for row in records if row["split"] == split))
        (staging / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
        write_notebook(staging / "audit.ipynb", output)
        # An existing destination is never intentionally merged or overwritten.
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    # Consumers must not treat an interrupted or merely staged preparation as complete.
    summary = dict(audit, complete=True, manifest=str(output / "manifest.jsonl"),
                   manifest_sha256=hashlib.sha256((output / "manifest.jsonl").read_bytes()).hexdigest())
    pending_summary = output / ".summary.json.tmp"
    pending_summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pending_summary.replace(output / "summary.json")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--source", dest="root", type=Path, required=True,
                        help="Extracted official repository root with train/ and test/")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--xy-origin", type=int, choices=(0, 1), default=0,
                        help="0 preserves official numeric x/y (default); 1 explicitly shifts x/y by -1")
    args = parser.parse_args()
    audit = prepare(args.root, args.output, args.xy_origin)
    print(json.dumps({"output": audit["output"], "split_counts": audit["split_counts"],
                      "skipped": audit["skipped"], "readiness": audit["readiness"],
                      "audit": str(Path(audit["output"]) / "audit.json")}, indent=2))


if __name__ == "__main__":
    main()
