#!/usr/bin/env python3
"""Freeze new approved tasks after full annotation/video and holdout checks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.build_approved_gray_rehearsal import checked_path, parse_label, validate_frame_sets
from scripts.anti_uav.build_manual_gray_video_rehearsal import sha256_file


def audit_new(task, video_root, manifest):
    included, negatives = validate_frame_sets(manifest)
    video = manifest["video"]
    path = checked_path(video_root, video["name"])
    if sha256_file(path) != video["sha256"]:
        raise ValueError(f"Video hash mismatch: {path}")
    for record in manifest["files"]:
        file = checked_path(task, record["path"])
        if file.stat().st_size != record["size"] or sha256_file(file) != record["sha256"]:
            raise ValueError(f"Annotation checksum mismatch: {file}")
    coco = json.loads((task / "coco/annotations.json").read_text())
    images = {row["id"]: row for row in coco["images"]}
    if len(images) != len(coco["images"]) or {i["frame_index"] for i in images.values()} != included or len(images) != len(included):
        raise ValueError(f"COCO frame set mismatch: {task}")
    boxes = {i: [] for i in images}
    for row in coco["annotations"]:
        if row["category_id"] != 0:
            raise ValueError("Unexpected COCO category")
        boxes[row["image_id"]].append(row["bbox"])
    if {int(p.stem) for p in (task / "yolo/labels").glob("*.txt")} != included:
        raise ValueError(f"YOLO frame set mismatch: {task}")
    for identity, image in images.items():
        width, height = image["width"], image["height"]
        if (width, height) != (video["frameWidth"], video["frameHeight"]):
            raise ValueError("COCO dimensions differ from manifest")
        label = parse_label((task / "yolo/labels" / f"{image['frame_index']:09d}.txt").read_text())
        if bool(label) != (image["frame_index"] not in negatives):
            raise ValueError("Positive/negative labels disagree")
        actual = sorted([[(cx-bw/2)*width, (cy-bh/2)*height, bw*width, bh*height]
                         for _, cx, cy, bw, bh in label])
        expected = sorted(boxes[identity])
        if len(actual) != len(expected) or any(abs(x-y) > .1 for a, b in zip(actual, expected) for x, y in zip(a, b)):
            raise ValueError(f"COCO/YOLO geometry mismatch: {task}, frame {image['frame_index']}")
    return dict(positive_frames=len(included-negatives), negative_frames=len(negatives),
                uncertain_frames=len(manifest["frames"].get("excludedUncertainFrameIndices", [])),
                unreviewed_frames=len(manifest["frames"].get("excludedUnreviewedFrameIndices", [])),
                video_and_annotation_hashes_valid=True, coco_yolo_geometry_valid=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--approved-root", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    base = json.loads((a.dataset / "manifest.json").read_text())
    allowed = set(base["train_video_hashes"])
    heldout = {base["test_sha256"], base["validation_sha256"]}
    if heldout & allowed:
        raise ValueError("Baseline/heldout overlap")
    rows, seen = [], set()
    for path in sorted(a.approved_root.glob("*/manifest.json")):
        before = sha256_file(path)
        m = json.loads(path.read_text())
        h = m["video"]["sha256"]
        if h in seen:
            raise ValueError(f"Multiple approved tasks for one video: {h}")
        seen.add(h)
        status = "heldout" if h in heldout else "existing_training" if h in allowed else "new_training_candidate"
        row = dict(task=str(path.parent), name=m["video"]["name"], sha256=h,
                   manifest_sha256=before, status=status)
        if status == "new_training_candidate":
            row.update(audit_new(path.parent, a.video_root, m))
            if row["positive_frames"] == 0:
                row["status"] = "negative_only_separate_review"
        if sha256_file(path) != before:
            raise ValueError(f"Manifest changed during audit: {path}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    new = [r for r in rows if r["status"] == "new_training_candidate"]
    report = dict(schema="approved_expansion_audit.v1", checked_at=datetime.now(timezone.utc).isoformat(),
                  source_dataset=str(a.dataset), source_manifest_sha256=sha256_file(a.dataset / "manifest.json"),
                  rows=rows, new_rows=new, new_videos=len(new), counts=dict(Counter(r["status"] for r in rows)),
                  new_positive_frames=sum(r["positive_frames"] for r in new),
                  new_negative_frames=sum(r["negative_frames"] for r in new),
                  expanded_training_videos=len(allowed)+len(new), heldout_sha256=sorted(heldout),
                  limitation="Exact-video hash isolation; reencoded/overlapping clips and same-session correlation need separate review.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows", "new_rows")}), flush=True)


if __name__ == "__main__":
    main()
