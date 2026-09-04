#!/usr/bin/env python3
"""Build a leakage-safe rehearsal split from finalized gray-video annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--include-video", action="append", required=True, help="Video filename; repeat per video")
    parser.add_argument("--eval-video", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-stride", type=int, default=3)
    parser.add_argument("--negative-stride", type=int, default=20)
    parser.add_argument("--positive-rehearsal-fraction", type=float, default=0.10)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as error:
        raise ValueError(f"Image path has no images component: {image_path}") from error
    return Path(*parts).with_suffix(".txt")


def resolve_yaml_path(value: str, yaml_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (yaml_path.parent / path).resolve()


def load_annotations(path: Path, include_videos: set[str]) -> dict[str, dict[int, dict]]:
    records: dict[str, dict[int, dict]] = {video: {} for video in include_videos}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        video = record.get("video")
        if video not in records:
            continue
        frame = int(record["frameIndex"])
        if frame in records[video]:
            raise ValueError(f"Duplicate annotation for {video} frame {frame} at line {line_number}")
        visible = bool(record.get("visible"))
        bbox = record.get("bboxXyxy")
        if visible != (bbox is not None):
            raise ValueError(f"Visible/bbox mismatch for {video} frame {frame}")
        if visible:
            if record.get("status") != "confirmed" or len(bbox) != 4:
                raise ValueError(f"Visible annotation is not finalized for {video} frame {frame}")
            width, height = int(record["frameWidth"]), int(record["frameHeight"])
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid bbox for {video} frame {frame}: {bbox}")
        records[video][frame] = record
    missing = include_videos.difference(video for video, frames in records.items() if frames)
    if missing:
        raise ValueError(f"No annotations found for: {sorted(missing)}")
    return records


def yolo_label(record: dict) -> str:
    x1, y1, x2, y2 = (float(value) for value in record["bboxXyxy"])
    width, height = float(record["frameWidth"]), float(record["frameHeight"])
    center_x = (x1 + x2) * 0.5 / width
    center_y = (y1 + y2) * 0.5 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"


def extract_video_samples(
    video_path: Path,
    annotations: dict[int, dict],
    output: Path,
    positive_stride: int,
    negative_stride: int,
    jpeg_quality: int,
) -> tuple[list[Path], list[Path], list[dict], dict]:
    frame_indices = sorted(annotations)
    if frame_indices != list(range(len(frame_indices))):
        raise ValueError(f"Annotations are not complete from frame 0 for {video_path.name}")

    selected = {
        frame: record
        for frame, record in annotations.items()
        if frame % (positive_stride if record["visible"] else negative_stride) == 0
    }
    image_dir = output / "images" / "manual_gray" / video_path.stem
    label_dir = output / "labels" / "manual_gray" / video_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    positive_paths: list[Path] = []
    negative_paths: list[Path] = []
    selected_records: list[dict] = []
    frame = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        record = annotations.get(frame)
        if record is None:
            raise RuntimeError(f"Decoded unannotated frame {frame} from {video_path.name}")
        expected_shape = (int(record["frameHeight"]), int(record["frameWidth"]))
        if image.shape[:2] != expected_shape:
            raise RuntimeError(
                f"Frame shape mismatch for {video_path.name} frame {frame}: {image.shape[:2]} != {expected_shape}"
            )
        if frame in selected:
            image_path = image_dir / f"{frame:06d}.jpg"
            output_label = label_dir / f"{frame:06d}.txt"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                raise RuntimeError(f"Unable to write image: {image_path}")
            output_label.write_text(yolo_label(record) if record["visible"] else "")
            (positive_paths if record["visible"] else negative_paths).append(image_path)
            selected_records.append(
                {
                    "video": video_path.name,
                    "frame": frame,
                    "visible": bool(record["visible"]),
                    "bbox_xyxy": record.get("bboxXyxy"),
                    "annotation_source": record.get("source"),
                    "image": str(image_path),
                }
            )
        frame += 1
    capture.release()
    if frame != len(annotations):
        raise RuntimeError(f"Decoded {frame} frames from {video_path.name}, expected {len(annotations)}")
    if len(selected_records) != len(selected):
        raise RuntimeError(f"Extracted {len(selected_records)} samples, expected {len(selected)}")
    stats = {
        "frames": frame,
        "visible_frames": sum(bool(record["visible"]) for record in annotations.values()),
        "absent_frames": sum(not bool(record["visible"]) for record in annotations.values()),
        "sampled_positive_frames": len(positive_paths),
        "sampled_negative_frames": len(negative_paths),
    }
    return positive_paths, negative_paths, selected_records, stats


def balanced_schedule(paths_by_video: dict[str, list[Path]], quota: int, rng: random.Random) -> list[Path]:
    if quota <= 0:
        return []
    if not paths_by_video or any(not paths for paths in paths_by_video.values()):
        raise ValueError("Every included video must contribute samples to a balanced schedule")
    videos = sorted(paths_by_video)
    shuffled = {video: list(paths) for video, paths in paths_by_video.items()}
    for paths in shuffled.values():
        rng.shuffle(paths)
    offsets = Counter()
    output: list[Path] = []
    for index in range(quota):
        video = videos[index % len(videos)]
        paths = shuffled[video]
        output.append(paths[offsets[video] % len(paths)])
        offsets[video] += 1
    rng.shuffle(output)
    return output


def replace_duplicate_class_slots(
    source_paths: list[Path],
    manual_positive_by_video: dict[str, list[Path]],
    manual_negative_by_video: dict[str, list[Path]],
    positive_fraction: float,
    seed: int,
) -> tuple[list[Path], dict]:
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("Positive rehearsal fraction must be in (0, 1)")

    manual_positive_paths = [path for paths in manual_positive_by_video.values() for path in paths]
    manual_negative_paths = [path for paths in manual_negative_by_video.values() for path in paths]
    all_paths = set(source_paths) | set(manual_positive_paths) | set(manual_negative_paths)
    positive_by_path = {path: bool(label_path(path).read_text().strip()) for path in all_paths}
    if not all(positive_by_path[path] for path in manual_positive_paths):
        raise ValueError("Every manual positive image must have a non-empty label")
    if any(positive_by_path[path] for path in manual_negative_paths):
        raise ValueError("Every manual negative image must have an empty label")

    source_flags = [positive_by_path[path] for path in source_paths]
    source_positive_count = sum(source_flags)
    source_negative_count = len(source_paths) - source_positive_count
    if source_positive_count == 0 or source_negative_count == 0:
        raise ValueError("Source schedule must contain both positive and negative samples")
    occurrences: Counter[Path] = Counter()
    replaceable_positive: list[int] = []
    for index, (path, positive) in enumerate(zip(source_paths, source_flags)):
        occurrences[path] += 1
        if positive and occurrences[path] > 1:
            replaceable_positive.append(index)

    requested_positive = max(round(source_positive_count * positive_fraction), len(manual_positive_paths))
    if requested_positive > len(replaceable_positive):
        raise ValueError(
            "Not enough duplicate positive slots to preserve old unique coverage: "
            f"{requested_positive}/{len(replaceable_positive)}"
        )

    rng = random.Random(seed)
    rng.shuffle(replaceable_positive)
    replacement_schedule = balanced_schedule(manual_positive_by_video, requested_positive, rng)
    output = list(source_paths)
    for index, replacement in zip(replaceable_positive, replacement_schedule):
        output[index] = replacement

    # Source negatives are all unique in the neg15 schedule. Append new negatives and enough
    # manual positives to retain the exact source class ratio rather than deleting old scenes.
    appended_positive_count = round(len(manual_negative_paths) * source_positive_count / source_negative_count)
    appended_positive = balanced_schedule(manual_positive_by_video, appended_positive_count, rng)
    appended = appended_positive + manual_negative_paths
    rng.shuffle(appended)
    output.extend(appended)

    output_flags = [positive_by_path[path] for path in output]
    if output_flags[: len(source_flags)] != source_flags:
        raise RuntimeError("Source positive/negative class sequence changed")
    if not set(source_paths).issubset(output):
        raise RuntimeError("Old unique source coverage changed")
    if not set(manual_positive_paths + manual_negative_paths).issubset(output):
        raise RuntimeError("Manual sampled-image coverage is incomplete")
    return output, {
        "source_samples": len(source_paths),
        "positive_samples": source_positive_count,
        "negative_samples": source_negative_count,
        "negative_fraction": source_negative_count / len(source_paths),
        "positive_rehearsal_slots": requested_positive,
        "appended_positive_slots": appended_positive_count,
        "appended_negative_slots": len(manual_negative_paths),
        "output_samples": len(output),
        "output_positive_samples": sum(output_flags),
        "output_negative_samples": len(output) - sum(output_flags),
        "output_negative_fraction": (len(output) - sum(output_flags)) / len(output),
        "manual_unique_positive_images": len(manual_positive_paths),
        "manual_unique_negative_images": len(manual_negative_paths),
        "source_class_sequence_unchanged": True,
        "old_unique_sample_coverage_unchanged": True,
        "manual_sample_coverage_complete": True,
    }


def main() -> None:
    args = parse_args()
    if args.positive_stride < 1 or args.negative_stride < 1:
        raise ValueError("Sampling strides must be positive")
    include_videos = set(args.include_video)
    if len(include_videos) != len(args.include_video):
        raise ValueError("--include-video entries must be unique")

    eval_hash = sha256_file(args.eval_video)
    annotations = load_annotations(args.annotations, include_videos)
    positive_by_video: dict[str, list[Path]] = {}
    negative_by_video: dict[str, list[Path]] = {}
    selected_records: list[dict] = []
    video_manifest: dict[str, dict] = {}
    for video_name in sorted(include_videos):
        video_path = args.videos_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        video_hash = sha256_file(video_path)
        if video_path.resolve() == args.eval_video.resolve() or video_hash == eval_hash:
            raise ValueError(f"Training video matches eval content: {video_path}")
        positives, negatives, records, stats = extract_video_samples(
            video_path,
            annotations[video_name],
            args.output,
            args.positive_stride,
            args.negative_stride,
            args.jpeg_quality,
        )
        positive_by_video[video_name] = positives
        negative_by_video[video_name] = negatives
        selected_records.extend(records)
        video_manifest[video_name] = {"path": str(video_path), "sha256": video_hash, **stats}

    source_data = yaml.safe_load(args.source_data.read_text())
    source_train = resolve_yaml_path(source_data["train"], args.source_data)
    source_paths = [Path(line.strip()) for line in source_train.read_text().splitlines() if line.strip()]
    output_paths, rehearsal = replace_duplicate_class_slots(
        source_paths,
        positive_by_video,
        negative_by_video,
        args.positive_rehearsal_fraction,
        args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    train_list = args.output / "train_manual_gray_rehearsal.txt"
    train_list.write_text("".join(f"{path}\n" for path in output_paths))
    output_data = dict(source_data)
    output_data["path"] = str(args.output)
    output_data["train"] = str(train_list)
    data_yaml = args.output / "train_rgb_monitor.yaml"
    data_yaml.write_text(yaml.safe_dump(output_data, sort_keys=False))

    records_path = args.output / "sampled_manual_labels.jsonl"
    records_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in selected_records))
    manifest = {
        "schema_version": "anti_uav.manual_gray_video_rehearsal.v1",
        "policy": {
            "eval_isolation": "Every included video path and SHA256 must differ from the eval video",
            "annotations": "Final visible boxes and explicit absent frames only",
            "sampling": "Temporal thinning by class, balanced equally across included videos",
            "rehearsal": "Replace duplicate positive slots; append balanced positives with new negatives to preserve the class ratio and all old samples",
        },
        "annotations": str(args.annotations),
        "annotations_sha256": sha256_file(args.annotations),
        "eval_video": str(args.eval_video),
        "eval_video_sha256": eval_hash,
        "source_data": str(args.source_data),
        "source_train": str(source_train),
        "source_train_sha256": sha256_file(source_train),
        "positive_stride": args.positive_stride,
        "negative_stride": args.negative_stride,
        "positive_rehearsal_fraction_requested": args.positive_rehearsal_fraction,
        "videos": video_manifest,
        "rehearsal": rehearsal,
        "train_list": str(train_list),
        "train_list_sha256": sha256_file(train_list),
        "data_yaml": str(data_yaml),
        "sampled_annotation_records": str(records_path),
        "seed": args.seed,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
