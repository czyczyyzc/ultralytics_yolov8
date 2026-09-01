#!/usr/bin/env python3
"""Build a leakage-safe gray-video rehearsal split from detector pseudo labels."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Detection:
    frame: int
    box: tuple[float, float, float, float]
    confidence: float
    source: str = "detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-video", type=Path, required=True)
    parser.add_argument("--eval-video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[544, 960], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--detect-conf", type=float, default=0.03)
    parser.add_argument("--seed-conf", type=float, default=0.25)
    parser.add_argument("--support-radius", type=int, default=30)
    parser.add_argument("--max-center-distance", type=float, default=0.08)
    parser.add_argument("--bridge-max-gap", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--rehearsal-fraction", type=float, default=0.05)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=20260901)
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


def center_distance(first: Detection, second: Detection) -> float:
    first_center = ((first.box[0] + first.box[2]) * 0.5, (first.box[1] + first.box[3]) * 0.5)
    second_center = ((second.box[0] + second.box[2]) * 0.5, (second.box[1] + second.box[3]) * 0.5)
    return float(np.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))


def size_compatible(first: Detection, second: Detection, max_ratio: float = 4.0) -> bool:
    first_size = (max(first.box[2] - first.box[0], 1e-9), max(first.box[3] - first.box[1], 1e-9))
    second_size = (max(second.box[2] - second.box[0], 1e-9), max(second.box[3] - second.box[1], 1e-9))
    return all(1.0 / max_ratio <= a / b <= max_ratio for a, b in zip(first_size, second_size))


def select_supported_detections(
    candidates: list[Detection],
    seed_conf: float,
    support_radius: int,
    max_center_distance: float,
) -> list[Detection]:
    """Keep strong seeds and weak detections spatially supported by a nearby seed."""
    seeds = [candidate for candidate in candidates if candidate.confidence >= seed_conf]
    if not seeds:
        raise RuntimeError("No pseudo-label seed detections met --seed-conf")
    seed_frames = [seed.frame for seed in seeds]
    selected: list[Detection] = []
    for candidate in candidates:
        if candidate.confidence >= seed_conf:
            selected.append(candidate)
            continue
        position = bisect.bisect_left(seed_frames, candidate.frame)
        neighbors = seeds[max(position - 1, 0) : min(position + 1, len(seeds) - 1) + 1]
        if any(
            abs(seed.frame - candidate.frame) <= support_radius
            and center_distance(seed, candidate) <= max_center_distance
            and size_compatible(seed, candidate)
            for seed in neighbors
        ):
            selected.append(candidate)
    return selected


def bridge_short_gaps(detections: list[Detection], max_gap: int, max_center_distance: float) -> list[Detection]:
    """Linearly bridge only short, spatially compatible gaps at the source frame rate."""
    by_frame = {detection.frame: detection for detection in detections}
    ordered = sorted(by_frame.values(), key=lambda detection: detection.frame)
    for first, second in zip(ordered, ordered[1:]):
        gap = second.frame - first.frame
        if gap <= 1 or gap > max_gap + 1:
            continue
        if center_distance(first, second) > max_center_distance or not size_compatible(first, second):
            continue
        for offset in range(1, gap):
            ratio = offset / gap
            box = tuple(a + (b - a) * ratio for a, b in zip(first.box, second.box))
            by_frame[first.frame + offset] = Detection(
                frame=first.frame + offset,
                box=box,
                confidence=min(first.confidence, second.confidence),
                source="interpolated",
            )
    return sorted(by_frame.values(), key=lambda detection: detection.frame)


def infer_candidates(
    video: Path, model_path: Path, imgsz: list[int], conf: float, device: str
) -> tuple[list[Detection], dict]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    candidates: list[Detection] = []
    frames = 0
    detection_frames = 0
    for frame, result in enumerate(
        model.predict(
            source=str(video),
            imgsz=imgsz,
            conf=conf,
            iou=0.45,
            max_det=20,
            device=device,
            stream=True,
            verbose=False,
        )
    ):
        frames += 1
        if result.boxes is None or not len(result.boxes):
            continue
        detection_frames += 1
        confidences = result.boxes.conf.cpu().numpy()
        best = int(confidences.argmax())
        height, width = result.orig_shape
        box = result.boxes.xyxy[best].cpu().numpy().astype(float)
        box[[0, 2]] /= width
        box[[1, 3]] /= height
        candidates.append(Detection(frame, tuple(float(value) for value in box), float(confidences[best])))
    return candidates, {"frames": frames, "detection_frames": detection_frames}


def yolo_label(detection: Detection) -> str:
    x1, y1, x2, y2 = detection.box
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    width = x2 - x1
    height = y2 - y1
    return f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n"


def extract_selected_frames(
    video: Path,
    detections: list[Detection],
    image_dir: Path,
    label_dir: Path,
    jpeg_quality: int,
) -> list[Path]:
    selected = {detection.frame: detection for detection in detections}
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")
    images: list[Path] = []
    frame = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        detection = selected.get(frame)
        if detection is not None:
            image_path = image_dir / f"{frame:06d}.jpg"
            label = label_dir / f"{frame:06d}.txt"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                raise RuntimeError(f"Unable to write image: {image_path}")
            label.write_text(yolo_label(detection))
            images.append(image_path)
        frame += 1
    capture.release()
    if len(images) != len(selected):
        raise RuntimeError(f"Extracted {len(images)} selected frames, expected {len(selected)}")
    return images


def replace_duplicate_positive_slots(
    source_paths: list[Path], pseudo_paths: list[Path], rehearsal_fraction: float, seed: int
) -> tuple[list[Path], dict]:
    if not 0.0 < rehearsal_fraction < 1.0:
        raise ValueError("--rehearsal-fraction must be in (0, 1)")
    if not pseudo_paths:
        raise ValueError("No pseudo-labeled training images")

    # Training schedules contain many repeated paths; cache label presence per unique image.
    positive_by_path = {
        path: bool(label_path(path).read_text().strip()) for path in set(source_paths) | set(pseudo_paths)
    }
    if not all(positive_by_path[path] for path in pseudo_paths):
        raise ValueError("Every pseudo-labeled image must have a non-empty label")
    positive_flags = [positive_by_path[path] for path in source_paths]
    positive_count = sum(positive_flags)
    source_unique_positives = {path for path, positive in zip(source_paths, positive_flags) if positive}
    source_negatives = [path for path, positive in zip(source_paths, positive_flags) if not positive]
    occurrences: Counter[Path] = Counter()
    replaceable: list[int] = []
    for index, (path, positive) in enumerate(zip(source_paths, positive_flags)):
        if not positive:
            continue
        occurrences[path] += 1
        if occurrences[path] > 1:
            replaceable.append(index)

    quota = min(round(positive_count * rehearsal_fraction), len(replaceable))
    rng = random.Random(seed)
    rng.shuffle(replaceable)
    schedule = pseudo_paths.copy()
    rng.shuffle(schedule)
    schedule = [schedule[index % len(schedule)] for index in range(quota)]
    output = list(source_paths)
    for index, path in zip(replaceable[:quota], schedule):
        output[index] = path

    output_flags = [positive_by_path[path] for path in output]
    output_negatives = [path for path, positive in zip(output, output_flags) if not positive]
    output_old_unique = {path for path, positive in zip(output, output_flags) if positive} & source_unique_positives
    if output_negatives != source_negatives:
        raise RuntimeError("Negative sample identity or order changed")
    if output_old_unique != source_unique_positives:
        raise RuntimeError("Old unique positive coverage changed")
    return output, {
        "source_samples": len(source_paths),
        "positive_samples": positive_count,
        "negative_samples": len(source_negatives),
        "negative_fraction": len(source_negatives) / len(source_paths),
        "replaceable_duplicate_positive_slots": len(replaceable),
        "pseudo_rehearsal_slots": quota,
        "pseudo_unique_images": len(pseudo_paths),
        "negative_samples_unchanged": True,
        "old_unique_positive_coverage_unchanged": True,
    }


def main() -> None:
    args = parse_args()
    if args.train_video.resolve() == args.eval_video.resolve():
        raise ValueError("Train and eval videos must differ")
    if sha256_file(args.train_video) == sha256_file(args.eval_video):
        raise ValueError("Train and eval video content must differ")
    if not 0.0 < args.detect_conf < args.seed_conf <= 1.0:
        raise ValueError("Require 0 < --detect-conf < --seed-conf <= 1")
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be positive")

    candidates, inference = infer_candidates(args.train_video, args.model, args.imgsz, args.detect_conf, args.device)
    supported = select_supported_detections(
        candidates, args.seed_conf, args.support_radius, args.max_center_distance
    )
    bridged = bridge_short_gaps(supported, args.bridge_max_gap, args.max_center_distance)
    sampled = [detection for detection in bridged if detection.frame % args.sample_stride == 0]

    video_id = args.train_video.stem
    pseudo_paths = extract_selected_frames(
        args.train_video,
        sampled,
        args.output / "images" / "gray" / video_id,
        args.output / "labels" / "gray" / video_id,
        args.jpeg_quality,
    )

    source_data = yaml.safe_load(args.source_data.read_text())
    source_train = resolve_yaml_path(source_data["train"], args.source_data)
    source_paths = [Path(line.strip()) for line in source_train.read_text().splitlines() if line.strip()]
    output_paths, rehearsal = replace_duplicate_positive_slots(
        source_paths, pseudo_paths, args.rehearsal_fraction, args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    train_list = args.output / "train_pseudo_gray_rehearsal.txt"
    train_list.write_text("".join(f"{path}\n" for path in output_paths))
    output_data = dict(source_data)
    output_data["path"] = str(args.output)
    output_data["train"] = str(train_list)
    data_yaml = args.output / "train_rgb_monitor.yaml"
    data_yaml.write_text(yaml.safe_dump(output_data, sort_keys=False))

    records = args.output / "pseudo_labels.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "frame": detection.frame,
                    "box_xyxy_normalized": detection.box,
                    "confidence": detection.confidence,
                    "source": detection.source,
                    "image": str(path),
                },
                sort_keys=True,
            )
            + "\n"
            for detection, path in zip(sampled, pseudo_paths)
        )
    )
    manifest = {
        "schema_version": "anti_uav.pseudo_labeled_gray_video_split.v1",
        "policy": {
            "eval_video_isolation": "eval video is recorded only and never decoded into training data",
            "unlabeled_frames": "excluded; never treated as negatives",
            "pseudo_labels": "single highest-confidence detection, seed-supported, short-gap interpolation",
            "rehearsal": "replace duplicate positive slots only; preserve all negatives and old unique positives",
        },
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "train_video": str(args.train_video),
        "train_video_sha256": sha256_file(args.train_video),
        "eval_video": str(args.eval_video),
        "eval_video_sha256": sha256_file(args.eval_video),
        "source_data": str(args.source_data),
        "source_train": str(source_train),
        "source_train_sha256": sha256_file(source_train),
        "input_height_width": args.imgsz,
        "detect_confidence": args.detect_conf,
        "seed_confidence": args.seed_conf,
        "support_radius_frames": args.support_radius,
        "max_center_distance_normalized": args.max_center_distance,
        "bridge_max_gap_frames": args.bridge_max_gap,
        "sample_stride": args.sample_stride,
        "inference": inference,
        "candidate_detection_frames": len(candidates),
        "supported_detection_frames": len(supported),
        "bridged_frames": len(bridged),
        "sampled_pseudo_positive_frames": len(sampled),
        "interpolated_sampled_frames": sum(detection.source == "interpolated" for detection in sampled),
        "rehearsal_fraction_requested": args.rehearsal_fraction,
        "rehearsal": rehearsal,
        "train_list": str(train_list),
        "data_yaml": str(data_yaml),
        "pseudo_label_records": str(records),
        "seed": args.seed,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
