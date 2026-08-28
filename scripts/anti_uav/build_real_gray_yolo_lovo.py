#!/usr/bin/env python3
"""Build positive-only real-gray YOLO LOVO folds mixed 1:1 with Anti-UAV300 RGB."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--rgb-train-list", type=Path, required=True)
    parser.add_argument("--rgb-val-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--extract-workers", type=int, default=4)
    parser.add_argument("--jpeg-qscale", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_paths(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as error:
        raise ValueError(f"Image path has no images component: {image_path}") from error
    return Path(*parts).with_suffix(".txt")


def probe_video(path: Path) -> dict[str, int | float | str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    stream = result["streams"][0]
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["avg_frame_rate"],
        "frames": int(stream["nb_frames"]),
        "duration_sec": float(result["format"]["duration"]),
    }


def extract_video(video: Path, image_dir: Path, expected_frames: int, jpeg_qscale: int) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    existing = list(image_dir.glob("*.jpg"))
    if len(existing) == expected_frames:
        return
    if existing:
        raise RuntimeError(f"Incomplete extraction already exists in {image_dir}: {len(existing)} files")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-q:v",
        str(jpeg_qscale),
        "-start_number",
        "0",
        str(image_dir / "%06d.jpg"),
    ]
    subprocess.run(command, check=True)
    actual_frames = sum(1 for _ in image_dir.glob("*.jpg"))
    if actual_frames != expected_frames:
        raise RuntimeError(f"{video.name}: extracted {actual_frames}, expected {expected_frames}")


def yolo_label(box: list[float], width: int, height: int) -> str:
    x, y, box_width, box_height = map(float, box)
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + box_width, 0.0), float(width))
    y2 = min(max(y + box_height, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid clipped box: {box}")
    center_x = (x1 + x2) * 0.5 / width
    center_y = (y1 + y2) * 0.5 / height
    normalized_width = (x2 - x1) / width
    normalized_height = (y2 - y1) / height
    return f"0 {center_x:.8f} {center_y:.8f} {normalized_width:.8f} {normalized_height:.8f}\n"


def write_lines(path: Path, values: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values))


def write_yaml(path: Path, train: Path, val: Path) -> None:
    path.write_text(
        f"path: {path.parent}\n"
        f"train: {train}\n"
        f"val: {val}\n\n"
        "names:\n"
        "  0: drone\n"
    )


def main() -> None:
    args = parse_args()
    videos = sorted(args.videos.glob("Video*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No videos found in {args.videos}")
    stems = [video.stem for video in videos]
    annotations: dict[str, dict] = {}
    metadata: dict[str, dict] = {}
    for video in videos:
        annotation_path = args.annotations / f"{video.stem}.visible.json"
        annotation = json.loads(annotation_path.read_text())
        info = probe_video(video)
        if len(annotation["exist"]) != info["frames"] or len(annotation["gt_rect"]) != info["frames"]:
            raise RuntimeError(f"Frame/annotation mismatch for {video.stem}")
        annotations[video.stem] = annotation
        metadata[video.stem] = {
            **info,
            "video": str(video),
            "video_sha256": sha256(video),
            "annotation": str(annotation_path),
            "annotation_sha256": sha256(annotation_path),
            "visible_frames": sum(bool(value) for value in annotation["exist"]),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.extract_workers) as executor:
        futures = []
        for video in videos:
            futures.append(
                executor.submit(
                    extract_video,
                    video,
                    args.output / "images" / "gray" / video.stem,
                    int(metadata[video.stem]["frames"]),
                    args.jpeg_qscale,
                )
            )
        for future in futures:
            future.result()

    images_by_video: dict[str, list[Path]] = {}
    positives_by_video: dict[str, list[Path]] = {}
    for stem in stems:
        info = metadata[stem]
        annotation = annotations[stem]
        image_dir = args.output / "images" / "gray" / stem
        label_dir = args.output / "labels" / "gray" / stem
        label_dir.mkdir(parents=True, exist_ok=True)
        images: list[Path] = []
        positives: list[Path] = []
        for frame_index, (present, box) in enumerate(zip(annotation["exist"], annotation["gt_rect"])):
            image_path = image_dir / f"{frame_index:06d}.jpg"
            target = label_dir / f"{frame_index:06d}.txt"
            target.write_text(
                yolo_label(box, int(info["width"]), int(info["height"])) if present else ""
            )
            images.append(image_path)
            if present:
                positives.append(image_path)
        images_by_video[stem] = images
        positives_by_video[stem] = positives

    rgb_train_all = read_paths(args.rgb_train_list)
    rgb_train_positive = [path for path in rgb_train_all if label_path(path).read_text().strip()]
    if not rgb_train_positive:
        raise RuntimeError("No positive Anti-UAV300 RGB training images")
    rgb_val = read_paths(args.rgb_val_list)

    folds: list[dict] = []
    for fold_index, holdout in enumerate(stems):
        fold_dir = args.output / "folds" / f"holdout_{holdout}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        gray_unique = [
            path
            for stem in stems
            if stem != holdout
            for path in positives_by_video[stem]
        ]
        rng = random.Random(args.seed + fold_index)
        gray_order = gray_unique.copy()
        rng.shuffle(gray_order)
        gray_balanced = [gray_order[index % len(gray_order)] for index in range(len(rgb_train_positive))]
        mixed = rgb_train_positive + gray_balanced
        rng.shuffle(mixed)

        train_list = fold_dir / "train_mixed_50_50.txt"
        holdout_all_list = fold_dir / "holdout_all_frames.txt"
        holdout_positive_list = fold_dir / "holdout_positive_frames.txt"
        write_lines(train_list, mixed)
        write_lines(holdout_all_list, images_by_video[holdout])
        write_lines(holdout_positive_list, positives_by_video[holdout])
        write_yaml(fold_dir / "train_rgb_monitor.yaml", train_list, args.rgb_val_list)
        write_yaml(fold_dir / "holdout_all.yaml", train_list, holdout_all_list)
        write_yaml(fold_dir / "holdout_positive.yaml", train_list, holdout_positive_list)
        folds.append(
            {
                "fold": f"holdout_{holdout}",
                "holdout": holdout,
                "rgb_unique_positive": len(rgb_train_positive),
                "gray_unique_positive": len(gray_unique),
                "gray_balanced_samples": len(gray_balanced),
                "train_samples": len(mixed),
                "holdout_all_frames": len(images_by_video[holdout]),
                "holdout_positive_frames": len(positives_by_video[holdout]),
                "gray_training_videos": [stem for stem in stems if stem != holdout],
            }
        )

    manifest = {
        "schema_version": "anti_uav.real_gray_yolo_lovo.v1",
        "seed": args.seed,
        "policy": {
            "new_gray_empty_frames_in_training": 0,
            "mix": "all positive Anti-UAV300 RGB plus repeated positive gray frames at 1:1",
            "holdout": "one complete gray video per fold",
            "training_validation": "Anti-UAV300 RGB val only",
            "post_training_evaluation": "complete gray holdout including absent frames",
        },
        "rgb": {
            "train_list": str(args.rgb_train_list),
            "train_images": len(rgb_train_all),
            "train_positive_images": len(rgb_train_positive),
            "val_list": str(args.rgb_val_list),
            "val_images": len(rgb_val),
        },
        "gray_videos": metadata,
        "folds": folds,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
