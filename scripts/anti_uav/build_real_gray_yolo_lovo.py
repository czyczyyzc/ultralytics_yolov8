#!/usr/bin/env python3
"""Build real-gray YOLO LOVO folds mixed with Anti-UAV300 RGB and negatives."""

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
    parser.add_argument(
        "--gray-data-root",
        type=Path,
        default=None,
        help="Optional existing extraction root containing images/gray and labels/gray.",
    )
    parser.add_argument(
        "--negative-fraction",
        type=float,
        default=0.0,
        help="Fraction of the final training list reserved for empty-label frames.",
    )
    parser.add_argument(
        "--hard-negative-list",
        type=Path,
        default=None,
        help="Optional ordered image list to prioritize within the negative quota.",
    )
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


def select_negatives(
    candidates: list[Path],
    hard_negatives: list[Path],
    positive_count: int,
    negative_fraction: float,
    seed: int,
) -> tuple[list[Path], int]:
    """Select unique negatives, prioritizing detector-mined hard examples."""
    if not 0.0 <= negative_fraction < 1.0:
        raise ValueError("negative_fraction must be in [0, 1)")
    if negative_fraction == 0.0:
        return [], 0

    unique_candidates = list(dict.fromkeys(candidates))
    candidate_set = set(unique_candidates)
    prioritized = [path for path in dict.fromkeys(hard_negatives) if path in candidate_set]
    target_count = round(positive_count * negative_fraction / (1.0 - negative_fraction))
    if target_count > len(unique_candidates):
        raise RuntimeError(
            f"Need {target_count} unique negatives for fraction {negative_fraction:.3f}, "
            f"but only {len(unique_candidates)} are available"
        )

    selected = prioritized[:target_count]
    selected_set = set(selected)
    remainder = [path for path in unique_candidates if path not in selected_set]
    random.Random(seed).shuffle(remainder)
    selected.extend(remainder[: target_count - len(selected)])
    return selected, min(len(prioritized), target_count)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.negative_fraction < 1.0:
        raise ValueError("--negative-fraction must be in [0, 1)")
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
    gray_data_root = args.gray_data_root or args.output
    with ThreadPoolExecutor(max_workers=args.extract_workers) as executor:
        futures = []
        for video in videos:
            futures.append(
                executor.submit(
                    extract_video,
                    video,
                    gray_data_root / "images" / "gray" / video.stem,
                    int(metadata[video.stem]["frames"]),
                    args.jpeg_qscale,
                )
            )
        for future in futures:
            future.result()

    images_by_video: dict[str, list[Path]] = {}
    positives_by_video: dict[str, list[Path]] = {}
    negatives_by_video: dict[str, list[Path]] = {}
    for stem in stems:
        info = metadata[stem]
        annotation = annotations[stem]
        image_dir = gray_data_root / "images" / "gray" / stem
        label_dir = gray_data_root / "labels" / "gray" / stem
        label_dir.mkdir(parents=True, exist_ok=True)
        images: list[Path] = []
        positives: list[Path] = []
        negatives: list[Path] = []
        for frame_index, (present, box) in enumerate(zip(annotation["exist"], annotation["gt_rect"])):
            image_path = image_dir / f"{frame_index:06d}.jpg"
            target = label_dir / f"{frame_index:06d}.txt"
            target.write_text(
                yolo_label(box, int(info["width"]), int(info["height"])) if present else ""
            )
            images.append(image_path)
            if present:
                positives.append(image_path)
            else:
                negatives.append(image_path)
        images_by_video[stem] = images
        positives_by_video[stem] = positives
        negatives_by_video[stem] = negatives

    rgb_train_all = read_paths(args.rgb_train_list)
    rgb_train_positive: list[Path] = []
    rgb_train_negative: list[Path] = []
    for path in rgb_train_all:
        target = rgb_train_positive if label_path(path).read_text().strip() else rgb_train_negative
        target.append(path)
    if not rgb_train_positive:
        raise RuntimeError("No positive Anti-UAV300 RGB training images")
    rgb_val = read_paths(args.rgb_val_list)
    hard_negatives = read_paths(args.hard_negative_list) if args.hard_negative_list else []
    all_gray_negatives = [path for stem in stems for path in negatives_by_video[stem]]
    all_negative_candidates = rgb_train_negative + all_gray_negatives
    write_lines(args.output / "negative_pool_all.txt", all_negative_candidates)

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
        positive_mixed = rgb_train_positive + gray_balanced
        negative_candidates = rgb_train_negative + [
            path for stem in stems if stem != holdout for path in negatives_by_video[stem]
        ]
        selected_negatives, hard_negative_count = select_negatives(
            negative_candidates,
            hard_negatives,
            len(positive_mixed),
            args.negative_fraction,
            args.seed + 1000 + fold_index,
        )
        mixed = positive_mixed + selected_negatives
        rng.shuffle(mixed)

        train_list = fold_dir / "train_mixed_50_50.txt"
        holdout_all_list = fold_dir / "holdout_all_frames.txt"
        holdout_positive_list = fold_dir / "holdout_positive_frames.txt"
        write_lines(train_list, mixed)
        write_lines(fold_dir / "train_positive_mixed_50_50.txt", positive_mixed)
        write_lines(fold_dir / "train_negative_selected.txt", selected_negatives)
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
                "rgb_unique_negative": len(rgb_train_negative),
                "gray_unique_negative": len(negative_candidates) - len(rgb_train_negative),
                "selected_negative_samples": len(selected_negatives),
                "selected_hard_negative_samples": hard_negative_count,
                "negative_fraction": len(selected_negatives) / len(mixed),
                "train_samples": len(mixed),
                "holdout_all_frames": len(images_by_video[holdout]),
                "holdout_positive_frames": len(positives_by_video[holdout]),
                "gray_training_videos": [stem for stem in stems if stem != holdout],
            }
        )

    final_dir = args.output / "final_all_gray"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_gray_unique = [path for stem in stems for path in positives_by_video[stem]]
    final_rng = random.Random(args.seed + len(stems))
    final_gray_order = final_gray_unique.copy()
    final_rng.shuffle(final_gray_order)
    final_gray_balanced = [
        final_gray_order[index % len(final_gray_order)] for index in range(len(rgb_train_positive))
    ]
    final_positive_mixed = rgb_train_positive + final_gray_balanced
    final_negatives, final_hard_negative_count = select_negatives(
        all_negative_candidates,
        hard_negatives,
        len(final_positive_mixed),
        args.negative_fraction,
        args.seed + 2000 + len(stems),
    )
    final_mixed = final_positive_mixed + final_negatives
    final_rng.shuffle(final_mixed)
    final_train_list = final_dir / "train_mixed_50_50.txt"
    write_lines(final_train_list, final_mixed)
    write_lines(final_dir / "train_positive_mixed_50_50.txt", final_positive_mixed)
    write_lines(final_dir / "train_negative_selected.txt", final_negatives)
    write_lines(final_dir / "train_gray_positive_unique.txt", final_gray_unique)
    write_lines(final_dir / "train_rgb_positive_unique.txt", rgb_train_positive)
    write_yaml(final_dir / "train_rgb_monitor.yaml", final_train_list, args.rgb_val_list)
    all_gray_negative_set = set(all_gray_negatives)
    final_training = {
        "name": final_dir.name,
        "rgb_unique_positive": len(rgb_train_positive),
        "gray_unique_positive": len(final_gray_unique),
        "gray_balanced_samples": len(final_gray_balanced),
        "rgb_unique_negative": len(rgb_train_negative),
        "gray_unique_negative": len(all_gray_negatives),
        "selected_negative_samples": len(final_negatives),
        "selected_hard_negative_samples": final_hard_negative_count,
        "negative_fraction": len(final_negatives) / len(final_mixed),
        "train_samples": len(final_mixed),
        "gray_training_videos": stems,
        "new_gray_empty_frames_in_training": sum(path in all_gray_negative_set for path in final_negatives),
        "train_list": str(final_train_list),
        "data_yaml": str(final_dir / "train_rgb_monitor.yaml"),
    }

    manifest = {
        "schema_version": "anti_uav.real_gray_yolo_lovo.v2",
        "seed": args.seed,
        "gray_data_root": str(gray_data_root),
        "policy": {
            "negative_fraction": args.negative_fraction,
            "hard_negative_list": str(args.hard_negative_list) if args.hard_negative_list else None,
            "mix": "1:1 Anti-UAV300 RGB and gray positives, plus unique empty-label negatives",
            "holdout": "one complete gray video per fold",
            "training_validation": "Anti-UAV300 RGB val only",
            "post_training_evaluation": "complete gray holdout including absent frames",
        },
        "rgb": {
            "train_list": str(args.rgb_train_list),
            "train_images": len(rgb_train_all),
            "train_positive_images": len(rgb_train_positive),
            "train_negative_images": len(rgb_train_negative),
            "val_list": str(args.rgb_val_list),
            "val_images": len(rgb_val),
        },
        "gray_videos": metadata,
        "folds": folds,
        "final_training": final_training,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
