#!/usr/bin/env python3
"""Build a leakage-safe, scale- and video-balanced RKNN calibration set."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from ultralytics.data.utils import img2label_paths


CATEGORIES = ("negative", "lt4", "tiny_4to8", "small_8to16", "other")
DEFAULT_FRACTIONS = {
    "negative": 0.15,
    "lt4": 0.15,
    "tiny_4to8": 0.30,
    "small_8to16": 0.20,
    "other": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Training data YAML containing a train list.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--exclude-token", nargs="*", default=())
    parser.add_argument(
        "--exclude-list",
        type=Path,
        action="append",
        default=[],
        help="Image manifest excluded by exact path. May be repeated; the data YAML val list is always excluded.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_from_yaml(value: str, data_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (data_path.parent / path).resolve()


def video_identity(image: Path) -> str:
    parent = image.parent
    if parent.name.lower() in {"rgb", "images", "frames", "imgs"}:
        parent = parent.parent
    return str(parent)


def read_long_edges(label: Path, width: int, height: int, input_hw: tuple[int, int]) -> list[float]:
    if not label.is_file():
        raise FileNotFoundError(f"Missing training label: {label}")
    gain = min(input_hw[1] / width, input_hw[0] / height)
    edges = []
    for line in label.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"Invalid YOLO label in {label}: {line}")
        box_width, box_height = float(fields[3]) * width, float(fields[4]) * height
        edges.append(max(box_width, box_height) * gain)
    return edges


def category_for(edges: list[float]) -> str:
    if not edges:
        return "negative"
    edge = min(edges)
    if edge < 4.0:
        return "lt4"
    if edge <= 8.0:
        return "tiny_4to8"
    if edge <= 16.0:
        return "small_8to16"
    return "other"


def cached_candidates(
    train_list: Path,
    source_paths: list[Path],
    labels: list[Path],
    input_hw: tuple[int, int],
) -> tuple[list[dict], Path | None]:
    """Use the audited Ultralytics label cache instead of opening every source image."""
    cache_path = train_list.with_suffix(".cache")
    if not cache_path.is_file():
        return [], None
    cache = np.load(cache_path, allow_pickle=True).item()
    cached = {row["im_file"]: row for row in cache.get("labels", [])}
    if any(str(path) not in cached for path in source_paths):
        return [], None
    candidates = []
    for image, label in zip(source_paths, labels):
        row = cached[str(image)]
        if row.get("bbox_format") != "xywh" or not row.get("normalized"):
            raise ValueError(f"Unsupported cached annotation format: {image}")
        height, width = (int(value) for value in row["shape"])
        boxes = np.asarray(row["bboxes"], dtype=np.float32)
        gain = min(input_hw[1] / width, input_hw[0] / height)
        edges = (np.maximum(boxes[:, 2] * width, boxes[:, 3] * height) * gain).tolist()
        candidates.append(
            dict(
                image=str(image),
                label=str(label),
                video=video_identity(image),
                category=category_for(edges),
                target_long_edges_px=[round(value, 4) for value in edges],
                source_hw=[height, width],
            )
        )
    return candidates, cache_path


def category_budgets(samples: int) -> dict[str, int]:
    raw = {name: samples * fraction for name, fraction in DEFAULT_FRACTIONS.items()}
    result = {name: int(value) for name, value in raw.items()}
    for name in sorted(CATEGORIES, key=lambda key: raw[key] - result[key], reverse=True)[: samples - sum(result.values())]:
        result[name] += 1
    return result


def round_robin(rows: list[dict], limit: int, rng: np.random.Generator) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["video"]].append(row)
    queues = {key: deque(np.asarray(values, dtype=object)[rng.permutation(len(values))].tolist()) for key, values in groups.items()}
    active = list(queues)
    rng.shuffle(active)
    selected = []
    while active and len(selected) < limit:
        following = []
        for key in active:
            selected.append(queues[key].popleft())
            if len(selected) == limit:
                break
            if queues[key]:
                following.append(key)
        active = following
        rng.shuffle(active)
    return selected


def letterbox(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    gain = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * gain)))
    resized_height = max(1, int(round(source_height * gain)))
    left = int(round((width - resized_width) * 0.5 - 0.1))
    top = int(round((height - resized_height) * 0.5 - 0.1))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if args.samples < len(CATEGORIES) or args.height <= 0 or args.width <= 0:
        raise ValueError("Invalid sample count or input shape")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refuse to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)

    data = yaml.safe_load(data_path.read_text())
    train_list = resolve_from_yaml(data["train"], data_path)
    source_paths = [Path(line.strip()).expanduser().resolve() for line in train_list.read_text().splitlines() if line.strip()]
    source_paths = list(dict.fromkeys(source_paths))
    excluded = tuple(args.exclude_token)
    if any(token in str(path) for path in source_paths for token in excluded):
        raise ValueError(f"Training list contains an excluded evaluation source: {excluded}")
    excluded_lists = [resolve_from_yaml(data["val"], data_path)] if data.get("val") else []
    excluded_lists.extend(path.expanduser().resolve() for path in args.exclude_list)
    excluded_paths = {
        str(Path(line.strip()).expanduser().resolve())
        for listing in excluded_lists
        for line in listing.read_text().splitlines()
        if line.strip()
    }
    overlap = excluded_paths.intersection(str(path) for path in source_paths)
    if overlap:
        raise ValueError(f"Training list overlaps excluded evaluation manifests by {len(overlap)} images")

    labels = [Path(value) for value in img2label_paths([str(path) for path in source_paths])]
    candidates, cache_path = cached_candidates(
        train_list,
        source_paths,
        labels,
        (args.height, args.width),
    )
    if not candidates:
        for image, label in zip(source_paths, labels):
            if not image.is_file():
                raise FileNotFoundError(image)
            decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError(f"Unable to decode training image: {image}")
            height, width = decoded.shape[:2]
            edges = read_long_edges(label, width, height, (args.height, args.width))
            candidates.append(
                dict(
                    image=str(image),
                    label=str(label),
                    video=video_identity(image),
                    category=category_for(edges),
                    target_long_edges_px=[round(value, 4) for value in edges],
                    source_hw=[height, width],
                )
            )

    rng = np.random.default_rng(args.seed)
    budgets = category_budgets(args.samples)
    selected = []
    for category in CATEGORIES:
        pool = [row for row in candidates if row["category"] == category]
        selected.extend(round_robin(pool, min(budgets[category], len(pool)), rng))
    if len(selected) < args.samples:
        selected_images = {row["image"] for row in selected}
        remainder = [row for row in candidates if row["image"] not in selected_images]
        selected.extend(round_robin(remainder, args.samples - len(selected), rng))
    if len(selected) != args.samples:
        raise ValueError(f"Only {len(selected)} unique calibration samples are available")
    rng.shuffle(selected)

    manifest = []
    for index, row in enumerate(selected):
        source = Path(row["image"])
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to decode calibration image: {source}")
        destination = image_dir / f"calib_{index:04d}.png"
        if not cv2.imwrite(str(destination), letterbox(image, args.height, args.width)):
            raise RuntimeError(f"Unable to write calibration image: {destination}")
        row["calibration_image"] = str(destination)
        manifest.append(str(destination))

    dataset_path = output / "dataset.txt"
    dataset_path.write_text("\n".join(manifest) + "\n")
    counts = {category: sum(row["category"] == category for row in selected) for category in CATEGORIES}
    audit = dict(
        data=str(data_path),
        train_list=str(train_list),
        train_list_sha256=sha256(train_list),
        label_cache=str(cache_path) if cache_path else None,
        label_cache_sha256=sha256(cache_path) if cache_path else None,
        unique_training_images=len(source_paths),
        input_hw=[args.height, args.width],
        samples=len(selected),
        seed=args.seed,
        excluded_tokens=list(excluded),
        excluded_lists=[str(path) for path in excluded_lists],
        excluded_images=len(excluded_paths),
        requested_fractions=DEFAULT_FRACTIONS,
        category_counts=counts,
        source_videos=len({row["video"] for row in selected}),
        dataset=str(dataset_path),
        rows=selected,
    )
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ("samples", "category_counts", "source_videos", "dataset")}, indent=2))


if __name__ == "__main__":
    main()
