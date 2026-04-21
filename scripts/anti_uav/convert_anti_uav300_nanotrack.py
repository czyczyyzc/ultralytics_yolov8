#!/usr/bin/env python3
"""Convert Anti-UAV300 into a NanoTrack-style crop511 dataset and JSON annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SOURCE_ROOT = Path("/mnt/chenziye/datasets/anti_uav/Anti-UAV300")
DEFAULT_OUTPUT_ROOT = Path("/mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack")
ALLOWED_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".mkv")
MODALITIES = ("rgb", "ir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help=f"Extracted Anti-UAV300 root. Default: {DEFAULT_SOURCE_ROOT}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help=f"Output NanoTrack dataset root. Default: {DEFAULT_OUTPUT_ROOT}")
    parser.add_argument("--modalities", nargs="+", choices=list(MODALITIES), default=list(MODALITIES), help="Modalities to export.")
    parser.add_argument("--frame-step", type=int, default=1, help="Keep every Nth positive frame.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when no explicit split file is provided.")
    parser.add_argument("--crop-size", type=int, default=511, help="Square crop size expected by NanoTrack preprocessing.")
    parser.add_argument("--exemplar-size", type=int, default=127, help="NanoTrack exemplar size used to derive crop scale.")
    parser.add_argument("--context-amount", type=float, default=0.5, help="Context padding around the target during crop export.")
    parser.add_argument("--min-box-size", type=float, default=2.0, help="Skip boxes smaller than this size in pixels.")
    parser.add_argument(
        "--background-frame-step",
        type=int,
        default=6,
        help="Export one same-scene background negative every N no-target frames. 0 disables background negatives.",
    )
    parser.add_argument(
        "--distractor-frame-step",
        type=int,
        default=2,
        help="Export one same-scene distractor negative every N positive frames. 0 disables distractor negatives.",
    )
    parser.add_argument(
        "--transition-window",
        type=int,
        default=8,
        help="Treat absent frames within this many frames of a visible segment as exit/re-entry transition negatives.",
    )
    parser.add_argument(
        "--hard-negative-errors",
        nargs="*",
        type=Path,
        default=[],
        help="Optional replay error logs or directories containing errors.jsonl files used to export hard negatives.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crops and JSON files.")
    return parser.parse_args()


def discover_sequences(source_root: Path) -> list[dict]:
    """Find Anti-UAV sequence directories with RGB/IR videos and labels."""
    sequences = {}
    for json_path in source_root.rglob("*_label.json"):
        sequence_dir = json_path.parent
        key = str(sequence_dir.relative_to(source_root))
        entry = sequences.setdefault(key, {"name": key.replace("/", "_"), "dir": sequence_dir, "modalities": {}})
        stem = json_path.stem.replace("_label", "")
        modality = stem.lower()
        if modality not in MODALITIES:
            continue
        video_path = find_video_for_modality(sequence_dir, stem)
        if video_path is None:
            continue
        entry["modalities"][modality] = {"video": video_path, "label": json_path}
    return [item for _, item in sorted(sequences.items()) if item["modalities"]]


def find_video_for_modality(sequence_dir: Path, stem: str) -> Path | None:
    """Find the matching video file for a modality."""
    for suffix in ALLOWED_VIDEO_SUFFIXES:
        candidate = sequence_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def split_sequences(sequence_names: list[str], val_ratio: float) -> tuple[set[str], set[str]]:
    """Deterministically split sequence names into train and val groups."""
    scored = sorted(((stable_score(name), name) for name in sequence_names), key=lambda item: (item[0], item[1]))
    val_count = max(1, int(round(len(scored) * val_ratio))) if len(scored) > 1 else 0
    val_names = {name for _, name in scored[:val_count]}
    train_names = {name for _, name in scored[val_count:]}
    if not train_names and val_names:
        moved = sorted(val_names)[-1]
        val_names.remove(moved)
        train_names.add(moved)
    return train_names, val_names


def stable_score(text: str) -> int:
    """Stable hash for deterministic sequence splits."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def compute_transition_absent_mask(boxes: list[list[float]], min_box_size: float, window: int) -> list[bool]:
    """Flag absent frames that sit close to a visible segment boundary."""
    present = [is_valid_bbox(box, min_box_size) for box in boxes]
    if window <= 0:
        return [False] * len(present)
    mask = [False] * len(present)
    for frame_index, is_present in enumerate(present):
        if is_present:
            continue
        left = max(0, frame_index - window)
        right = min(len(present), frame_index + window + 1)
        mask[frame_index] = any(present[left:frame_index]) or any(present[frame_index + 1:right])
    return mask


def normalize_xyxy_bbox(bbox: list[float] | tuple[float, ...]) -> list[float] | None:
    """Convert an xyxy bbox into xywh when it has a valid extent."""
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def collect_hard_negative_entries(inputs: list[Path]) -> dict[str, dict[int, list[float]]]:
    """Load replay-derived hard negatives from errors.jsonl logs."""
    collected: dict[str, dict[int, list[float]]] = {}
    if not inputs:
        return collected

    error_files: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            continue
        if path.is_file():
            error_files.append(path)
        else:
            error_files.extend(sorted(path.rglob("errors.jsonl")))

    for error_path in error_files:
        sequence_name = error_path.parent.name
        sequence_entries = collected.setdefault(sequence_name, {})
        for line in error_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") not in {"false_positive", "localization_error", "bad_alert"}:
                continue
            bbox_xywh = normalize_xyxy_bbox(payload.get("bbox"))
            frame_index = int(payload.get("frame_index", 0)) - 1
            if bbox_xywh is None or frame_index < 0 or frame_index in sequence_entries:
                continue
            sequence_entries[frame_index] = bbox_xywh
    return collected


def read_boxes(label_path: Path) -> list[list[float]]:
    """Load gt_rect boxes from an Anti-UAV label JSON file."""
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "gt_rect" in payload:
        return payload["gt_rect"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported annotation schema in {label_path}")


def crop_like_nanotrack(
    frame: np.ndarray,
    bbox_xywh: list[float],
    *,
    crop_size: int,
    exemplar_size: int,
    context_amount: float,
) -> np.ndarray:
    """Export a centered search-style crop compatible with NanoTrack crop511 datasets."""
    x, y, w, h = [float(v) for v in bbox_xywh[:4]]
    cx = x + (w - 1.0) / 2.0
    cy = y + (h - 1.0) / 2.0
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = max(np.sqrt(max(wc_z * hc_z, 1.0)), 1.0)
    s_x = s_z * (crop_size / float(exemplar_size))
    return get_subwindow_numpy(frame, (cx, cy), crop_size, int(round(s_x)))


def is_valid_bbox(box: list[float] | tuple[float, ...], min_box_size: float) -> bool:
    """Return True when the Anti-UAV box has a usable positive extent."""
    return bool(box) and len(box) >= 4 and float(box[2]) >= min_box_size and float(box[3]) >= min_box_size


def estimate_reference_size(boxes: list[list[float]], min_box_size: float) -> tuple[float, float]:
    """Estimate a stable per-sequence target size for background negative crops."""
    valid_sizes = np.array([[float(box[2]), float(box[3])] for box in boxes if is_valid_bbox(box, min_box_size)], dtype=np.float32)
    if len(valid_sizes) == 0:
        return 32.0, 32.0
    median_w, median_h = np.median(valid_sizes, axis=0)
    return float(max(median_w, min_box_size)), float(max(median_h, min_box_size))


def sample_background_bbox(frame_shape: tuple[int, ...], ref_size: tuple[float, float], seed_key: str) -> list[float]:
    """Pick a deterministic background crop anchor for frames without a target."""
    im_h, im_w = frame_shape[:2]
    w = min(max(ref_size[0], 12.0), max(im_w - 2, 12))
    h = min(max(ref_size[1], 12.0), max(im_h - 2, 12))
    anchors = np.array(
        [
            [0.20, 0.20],
            [0.50, 0.20],
            [0.80, 0.20],
            [0.20, 0.50],
            [0.50, 0.50],
            [0.80, 0.50],
            [0.20, 0.80],
            [0.50, 0.80],
            [0.80, 0.80],
        ],
        dtype=np.float32,
    )
    anchor = anchors[stable_score(seed_key) % len(anchors)]
    cx = float(np.clip(anchor[0] * im_w, w / 2.0, im_w - w / 2.0))
    cy = float(np.clip(anchor[1] * im_h, h / 2.0, im_h - h / 2.0))
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def sample_distractor_bbox(frame_shape: tuple[int, ...], gt_bbox_xywh: list[float], seed_key: str) -> list[float] | None:
    """Sample a near-target negative crop that stays in-scene but does not overlap the GT box."""
    im_h, im_w = frame_shape[:2]
    gx, gy, gw, gh = [float(value) for value in gt_bbox_xywh[:4]]
    gcx = gx + gw / 2.0
    gcy = gy + gh / 2.0
    min_center_distance = max(np.hypot(gw, gh) * 1.35, 12.0)
    seed = stable_score(seed_key)
    base_angle = (seed % 360) * np.pi / 180.0
    radii = (1.0, 1.35, 1.7, 2.1)
    angle_offsets = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)

    for radius in radii:
        for angle_offset in angle_offsets:
            angle = base_angle + angle_offset
            cx = gcx + np.cos(angle) * min_center_distance * radius
            cy = gcy + np.sin(angle) * min_center_distance * radius
            x = float(np.clip(cx - gw / 2.0, 0.0, max(im_w - gw, 0.0)))
            y = float(np.clip(cy - gh / 2.0, 0.0, max(im_h - gh, 0.0)))
            candidate = [x, y, gw, gh]
            if bbox_iou_xywh(candidate, gt_bbox_xywh) <= 0.05:
                return candidate
    return None


def bbox_iou_xywh(box1: list[float], box2: list[float]) -> float:
    """IoU helper for xywh-format boxes."""
    ax1, ay1, aw, ah = box1
    bx1, by1, bw, bh = box2
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = max(aw * ah + bw * bh - inter, 1e-6)
    return float(inter / union)


def get_subwindow_numpy(frame: np.ndarray, center_xy: tuple[float, float], crop_size: int, original_size: int) -> np.ndarray:
    """Numpy/OpenCV version of the Siamese tracker crop helper."""
    if crop_size <= 0 or original_size <= 0:
        raise ValueError("crop_size and original_size must be positive")

    cx, cy = center_xy
    avg_chans = np.mean(frame, axis=(0, 1))
    im_h, im_w = frame.shape[:2]
    c = (original_size + 1) / 2.0
    xmin = np.floor(cx - c + 0.5)
    xmax = xmin + original_size - 1
    ymin = np.floor(cy - c + 0.5)
    ymax = ymin + original_size - 1

    left_pad = int(max(0.0, -xmin))
    top_pad = int(max(0.0, -ymin))
    right_pad = int(max(0.0, xmax - im_w + 1))
    bottom_pad = int(max(0.0, ymax - im_h + 1))

    xmin += left_pad
    xmax += left_pad
    ymin += top_pad
    ymax += top_pad

    padded = frame
    if any((top_pad, bottom_pad, left_pad, right_pad)):
        padded = np.zeros((im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, frame.shape[2]), dtype=np.uint8)
        padded[top_pad:top_pad + im_h, left_pad:left_pad + im_w, :] = frame
        if top_pad:
            padded[:top_pad, left_pad:left_pad + im_w, :] = avg_chans
        if bottom_pad:
            padded[im_h + top_pad:, left_pad:left_pad + im_w, :] = avg_chans
        if left_pad:
            padded[:, :left_pad, :] = avg_chans
        if right_pad:
            padded[:, im_w + left_pad:, :] = avg_chans

    crop = padded[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size))
    return crop


def export_sequence(
    sequence: dict,
    modality: str,
    output_root: Path,
    frame_step: int,
    crop_size: int,
    exemplar_size: int,
    context_amount: float,
    min_box_size: float,
    background_frame_step: int,
    distractor_frame_step: int,
    transition_window: int,
    hard_negative_entries: dict[int, list[float]] | None,
    overwrite: bool,
) -> dict:
    """Export one sequence/modality to NanoTrack crops plus JSON metadata."""
    meta = sequence["modalities"][modality]
    boxes = read_boxes(meta["label"])
    video_path = meta["video"]
    crop_root = output_root / modality / "crop511"
    sequence_dir = crop_root / sequence["name"]
    sequence_dir.mkdir(parents=True, exist_ok=True)

    positive_track_key = "00"
    distractor_track_key = "__neg__"
    background_track_key = "__bg__"
    transition_background_track_key = "__bg_transition__"
    hard_negative_track_key = "__hardneg__"
    track_meta = {
        positive_track_key: {},
        distractor_track_key: {},
        background_track_key: {},
        transition_background_track_key: {},
        hard_negative_track_key: {},
    }
    counts = {"positive": 0, "distractor": 0, "background": 0, "transition_background": 0, "hard_negative": 0}
    reference_size = estimate_reference_size(boxes, min_box_size)
    transition_absent_mask = compute_transition_absent_mask(boxes, min_box_size, transition_window)
    hard_negative_entries = hard_negative_entries or {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_index = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_key = f"{frame_index:06d}"
            bbox = boxes[frame_index] if frame_index < len(boxes) else []
            has_target = is_valid_bbox(bbox, min_box_size)

            if has_target:
                x, y, w, h = [float(value) for value in bbox[:4]]
                if frame_index % frame_step == 0:
                    crop = crop_like_nanotrack(
                        frame,
                        [x, y, w, h],
                        crop_size=crop_size,
                        exemplar_size=exemplar_size,
                        context_amount=context_amount,
                    )
                    crop_path = sequence_dir / f"{frame_key}.{positive_track_key}.x.jpg"
                    if overwrite or not crop_path.exists():
                        cv2.imwrite(str(crop_path), crop)
                    track_meta[positive_track_key][frame_key] = [0.0, 0.0, float(w), float(h)]
                    counts["positive"] += 1

                if distractor_frame_step > 0 and frame_index % distractor_frame_step == 0:
                    distractor_bbox = sample_distractor_bbox(frame.shape, [x, y, w, h], f"{sequence['name']}:{modality}:neg:{frame_key}")
                    if distractor_bbox is not None:
                        distractor_crop = crop_like_nanotrack(
                            frame,
                            distractor_bbox,
                            crop_size=crop_size,
                            exemplar_size=exemplar_size,
                            context_amount=context_amount,
                        )
                        distractor_path = sequence_dir / f"{frame_key}.{distractor_track_key}.x.jpg"
                        if overwrite or not distractor_path.exists():
                            cv2.imwrite(str(distractor_path), distractor_crop)
                        track_meta[distractor_track_key][frame_key] = [0.0, 0.0, float(distractor_bbox[2]), float(distractor_bbox[3])]
                        counts["distractor"] += 1
            else:
                background_bbox = None
                if background_frame_step > 0 and frame_index % background_frame_step == 0:
                    background_bbox = sample_background_bbox(frame.shape, reference_size, f"{sequence['name']}:{modality}:bg:{frame_key}")
                    background_crop = crop_like_nanotrack(
                        frame,
                        background_bbox,
                        crop_size=crop_size,
                        exemplar_size=exemplar_size,
                        context_amount=context_amount,
                    )
                    background_path = sequence_dir / f"{frame_key}.{background_track_key}.x.jpg"
                    if overwrite or not background_path.exists():
                        cv2.imwrite(str(background_path), background_crop)
                    track_meta[background_track_key][frame_key] = [0.0, 0.0, float(background_bbox[2]), float(background_bbox[3])]
                    counts["background"] += 1

                if transition_absent_mask[frame_index]:
                    if background_bbox is None:
                        background_bbox = sample_background_bbox(
                            frame.shape,
                            reference_size,
                            f"{sequence['name']}:{modality}:bg_transition:{frame_key}",
                        )
                    transition_crop = crop_like_nanotrack(
                        frame,
                        background_bbox,
                        crop_size=crop_size,
                        exemplar_size=exemplar_size,
                        context_amount=context_amount,
                    )
                    transition_path = sequence_dir / f"{frame_key}.{transition_background_track_key}.x.jpg"
                    if overwrite or not transition_path.exists():
                        cv2.imwrite(str(transition_path), transition_crop)
                    track_meta[transition_background_track_key][frame_key] = [
                        0.0,
                        0.0,
                        float(background_bbox[2]),
                        float(background_bbox[3]),
                    ]
                    counts["transition_background"] += 1

            hard_negative_bbox = hard_negative_entries.get(frame_index)
            if hard_negative_bbox is not None:
                hard_negative_crop = crop_like_nanotrack(
                    frame,
                    hard_negative_bbox,
                    crop_size=crop_size,
                    exemplar_size=exemplar_size,
                    context_amount=context_amount,
                )
                hard_negative_path = sequence_dir / f"{frame_key}.{hard_negative_track_key}.x.jpg"
                if overwrite or not hard_negative_path.exists():
                    cv2.imwrite(str(hard_negative_path), hard_negative_crop)
                track_meta[hard_negative_track_key][frame_key] = [
                    0.0,
                    0.0,
                    float(hard_negative_bbox[2]),
                    float(hard_negative_bbox[3]),
                ]
                counts["hard_negative"] += 1
            frame_index += 1
    finally:
        cap.release()

    filtered_tracks = {track_name: frames for track_name, frames in track_meta.items() if frames}
    return {"meta": {sequence["name"]: filtered_tracks} if filtered_tracks else {}, "counts": counts}


def merge_meta(target: dict, payload: dict) -> None:
    """Merge nested video->track->frame NanoTrack metadata."""
    for video_name, tracks in payload.items():
        video_entry = target.setdefault(video_name, {})
        for track_name, frames in tracks.items():
            video_entry.setdefault(track_name, {}).update(frames)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    sequences = discover_sequences(source_root)
    if not sequences:
        raise RuntimeError(f"No Anti-UAV sequences found under {source_root}")
    hard_negative_entries = collect_hard_negative_entries(args.hard_negative_errors)

    train_sequences, val_sequences = split_sequences([sequence["name"] for sequence in sequences], args.val_ratio)
    summary = {"source_root": str(source_root), "output_root": str(output_root), "sequence_count": len(sequences), "modalities": {}}

    for modality in args.modalities:
        train_meta = {}
        val_meta = {}
        train_counts = {"positive": 0, "distractor": 0, "background": 0, "transition_background": 0, "hard_negative": 0}
        val_counts = {"positive": 0, "distractor": 0, "background": 0, "transition_background": 0, "hard_negative": 0}
        split_manifest = {"train": [], "val": []}
        for sequence in sequences:
            if modality not in sequence["modalities"]:
                continue
            result = export_sequence(
                sequence=sequence,
                modality=modality,
                output_root=output_root,
                frame_step=max(1, args.frame_step),
                crop_size=args.crop_size,
                exemplar_size=args.exemplar_size,
                context_amount=args.context_amount,
                min_box_size=args.min_box_size,
                background_frame_step=max(0, args.background_frame_step),
                distractor_frame_step=max(0, args.distractor_frame_step),
                transition_window=max(0, args.transition_window),
                hard_negative_entries=hard_negative_entries.get(sequence["name"]),
                overwrite=args.overwrite,
            )
            if sequence["name"] in train_sequences:
                merge_meta(train_meta, result["meta"])
                for key in train_counts:
                    train_counts[key] += result["counts"][key]
                if result["meta"]:
                    split_manifest["train"].append(
                        {
                            "name": sequence["name"],
                            "source_dir": str(sequence["dir"].resolve()),
                            "video": str(sequence["modalities"][modality]["video"].resolve()),
                            "label": str(sequence["modalities"][modality]["label"].resolve()),
                        }
                    )
            else:
                merge_meta(val_meta, result["meta"])
                for key in val_counts:
                    val_counts[key] += result["counts"][key]
                if result["meta"]:
                    split_manifest["val"].append(
                        {
                            "name": sequence["name"],
                            "source_dir": str(sequence["dir"].resolve()),
                            "video": str(sequence["modalities"][modality]["video"].resolve()),
                            "label": str(sequence["modalities"][modality]["label"].resolve()),
                        }
                    )

        modality_root = output_root / modality
        modality_root.mkdir(parents=True, exist_ok=True)
        train_path = modality_root / "train.json"
        val_path = modality_root / "val.json"
        split_manifest_path = modality_root / "split_manifest.json"
        if args.overwrite or not train_path.exists():
            train_path.write_text(json.dumps(train_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.overwrite or not val_path.exists():
            val_path.write_text(json.dumps(val_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.overwrite or not split_manifest_path.exists():
            split_manifest_path.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        summary["modalities"][modality] = {
            "crop_root": str((modality_root / "crop511").resolve()),
            "train_json": str(train_path.resolve()),
            "val_json": str(val_path.resolve()),
            "split_manifest_json": str(split_manifest_path.resolve()),
            "train_sequences": sum(1 for name in train_sequences if name in train_meta),
            "val_sequences": sum(1 for name in val_sequences if name in val_meta),
            "train_positive_frames": train_counts["positive"],
            "train_distractor_frames": train_counts["distractor"],
            "train_background_frames": train_counts["background"],
            "train_transition_background_frames": train_counts["transition_background"],
            "train_hard_negative_frames": train_counts["hard_negative"],
            "val_positive_frames": val_counts["positive"],
            "val_distractor_frames": val_counts["distractor"],
            "val_background_frames": val_counts["background"],
            "val_transition_background_frames": val_counts["transition_background"],
            "val_hard_negative_frames": val_counts["hard_negative"],
        }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
