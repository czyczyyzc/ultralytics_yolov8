#!/usr/bin/env python3
"""Prepare external RGB anti-UAV datasets for YOLO and NanoTrack training.

The script is intentionally conservative:

* drone/UAV boxes are exported as YOLO class 0.
* bird/airplane/helicopter frames are exported as hard-negative empty-label images.
* tracking-style videos with per-frame boxes are exported to both YOLO frames and NanoTrack crop511.

Supported inputs are layout-tolerant rather than tied to one archive name. This lets the downloader unpack public
datasets into a raw directory first, then run this converter once the actual folder names are known.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.convert_anti_uav300_nanotrack import crop_like_nanotrack  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}
DRONE_TOKENS = ("drone", "uav", "multirotor", "quadrotor", "mavic", "phantom", "x500")
NEGATIVE_TOKENS = ("bird", "airplane", "plane", "helicopter", "heli")


@dataclass(frozen=True)
class FrameRecord:
    dataset: str
    split: str
    image_path: Path
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    is_hard_negative: bool = False
    sequence_name: str | None = None
    frame_index: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Root containing downloaded/extracted external datasets.")
    parser.add_argument("--yolo-root", type=Path, required=True, help="Output YOLO root with images/labels/{train,val}.")
    parser.add_argument("--nanotrack-root", type=Path, required=True, help="Output NanoTrack root with rgb/crop511 and train/val json.")
    parser.add_argument("--datasets", nargs="+", default=["dut", "halmstad"], choices=("dut", "halmstad", "aod4", "generic-yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio for unsplit external sequences/images.")
    parser.add_argument("--frame-step", type=int, default=3, help="Keep every Nth frame from videos for YOLO/NanoTrack export.")
    parser.add_argument("--negative-frame-step", type=int, default=12, help="Keep every Nth hard-negative frame from non-drone videos.")
    parser.add_argument("--max-video-frames", type=int, default=0, help="Optional cap per video, 0 keeps all sampled frames.")
    parser.add_argument("--crop-size", type=int, default=511)
    parser.add_argument("--exemplar-size", type=int, default=127)
    parser.add_argument("--context-amount", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_split(name: str, val_ratio: float) -> str:
    import hashlib

    score = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if score < val_ratio else "train"


def clamp_box(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def xyxy_to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def safe_name(*parts: object) -> str:
    raw = "_".join(str(part) for part in parts if part is not None and str(part) != "")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def video_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def parse_voc_xml(path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    root = ET.parse(path).getroot()
    parsed: list[tuple[str, tuple[float, float, float, float]]] = []
    for obj in root.findall(".//object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            x1 = float(box.findtext("xmin", "nan"))
            y1 = float(box.findtext("ymin", "nan"))
            x2 = float(box.findtext("xmax", "nan"))
            y2 = float(box.findtext("ymax", "nan"))
        except ValueError:
            continue
        if all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            parsed.append((name, (x1, y1, x2, y2)))
    return parsed


def parse_txt_boxes(path: Path, width: int, height: int) -> list[tuple[str, tuple[float, float, float, float]]]:
    parsed: list[tuple[str, tuple[float, float, float, float]]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.replace(",", " ").split()
        label = "drone"
        if parts and not is_number(parts[0]):
            label = parts[0].lower()
            parts = parts[1:]
        values = [float(part) for part in parts if is_number(part)]
        if len(values) < 4:
            continue
        if len(values) >= 5 and values[0] in {0.0, 1.0}:
            values = values[1:]
        a, b, c, d = values[:4]
        if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1 and 0 <= d <= 1:
            x1 = (a - c / 2) * width
            y1 = (b - d / 2) * height
            x2 = (a + c / 2) * width
            y2 = (b + d / 2) * height
        elif c > a and d > b:
            x1, y1, x2, y2 = a, b, c, d
        else:
            x1, y1, x2, y2 = a, b, a + c, b + d
        parsed.append((label, (x1, y1, x2, y2)))
    return parsed


def is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def find_annotation_for_image(image_path: Path) -> Path | None:
    candidates = [
        image_path.with_suffix(".xml"),
        image_path.with_suffix(".txt"),
        image_path.parent.parent / "xml" / f"{image_path.stem}.xml",
        image_path.parent.parent / "XML" / f"{image_path.stem}.xml",
        image_path.parent.parent / "Annotations" / f"{image_path.stem}.xml",
        image_path.parent.parent / "annotations" / f"{image_path.stem}.xml",
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent.parent / "Labels" / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def label_kind(text: str) -> str:
    normalized = text.lower()
    if any(token in normalized for token in DRONE_TOKENS):
        return "drone"
    if any(token in normalized for token in NEGATIVE_TOKENS):
        return "negative"
    return "unknown"


def load_static_image_records(root: Path, dataset_name: str, val_ratio: float) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for image_path in image_paths(root):
        lowered_parts = {part.lower() for part in image_path.parts}
        if "anti-uav-tracking-v0" in lowered_parts or "_external_rgb_frames" in lowered_parts:
            continue
        ann_path = find_annotation_for_image(image_path)
        if ann_path is None:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        if ann_path.suffix.lower() == ".xml":
            raw_boxes = parse_voc_xml(ann_path)
        else:
            raw_boxes = parse_txt_boxes(ann_path, width, height)

        drone_boxes: list[tuple[float, float, float, float]] = []
        has_negative_label = False
        for label, box in raw_boxes:
            kind = label_kind(label)
            if kind == "negative":
                has_negative_label = True
                continue
            clamped = clamp_box(box, width, height)
            if clamped is not None and kind in {"drone", "unknown"}:
                drone_boxes.append(clamped)
        if not drone_boxes and not has_negative_label:
            continue
        split = infer_split(image_path, stable_split(str(image_path.relative_to(root)), val_ratio))
        records.append(
            FrameRecord(
                dataset=dataset_name,
                split=split,
                image_path=image_path,
                boxes_xyxy=tuple(drone_boxes),
                is_hard_negative=has_negative_label and not drone_boxes,
            )
        )
    return records


def infer_split(path: Path, fallback: str) -> str:
    lowered = {part.lower() for part in path.parts}
    if "train" in lowered or "training" in lowered:
        return "train"
    if "val" in lowered or "valid" in lowered or "validation" in lowered:
        return "val"
    return fallback


def try_load_mcos_groundtruth(path: Path) -> list[tuple[float, float, float, float] | None] | None:
    try:
        from mcos_decoder import load_groundtruth
    except Exception:
        return None
    boxes = load_groundtruth(str(path))
    normalized: list[tuple[float, float, float, float] | None] = []
    for box in boxes:
        if box is None:
            normalized.append(None)
        else:
            x, y, w, h = [float(value) for value in box[:4]]
            normalized.append((x, y, x + w, y + h) if w > 0 and h > 0 else None)
    return normalized


def find_label_for_video(video_path: Path) -> Path | None:
    stem = video_path.stem
    candidates = [
        video_path.with_name(f"{stem}_LABELS.mat"),
        video_path.with_name(f"{stem}_LABEL.mat"),
        video_path.with_name(f"{stem}.mat"),
        video_path.parent / f"{stem}_LABELS.mat",
        video_path.parent / f"{stem}_LABEL.mat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(video_path.parent.glob(f"*{stem}*.mat"))
    return matches[0] if matches else None


def load_video_records(root: Path, dataset_name: str, val_ratio: float, frame_step: int, negative_frame_step: int, max_video_frames: int) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    extracted_root = root / "_external_rgb_frames"
    for video_path in video_paths(root):
        kind = label_kind(video_path.stem)
        if kind == "unknown":
            continue
        label_path = find_label_for_video(video_path)
        boxes: list[tuple[float, float, float, float] | None] | None = None
        if label_path and label_path.suffix.lower() == ".mat":
            boxes = try_load_mcos_groundtruth(label_path)
        if kind == "drone" and boxes is None:
            continue

        sequence_name = safe_name(dataset_name, video_path.relative_to(root).with_suffix(""))
        split = infer_split(video_path, stable_split(sequence_name, val_ratio))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        kept = 0
        frame_index = 0
        sample_step = frame_step if kind == "drone" else negative_frame_step
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % max(1, sample_step) != 0:
                frame_index += 1
                continue
            if max_video_frames and kept >= max_video_frames:
                break
            height, width = frame.shape[:2]
            box = boxes[frame_index] if boxes is not None and frame_index < len(boxes) else None
            clamped = clamp_box(box, width, height) if box is not None else None
            if kind == "drone" and clamped is None:
                frame_index += 1
                continue
            frame_dir = extracted_root / sequence_name
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / f"{frame_index:06d}.jpg"
            if not frame_path.exists():
                cv2.imwrite(str(frame_path), frame)
            records.append(
                FrameRecord(
                    dataset=dataset_name,
                    split=split,
                    image_path=frame_path,
                    boxes_xyxy=((clamped,) if clamped is not None else ()),
                    is_hard_negative=(kind == "negative"),
                    sequence_name=sequence_name,
                    frame_index=frame_index,
                )
            )
            kept += 1
            frame_index += 1
        cap.release()
    return records


def parse_xywh_gt(path: Path) -> list[tuple[float, float, float, float] | None]:
    boxes: list[tuple[float, float, float, float] | None] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        values = [float(part) for part in line.replace(",", " ").split() if is_number(part)]
        if len(values) < 4:
            boxes.append(None)
            continue
        x, y, w, h = values[:4]
        boxes.append((x, y, x + w, y + h) if w > 0 and h > 0 else None)
    return boxes


def load_image_sequence_records(root: Path, dataset_name: str, val_ratio: float, frame_step: int, max_video_frames: int) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for tracking_root in root.rglob("Anti-UAV-Tracking-V0"):
        if not tracking_root.is_dir():
            continue
        gt_root = tracking_root.with_name(f"{tracking_root.name}GT")
        if not gt_root.exists():
            continue
        for sequence_dir in sorted(path for path in tracking_root.iterdir() if path.is_dir()):
            gt_path = gt_root / f"{sequence_dir.name}_gt.txt"
            if not gt_path.exists():
                continue
            boxes = parse_xywh_gt(gt_path)
            images = sorted(path for path in sequence_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            sequence_name = safe_name(dataset_name, tracking_root.name, sequence_dir.name)
            split = infer_split(sequence_dir, stable_split(sequence_name, val_ratio))
            kept = 0
            for position, image_path in enumerate(images):
                frame_number = int(image_path.stem) if image_path.stem.isdigit() else position + 1
                frame_index = frame_number - 1
                if frame_index % max(1, frame_step) != 0:
                    continue
                if max_video_frames and kept >= max_video_frames:
                    break
                if frame_index < 0 or frame_index >= len(boxes) or boxes[frame_index] is None:
                    continue
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                height, width = frame.shape[:2]
                clamped = clamp_box(boxes[frame_index], width, height)
                if clamped is None:
                    continue
                records.append(
                    FrameRecord(
                        dataset=dataset_name,
                        split=split,
                        image_path=image_path,
                        boxes_xyxy=(clamped,),
                        sequence_name=sequence_name,
                        frame_index=frame_index,
                    )
                )
                kept += 1
    return records


def load_aod4_records(root: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for aod_root in root.rglob("AOD 4"):
        if not aod_root.is_dir():
            continue
        coco_root = aod_root / "Annotations" / "COCO Annotation format"
        image_root = aod_root / "Images"
        if not coco_root.exists() or not image_root.exists():
            continue
        for source_split, target_split in (("train", "train"), ("valid", "val"), ("test", "val")):
            annotation_path = coco_root / source_split / "_annotations.coco.json"
            split_image_root = image_root / source_split
            if not annotation_path.exists() or not split_image_root.exists():
                continue
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            categories = {int(item["id"]): str(item.get("name", "")).lower() for item in data.get("categories", [])}
            images = {int(item["id"]): item for item in data.get("images", [])}
            grouped_annotations: dict[int, list[dict[str, object]]] = {}
            for annotation in data.get("annotations", []):
                grouped_annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
            for image_id, image_info in images.items():
                file_name = str(image_info.get("file_name", ""))
                image_path = split_image_root / file_name
                if not image_path.exists():
                    continue
                width = int(image_info.get("width") or 0)
                height = int(image_info.get("height") or 0)
                if width <= 0 or height <= 0:
                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        continue
                    height, width = frame.shape[:2]
                drone_boxes: list[tuple[float, float, float, float]] = []
                has_non_drone = False
                for annotation in grouped_annotations.get(image_id, []):
                    category_name = categories.get(int(annotation.get("category_id", -1)), "")
                    bbox = annotation.get("bbox") or []
                    if not isinstance(bbox, list) or len(bbox) < 4:
                        continue
                    x, y, w, h = [float(value) for value in bbox[:4]]
                    clamped = clamp_box((x, y, x + w, y + h), width, height)
                    if clamped is None:
                        continue
                    if "drone" in category_name:
                        drone_boxes.append(clamped)
                    else:
                        has_non_drone = True
                if not drone_boxes and not has_non_drone:
                    continue
                records.append(
                    FrameRecord(
                        dataset="aod4",
                        split=target_split,
                        image_path=image_path,
                        boxes_xyxy=tuple(drone_boxes),
                        is_hard_negative=has_non_drone and not drone_boxes,
                    )
                )
    return records


def discover_records(args: argparse.Namespace) -> tuple[list[FrameRecord], dict[str, object]]:
    raw_root = args.raw_root.expanduser().resolve()
    records: list[FrameRecord] = []
    notes: dict[str, object] = {"raw_root": str(raw_root), "datasets": {}}

    for dataset in args.datasets:
        if dataset == "dut":
            roots = [path for path in raw_root.rglob("*") if path.is_dir() and "dut" in path.name.lower()]
            if not roots and (raw_root / "dut_anti_uav").exists():
                roots = [raw_root / "dut_anti_uav"]
            roots = roots or [raw_root]
            before = len(records)
            for root in roots[:4]:
                records.extend(load_static_image_records(root, "dut_anti_uav", args.val_ratio))
                records.extend(load_image_sequence_records(root, "dut_anti_uav", args.val_ratio, args.frame_step, args.max_video_frames))
                records.extend(load_video_records(root, "dut_anti_uav", args.val_ratio, args.frame_step, args.negative_frame_step, args.max_video_frames))
            notes["datasets"][dataset] = {"records": len(records) - before, "roots": [str(root) for root in roots[:4]]}
        elif dataset == "halmstad":
            roots = [path for path in raw_root.rglob("*") if path.is_dir() and ("halmstad" in path.name.lower() or "drone-detection" in path.name.lower())]
            roots = roots or [raw_root]
            before = len(records)
            for root in roots[:4]:
                records.extend(load_video_records(root, "halmstad_drone_detection", args.val_ratio, args.frame_step, args.negative_frame_step, args.max_video_frames))
            notes["datasets"][dataset] = {"records": len(records) - before, "roots": [str(root) for root in roots[:4]]}
        elif dataset == "aod4":
            roots = [raw_root / "aod4"] if (raw_root / "aod4").exists() else []
            roots = roots or [path for path in raw_root.iterdir() if path.is_dir() and "aod" in path.name.lower()]
            roots = roots or [raw_root]
            before = len(records)
            for root in roots[:1]:
                records.extend(load_aod4_records(root))
            notes["datasets"][dataset] = {"records": len(records) - before, "roots": [str(root) for root in roots[:1]]}
        elif dataset == "generic-yolo":
            before = len(records)
            records.extend(load_static_image_records(raw_root, "generic_yolo", args.val_ratio))
            notes["datasets"][dataset] = {"records": len(records) - before, "roots": [str(raw_root)]}
    return records, notes


def export_yolo(records: list[FrameRecord], yolo_root: Path, overwrite: bool) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "positive": 0, "hard_negative": 0}
    for record in records:
        split = record.split
        stem = safe_name(record.dataset, record.sequence_name, record.image_path.stem)
        image_out = yolo_root / "images" / split / f"{stem}.jpg"
        label_out = yolo_root / "labels" / split / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not image_out.exists():
            shutil.copy2(record.image_path, image_out)
        frame = cv2.imread(str(record.image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        lines = [xyxy_to_yolo(box, width, height) for box in record.boxes_xyxy]
        label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] += 1
        if lines:
            counts["positive"] += 1
        else:
            counts["hard_negative"] += 1
    write_yolo_yaml(yolo_root)
    return counts


def write_yolo_yaml(yolo_root: Path) -> None:
    (yolo_root / "ExternalRGBDrone.yaml").write_text(
        f"path: {yolo_root}\ntrain: images/train\nval: images/val\n\nnames:\n  0: drone\n",
        encoding="utf-8",
    )


def export_nanotrack(records: list[FrameRecord], nanotrack_root: Path, args: argparse.Namespace) -> dict[str, int]:
    modality_root = nanotrack_root / "rgb"
    crop_root = modality_root / "crop511"
    crop_root.mkdir(parents=True, exist_ok=True)
    split_meta: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {"train": {}, "val": {}}
    counts = {"train_sequences": 0, "val_sequences": 0, "train_frames": 0, "val_frames": 0}
    grouped: dict[tuple[str, str, str], list[FrameRecord]] = {}
    for record in records:
        if not record.sequence_name or not record.boxes_xyxy:
            continue
        grouped.setdefault((record.split, record.dataset, record.sequence_name), []).append(record)

    for (split, dataset, sequence_name), group in sorted(grouped.items()):
        video_name = safe_name(dataset, sequence_name)
        target_dir = crop_root / video_name
        target_dir.mkdir(parents=True, exist_ok=True)
        frames_meta: dict[str, list[float]] = {}
        for item in sorted(group, key=lambda record: record.frame_index or 0):
            frame = cv2.imread(str(item.image_path))
            if frame is None:
                continue
            x1, y1, x2, y2 = item.boxes_xyxy[0]
            w = x2 - x1
            h = y2 - y1
            crop = crop_like_nanotrack(
                frame,
                [x1, y1, w, h],
                crop_size=args.crop_size,
                exemplar_size=args.exemplar_size,
                context_amount=args.context_amount,
            )
            frame_key = f"{item.frame_index or 0:06d}"
            crop_path = target_dir / f"{frame_key}.00.x.jpg"
            if args.overwrite or not crop_path.exists():
                cv2.imwrite(str(crop_path), crop)
            frames_meta[frame_key] = [0.0, 0.0, float(w), float(h)]
        if frames_meta:
            split_meta[split][video_name] = {"00": frames_meta}
            counts[f"{split}_frames"] += len(frames_meta)

    for split in ("train", "val"):
        (modality_root / f"{split}.json").write_text(json.dumps(split_meta[split], indent=2, ensure_ascii=False), encoding="utf-8")
        counts[f"{split}_sequences"] = len(split_meta[split])
    return counts


def main() -> None:
    args = parse_args()
    yolo_root = args.yolo_root.expanduser().resolve()
    nanotrack_root = args.nanotrack_root.expanduser().resolve()
    records, notes = discover_records(args)
    yolo_counts = export_yolo(records, yolo_root, args.overwrite)
    nanotrack_counts = export_nanotrack(records, nanotrack_root, args)
    summary = {
        **notes,
        "yolo_root": str(yolo_root),
        "nanotrack_root": str(nanotrack_root),
        "records": len(records),
        "yolo": yolo_counts,
        "nanotrack": nanotrack_counts,
    }
    yolo_root.mkdir(parents=True, exist_ok=True)
    nanotrack_root.mkdir(parents=True, exist_ok=True)
    (yolo_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (nanotrack_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
