#!/usr/bin/env python3
"""Run the alerting-only anti-UAV pipeline on RK3588 with YOLO (.onnx/.rknn) and NanoTrack RKNN."""

from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import importlib.util
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator, Optional, Sequence

import cv2
import numpy as np


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
DEFAULT_PRESENCE_MODEL_ENV = "ANTI_UAV_DEFAULT_PRESENCE_MODEL"
DEFAULT_PRESENCE_MODEL_NAME = "pair_presence_edl.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Detector model path (.rknn or .onnx).")
    parser.add_argument("--source", required=True, help="Image, directory, video, or camera index.")
    parser.add_argument("--input-size", default="640,640", help="Detector input size as H,W.")
    parser.add_argument("--tracker", default="nanotrack_rknn", choices=("nanotrack_rknn", "template_match"))
    parser.add_argument("--nanotrack-root", default="", help="NanoTrack_RK3588_python checkout root.")
    parser.add_argument("--nanotrack-config", default="", help="NanoTrack config yaml path.")
    parser.add_argument("--nanotrack-tback", default="", help="NanoTrack template-backbone RKNN path.")
    parser.add_argument("--nanotrack-xback", default="", help="NanoTrack search-backbone RKNN path.")
    parser.add_argument("--nanotrack-head", default="", help="NanoTrack head RKNN path.")
    parser.add_argument(
        "--presence-verifier",
        default="",
        choices=("", "none", "heuristic", "mlp", "pair_head", "pair_head_edl"),
        help="Optional lightweight presence verifier over tracker outputs. Defaults to pair_head_edl when a default checkpoint is available; use 'none' to disable it explicitly.",
    )
    parser.add_argument("--presence-model", default="", help="Optional presence verifier checkpoint path.")
    parser.add_argument("--presence-metadata", default="", help="Optional presence verifier sidecar JSON path.")
    parser.add_argument("--presence-device", default="", help="Optional torch device for the presence verifier.")
    parser.add_argument(
        "--presence-score-thresh",
        type=float,
        default=0.45,
        help="Presence score threshold below which tracking is treated as suspect.",
    )
    parser.add_argument(
        "--presence-uncertainty-thresh",
        type=float,
        default=0.25,
        help="Optional uncertainty threshold for evidential presence verifiers. Negative disables it.",
    )
    parser.add_argument(
        "--presence-refresh-streak",
        type=int,
        default=2,
        help="Require this many low-presence frames before forcing detector refresh.",
    )
    parser.add_argument("--class-names", default="", help="Comma-separated classes or a newline-delimited class-name file.")
    parser.add_argument("--target-class-names", default="", help="Optional allowlist of detector classes.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detector confidence threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="Detection NMS IoU threshold.")
    parser.add_argument(
        "--detector-pre-nms-topk",
        type=int,
        default=300,
        help="Keep only top-K detector candidates before NMS; 0 disables the cap.",
    )
    parser.add_argument(
        "--detector-max-det",
        type=int,
        default=128,
        help="Maximum detections kept after detector NMS; 0 disables the cap.",
    )
    parser.add_argument(
        "--detector-postprocess-backend",
        default="auto",
        choices=("auto", "python", "cpp"),
        help="Postprocess backend for RKOPT YOLOv8 multi-output models.",
    )
    parser.add_argument(
        "--detector-postprocess-lib",
        default="",
        help="Optional rk_yolov8_postprocess shared library path for --detector-postprocess-backend cpp/auto.",
    )
    parser.add_argument(
        "--detector-assist-policy",
        default="edtc_like",
        choices=("granular", "edtc_like"),
        help="Detector-assisted tracking policy. Defaults to edtc_like on RK3588.",
    )
    parser.add_argument("--detect-interval", type=int, default=2, help="Run detector every N frames while tracking.")
    parser.add_argument("--max-lost", type=int, default=30, help="Frames to wait before dropping a lost target.")
    parser.add_argument("--tracker-score-thresh", type=float, default=0.35, help="Tracker confidence threshold.")
    parser.add_argument("--min-confirm-detections", type=int, default=2, help="Detector-backed hits before confirmation.")
    parser.add_argument("--disable-roi-redetect", action="store_true", help="Disable ROI re-detection.")
    parser.add_argument("--no-manual-confirmation", action="store_true", help="Auto-confirm alerts after a short warmup.")
    parser.add_argument("--save-output", default="", help="Annotated output image/video path.")
    parser.add_argument("--state-log", default="", help="Optional JSONL state log path.")
    parser.add_argument("--alert-log", default="", help="Optional JSONL alert log path.")
    parser.add_argument("--alert-crops", default="", help="Optional directory for confirmed alert crops.")
    parser.add_argument("--repeat-image", type=int, default=1, help="Repeat a single image N times to smoke-test tracking.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap, 0 means unlimited.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Warmup frames excluded from FPS statistics.")
    parser.add_argument("--benchmark-json", default="", help="Optional JSON summary path for end-to-end FPS.")
    parser.add_argument("--disable-annotate", action="store_true", help="Skip drawing overlays to reduce benchmark overhead.")
    parser.add_argument("--show", action="store_true", help="Show annotated frames.")
    return parser.parse_args()


def resolve_default_presence_model() -> Path | None:
    """Find the current default pair-head presence verifier checkpoint."""
    candidates: list[Path] = []
    env_path = os.getenv(DEFAULT_PRESENCE_MODEL_ENV, "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    runs_root = ROOT / "runs" / "anti_uav"
    preferred = runs_root / "presence_pair_trainonly24_a52f825_model" / DEFAULT_PRESENCE_MODEL_NAME
    candidates.append(preferred)
    candidates.append(Path("/home/orangepi/pair_presence_edl.rknn"))
    candidates.append(Path("/home/orangepi/pair_presence_edl.pt"))
    candidates.append(Path("/home/orangepi/ultralytics_yolov8_min/runs/anti_uav/presence_pair_trainonly24_a52f825_model") / DEFAULT_PRESENCE_MODEL_NAME)
    if runs_root.exists():
        dynamic = sorted(
            runs_root.glob(f"presence_pair*_model/{DEFAULT_PRESENCE_MODEL_NAME}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(dynamic)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def torch_is_available() -> bool:
    """Return whether torch is importable in the current Python environment."""
    return importlib.util.find_spec("torch") is not None


class TimingAccumulator:
    """Collect coarse component timings for board-side benchmarking."""

    def __init__(self):
        self.totals_ms = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, key: str, value_ms: float) -> None:
        self.totals_ms[key] += float(value_ms)
        self.counts[key] += 1

    def total(self, key: str) -> float:
        return float(self.totals_ms.get(key, 0.0))

    def count(self, key: str) -> int:
        return int(self.counts.get(key, 0))


def timer_add(timer: Optional[TimingAccumulator], key: str, start: float) -> None:
    if timer is not None:
        timer.add(key, (perf_counter() - start) * 1000.0)


def parse_hw(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected H,W, got: {value}")
    return parts[0], parts[1]


def parse_name_list(value: str, default: Optional[Sequence[str]] = None) -> Optional[list[str]]:
    if not value:
        return list(default) if default is not None else None
    path = Path(value).expanduser()
    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [part.strip() for part in value.split(",") if part.strip()]


def load_anti_uav_module():
    module_path = ROOT / "ultralytics" / "solutions" / "anti_uav.py"
    module_name = "ultralytics_solutions_anti_uav_local"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load anti_uav module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_nanotrack_root(raw: str) -> Path:
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.append(Path("/data/codes/NanoTrack_RK3588_python"))
    candidates.append(ROOT / "third_party" / "NanoTrack_RK3588_python")
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "models" / "rknnlite_rk3588_tracker.py").exists():
            return resolved
    raise FileNotFoundError(
        "NanoTrack RK3588 runtime checkout not found. Pass --nanotrack-root or place it under /data/codes/NanoTrack_RK3588_python."
    )


def letterbox(image: np.ndarray, new_shape: tuple[int, int], pad_color: tuple[int, int, int] = (0, 0, 0)):
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return image, ratio, dw, dh


class LetterboxRGBPreprocessor:
    """Reusable BGR->letterboxed-RGB preprocessor for RKNN detector input."""

    def __init__(self, input_hw: tuple[int, int], pad_color: tuple[int, int, int] = (0, 0, 0)):
        self.input_hw = input_hw
        self.pad_color = pad_color
        self.canvas = np.empty((input_hw[0], input_hw[1], 3), dtype=np.uint8)
        self._resize_bgr_cache: dict[tuple[int, int], np.ndarray] = {}

    def _get_resize_buffer(self, height: int, width: int) -> np.ndarray:
        key = (height, width)
        bgr = self._resize_bgr_cache.get(key)
        if bgr is None:
            bgr = np.empty((height, width, 3), dtype=np.uint8)
            self._resize_bgr_cache[key] = bgr
        return bgr

    def __call__(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        source_h, source_w = image_bgr.shape[:2]
        target_h, target_w = self.input_hw
        ratio = min(target_h / source_h, target_w / source_w)
        resized_w = int(round(source_w * ratio))
        resized_h = int(round(source_h * ratio))
        dw = (target_w - resized_w) / 2.0
        dh = (target_h - resized_h) / 2.0
        left = int(round(dw - 0.1))
        top = int(round(dh - 0.1))

        if self.pad_color == (0, 0, 0):
            self.canvas.fill(0)
        else:
            self.canvas[:, :] = self.pad_color

        if source_w == resized_w and source_h == resized_h:
            resized_bgr = image_bgr
        else:
            resized_bgr = self._get_resize_buffer(resized_h, resized_w)
            cv2.resize(image_bgr, (resized_w, resized_h), dst=resized_bgr, interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(
            resized_bgr,
            cv2.COLOR_BGR2RGB,
            dst=self.canvas[top : top + resized_h, left : left + resized_w],
        )
        return self.canvas, ratio, dw, dh


def undo_letterbox(boxes: np.ndarray, ratio: float, dw: float, dh: float, image_shape: tuple[int, int]) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, [0, 2]] /= max(ratio, 1e-6)
    boxes[:, [1, 3]] /= max(ratio, 1e-6)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_shape[0])
    return boxes


def softmax_numpy(values: np.ndarray, axis: int) -> np.ndarray:
    values = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def ensure_nchw(tensor: np.ndarray) -> np.ndarray:
    """Normalize RKNN/ONNX outputs to NCHW without assuming RKNN keeps ONNX output format."""
    tensor = np.asarray(tensor)
    if tensor.ndim == 3:
        tensor = tensor[None]
    if tensor.ndim != 4:
        raise ValueError(f"Expected rank-4 RK-optimized YOLOv8 output, got {tensor.shape}")

    channels_first = tensor.shape[1]
    channels_last = tensor.shape[-1]
    if channels_first in {1, 4, 64} or 1 <= channels_first <= 256 and channels_first < tensor.shape[2]:
        return np.ascontiguousarray(tensor)
    if channels_last in {1, 4, 64} or 1 <= channels_last <= 256 and channels_last < tensor.shape[1]:
        return np.ascontiguousarray(tensor.transpose(0, 3, 1, 2))
    return np.ascontiguousarray(tensor)


def dfl_numpy(position: np.ndarray) -> np.ndarray:
    n, channels, grid_h, grid_w = position.shape
    bins = channels // 4
    if bins <= 0:
        raise ValueError(f"Invalid DFL channel count for YOLOv8 bbox output: {position.shape}")
    values = position.reshape(n, 4, bins, grid_h, grid_w)
    values = softmax_numpy(values, axis=2)
    acc = np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1)
    return (values * acc).sum(axis=2)


def box_process(position: np.ndarray, input_hw: tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    grid = np.concatenate((col.reshape(1, 1, grid_h, grid_w), row.reshape(1, 1, grid_h, grid_w)), axis=1)
    stride = np.array([input_hw[1] // grid_h, input_hw[0] // grid_w], dtype=np.float32).reshape(1, 2, 1, 1)
    position = dfl_numpy(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def dfl_selected_numpy(logits: np.ndarray) -> np.ndarray:
    """Decode DFL logits for selected grid cells only."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError(f"Expected selected DFL logits as C,K, got {logits.shape}")
    channels, count = logits.shape
    bins = channels // 4
    if bins <= 0:
        raise ValueError(f"Invalid DFL channel count for YOLOv8 bbox output: {logits.shape}")
    values = logits.reshape(4, bins, count)
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)
    probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    acc = np.arange(bins, dtype=np.float32).reshape(1, bins, 1)
    return (probs * acc).sum(axis=1)


def flatten_branch(branch: np.ndarray) -> np.ndarray:
    channels = branch.shape[1]
    branch = branch.transpose(0, 2, 3, 1)
    return branch.reshape(-1, channels)


def group_model_zoo_outputs(outputs: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Group RKNN model-zoo YOLOv8 outputs as (bbox_logits, class_conf, score_sum)."""
    tensors = [ensure_nchw(np.asarray(output)) for output in outputs]
    if len(tensors) not in {6, 9}:
        raise ValueError(f"Unsupported RKNN/optimized ONNX output layout with {len(tensors)} tensors")

    branches = 3
    pair_per_branch = len(tensors) // branches

    # Normal export order is branch-major: bbox, class, optional score_sum for each scale.
    branch_major: list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    branch_major_valid = True
    for branch_index in range(branches):
        start = pair_per_branch * branch_index
        box = tensors[start]
        cls = tensors[start + 1]
        score_sum = tensors[start + 2] if pair_per_branch == 3 else None
        same_spatial = box.shape[2:] == cls.shape[2:] and (score_sum is None or box.shape[2:] == score_sum.shape[2:])
        looks_like_box = box.shape[1] % 4 == 0 and box.shape[1] >= 4
        if not same_spatial or not looks_like_box:
            branch_major_valid = False
            break
        branch_major.append((box, cls, score_sum))
    if branch_major_valid:
        return branch_major

    # Fallback for runtimes that reorder output tensors. For single-class YOLO the class and
    # score_sum tensors are both 1-channel, so the original order inside a scale is preserved.
    by_spatial: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for index, tensor in enumerate(tensors):
        by_spatial[(tensor.shape[2], tensor.shape[3])].append((index, tensor))
    if len(by_spatial) != branches:
        raise ValueError(f"Unable to group RK-optimized YOLOv8 outputs by scale: {[t.shape for t in tensors]}")

    grouped: list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    for spatial, items in sorted(by_spatial.items(), key=lambda item: item[0][0] * item[0][1], reverse=True):
        ordered = [tensor for _, tensor in sorted(items, key=lambda item: item[0])]
        box_candidates = [tensor for tensor in ordered if tensor.shape[1] % 4 == 0 and tensor.shape[1] > 4]
        if not box_candidates:
            raise ValueError(f"Scale {spatial} has no bbox-logit tensor: {[tensor.shape for tensor in ordered]}")
        box = box_candidates[0]
        rest = [tensor for tensor in ordered if tensor is not box]
        if not rest:
            raise ValueError(f"Scale {spatial} has no class tensor: {[tensor.shape for tensor in ordered]}")
        class_candidates = [tensor for tensor in rest if tensor.shape[1] != 1]
        cls = class_candidates[0] if class_candidates else rest[0]
        score_candidates = [tensor for tensor in rest if tensor is not cls and tensor.shape[1] == 1]
        score_sum = score_candidates[0] if score_candidates else None
        grouped.append((box, cls, score_sum))
    return grouped


def decode_model_zoo_outputs(outputs: Sequence[np.ndarray], input_hw: tuple[int, int], conf_thresh: float):
    """Decode RKOPT YOLOv8 outputs using model-zoo-style early grid filtering."""
    decoded_boxes: list[np.ndarray] = []
    decoded_class_ids: list[np.ndarray] = []
    decoded_scores: list[np.ndarray] = []
    for box_logits, class_conf, score_sum in group_model_zoo_outputs(outputs):
        box_logits = box_logits.astype(np.float32, copy=False)
        class_conf = class_conf.astype(np.float32, copy=False)
        grid_h, grid_w = box_logits.shape[2:4]
        class_map = class_conf[0]
        if class_map.shape[0] == 1:
            class_ids_map = np.zeros((grid_h, grid_w), dtype=np.int32)
            class_scores_map = class_map[0]
        else:
            class_ids_map = np.argmax(class_map, axis=0).astype(np.int32)
            class_scores_map = np.take_along_axis(class_map, class_ids_map[None], axis=0)[0]
        if score_sum is None:
            score_sum_map = np.ones((grid_h, grid_w), dtype=np.float32)
        else:
            score_sum_map = score_sum[0, 0].astype(np.float32, copy=False)

        # Match rknn_model_zoo C postprocess: score_sum is a fast pre-filter,
        # while final confidence remains max class confidence. For our
        # single-class detector, multiplying them would square the score.
        keep = (score_sum_map >= conf_thresh) & (class_scores_map >= conf_thresh)
        if not np.any(keep):
            continue

        ys, xs = np.nonzero(keep)
        logits = box_logits[0, :, ys, xs]
        if logits.shape[0] != box_logits.shape[1]:
            logits = logits.T
        distances = dfl_selected_numpy(logits)
        stride_x = float(input_hw[1]) / float(grid_w)
        stride_y = float(input_hw[0]) / float(grid_h)
        xs_f = xs.astype(np.float32) + 0.5
        ys_f = ys.astype(np.float32) + 0.5
        boxes = np.stack(
            (
                (xs_f - distances[0]) * stride_x,
                (ys_f - distances[1]) * stride_y,
                (xs_f + distances[2]) * stride_x,
                (ys_f + distances[3]) * stride_y,
            ),
            axis=1,
        ).astype(np.float32)
        decoded_boxes.append(boxes)
        decoded_class_ids.append(class_ids_map[ys, xs].astype(np.int32, copy=False))
        decoded_scores.append(class_scores_map[ys, xs].astype(np.float32, copy=False))

    if not decoded_boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)
    return np.concatenate(decoded_boxes, axis=0), np.concatenate(decoded_class_ids, axis=0), np.concatenate(decoded_scores, axis=0)


def decode_ultralytics_output(output: np.ndarray, conf_thresh: float):
    output = np.asarray(output)
    if output.ndim != 3:
        raise ValueError(f"Unsupported Ultralytics ONNX output rank: {output.shape}")
    output = output[0]
    if output.shape[0] < output.shape[1]:
        output = output.transpose(1, 0)
    if output.shape[1] < 5:
        raise ValueError(f"Unsupported Ultralytics ONNX output shape: {output.shape}")

    boxes_xywh = output[:, :4]
    class_scores = output[:, 4:]
    if class_scores.size == 0:
        class_ids = np.zeros((boxes_xywh.shape[0],), dtype=np.int32)
        scores = np.ones((boxes_xywh.shape[0],), dtype=np.float32)
    else:
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = np.max(class_scores, axis=1).astype(np.float32)

    keep = scores >= conf_thresh
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)

    boxes_xywh = boxes_xywh[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]

    cx, cy, width, height = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    boxes = np.stack((cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0), axis=1)
    return boxes.astype(np.float32), class_ids, scores


def bbox_iou_vector(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, 0] * wh[:, 1]
    box_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return inter / np.clip(box_area + areas - inter, 1e-6, None)


def nms_boxes_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        ious = bbox_iou_vector(boxes[index], boxes[order[1:]])
        order = order[1:][ious <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def apply_classwise_nms_numpy(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float,
) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        local_keep = nms_boxes_numpy(boxes[indices], scores[indices], iou_thresh)
        kept.extend(indices[local_keep].tolist())
    return np.asarray(sorted(kept, key=lambda idx: float(scores[idx]), reverse=True), dtype=np.int64)


def resolve_cpp_postprocess_lib(raw: str = "") -> Path | None:
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            ROOT / "build" / "rk_yolov8_postprocess.so",
            ROOT / "scripts" / "anti_uav" / "rk_yolov8_postprocess.so",
            Path("/home/orangepi/ultralytics_yolov8/build/rk_yolov8_postprocess.so"),
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


class CppYoloV8PostProcessor:
    """ctypes wrapper around the lightweight C++ RKOPT YOLOv8 postprocess."""

    def __init__(self, library_path: Path):
        self.library_path = library_path.expanduser().resolve()
        self.library = ctypes.CDLL(str(self.library_path))
        self.function = self.library.rk_yolov8_postprocess_float
        self.function.restype = ctypes.c_int

    @staticmethod
    def _void_ptr(array: np.ndarray | None) -> ctypes.c_void_p:
        if array is None:
            return ctypes.c_void_p(0)
        return ctypes.c_void_p(int(array.ctypes.data))

    def __call__(
        self,
        branches: Sequence[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        input_hw: tuple[int, int],
        conf_thresh: float,
        nms_iou: float,
        max_det: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(branches) != 3:
            raise ValueError(f"C++ YOLOv8 postprocess expects 3 branches, got {len(branches)}")

        arrays = []
        args = []
        for box, cls, score_sum in branches:
            box_arr = np.ascontiguousarray(box[0], dtype=np.float32)
            cls_arr = np.ascontiguousarray(cls[0], dtype=np.float32)
            score_arr = None if score_sum is None else np.ascontiguousarray(score_sum[0, 0], dtype=np.float32)
            arrays.extend([box_arr, cls_arr, score_arr])
            args.extend(
                [
                    self._void_ptr(box_arr),
                    self._void_ptr(cls_arr),
                    self._void_ptr(score_arr),
                    ctypes.c_int(int(box_arr.shape[0])),
                    ctypes.c_int(int(cls_arr.shape[0])),
                    ctypes.c_int(int(box_arr.shape[1])),
                    ctypes.c_int(int(box_arr.shape[2])),
                ]
            )

        cap = int(max_det) if max_det > 0 else 512
        out_boxes = np.empty((cap, 4), dtype=np.float32)
        out_classes = np.empty((cap,), dtype=np.int32)
        out_scores = np.empty((cap,), dtype=np.float32)
        count = self.function(
            *args,
            ctypes.c_int(int(input_hw[0])),
            ctypes.c_int(int(input_hw[1])),
            ctypes.c_float(float(conf_thresh)),
            ctypes.c_float(float(nms_iou)),
            ctypes.c_int(int(max_det)),
            self._void_ptr(out_boxes),
            self._void_ptr(out_classes),
            self._void_ptr(out_scores),
        )
        if count < 0:
            raise RuntimeError(f"C++ YOLOv8 postprocess failed with code {count}")
        count = min(int(count), cap)
        return out_boxes[:count].copy(), out_classes[:count].copy(), out_scores[:count].copy()


class YoloBoardBackend:
    """Small detector wrapper for RKNN or ONNXRuntime."""

    def __init__(
        self,
        model_path: Path,
        input_hw: tuple[int, int],
        conf_thresh: float,
        *,
        nms_iou: float = 0.45,
        max_det: int = 128,
        postprocess_backend: str = "auto",
        postprocess_lib: str = "",
        timer: Optional[TimingAccumulator] = None,
    ):
        self.model_path = model_path
        self.input_hw = input_hw
        self.conf_thresh = conf_thresh
        self.nms_iou = float(nms_iou)
        self.max_det = int(max_det)
        self.timer = timer
        self.preprocessor = LetterboxRGBPreprocessor(input_hw, pad_color=(0, 0, 0))
        self.postprocess_backend = postprocess_backend
        self.cpp_postprocessor: CppYoloV8PostProcessor | None = None
        if postprocess_backend in {"auto", "cpp"}:
            cpp_lib = resolve_cpp_postprocess_lib(postprocess_lib)
            if cpp_lib is not None:
                self.cpp_postprocessor = CppYoloV8PostProcessor(cpp_lib)
            elif postprocess_backend == "cpp":
                raise FileNotFoundError(
                    "C++ detector postprocess requested, but rk_yolov8_postprocess.so was not found. "
                    "Run scripts/anti_uav/build_rk_yolov8_postprocess.sh on the target board or pass --detector-postprocess-lib."
                )
        self.kind = model_path.suffix.lower()
        self.rknn_input_mode: Optional[str] = None
        if self.kind == ".rknn":
            try:
                from rknnlite.api import RKNNLite

                runtime = RKNNLite()
                runtime_name = "rknnlite"
            except ImportError:
                from rknn.api import RKNN

                runtime = RKNN()
                runtime_name = "rknn"
            ret = runtime.load_rknn(str(model_path))
            if ret != 0:
                raise RuntimeError(f"load_rknn failed with code {ret}")
            ret = runtime.init_runtime()
            if ret != 0:
                raise RuntimeError(f"init_runtime failed with code {ret}")
            self.runtime = runtime
            self.runtime_name = runtime_name
            self.input_name = None
        elif self.kind == ".onnx":
            import onnxruntime as ort

            self.runtime = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.runtime_name = "onnxruntime"
            self.input_name = self.runtime.get_inputs()[0].name
        else:
            raise ValueError(f"Unsupported model suffix: {model_path.suffix}. Use .rknn or .onnx")

    def _infer_rknn(self, rgb: np.ndarray):
        candidates = {
            "nhwc_batch": (np.ascontiguousarray(rgb[None]), "nhwc"),
            "nchw_batch": (np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), "nchw"),
            "raw_hwc": (np.ascontiguousarray(rgb), None),
        }
        order = [self.rknn_input_mode] if self.rknn_input_mode else []
        order.extend(mode for mode in candidates if mode not in order)

        last_error: Optional[Exception] = None
        for mode in order:
            tensor, data_format = candidates[mode]
            kwargs = {"inputs": [tensor]}
            if data_format is not None:
                kwargs["data_format"] = [data_format]
            try:
                outputs = self.runtime.inference(**kwargs)
            except TypeError:
                kwargs.pop("data_format", None)
                try:
                    outputs = self.runtime.inference(**kwargs)
                except Exception as exc:  # pragma: no cover - board-only fallback
                    last_error = exc
                    continue
            except Exception as exc:  # pragma: no cover - board-only fallback
                last_error = exc
                continue

            if outputs is not None:
                self.rknn_input_mode = mode
                return outputs

        raise RuntimeError(f"RKNN inference failed for all input layouts, last error: {last_error}")

    def infer(self, image_bgr: np.ndarray):
        start = perf_counter()
        rgb, ratio, dw, dh = self.preprocessor(image_bgr)
        timer_add(self.timer, "detector_preprocess_letterbox_ms", start)
        start = perf_counter()
        if self.kind == ".rknn":
            outputs = self._infer_rknn(rgb)
        else:
            tensor = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
            outputs = self.runtime.run(None, {self.input_name: tensor})
        timer_add(self.timer, "detector_inference_ms", start)

        start = perf_counter()
        outputs = [np.asarray(output) for output in outputs]
        timer_add(self.timer, "detector_output_array_ms", start)
        start = perf_counter()
        if len(outputs) == 1:
            boxes, class_ids, scores = decode_ultralytics_output(outputs[0], self.conf_thresh)
        elif self.cpp_postprocessor is not None:
            branches = group_model_zoo_outputs(outputs)
            boxes, class_ids, scores = self.cpp_postprocessor(
                branches,
                self.input_hw,
                self.conf_thresh,
                self.nms_iou,
                self.max_det,
            )
        else:
            boxes, class_ids, scores = decode_model_zoo_outputs(outputs, self.input_hw, self.conf_thresh)
        timer_add(self.timer, "detector_decode_ms", start)
        start = perf_counter()
        boxes = undo_letterbox(boxes, ratio, dw, dh, image_bgr.shape[:2])
        timer_add(self.timer, "detector_undo_letterbox_ms", start)
        return boxes, class_ids, scores

    def release(self) -> None:
        release = getattr(self.runtime, "release", None)
        if callable(release):
            release()


class BoardYoloDetectionAdapter:
    """Detection adapter with ROI re-detection compatible with AntiUAVSystem."""

    def __init__(
        self,
        anti_uav_module,
        model_path: Path,
        input_hw: tuple[int, int],
        class_names: Optional[Sequence[str]],
        target_class_names: Optional[Iterable[str]],
        conf: float,
        nms_iou: float,
        pre_nms_topk: int = 300,
        max_det: int = 128,
        postprocess_backend: str = "auto",
        postprocess_lib: str = "",
        enable_roi: bool = True,
        roi_expand: float = 2.5,
        timer: Optional[TimingAccumulator] = None,
    ):
        self.anti_uav = anti_uav_module
        self.backend = YoloBoardBackend(
            model_path,
            input_hw=input_hw,
            conf_thresh=conf,
            nms_iou=nms_iou,
            max_det=max_det,
            postprocess_backend=postprocess_backend,
            postprocess_lib=postprocess_lib,
            timer=timer,
        )
        self.class_names = list(class_names or [])
        self.target_class_names = {name.lower() for name in target_class_names} if target_class_names else None
        self.nms_iou = nms_iou
        self.pre_nms_topk = int(pre_nms_topk)
        self.max_det = int(max_det)
        self.enable_roi = enable_roi
        self.roi_expand = roi_expand
        self.timer = timer
        self.filters = [
            anti_uav_module.AreaFilter(min_area_px=16),
            anti_uav_module.AspectRatioFilter(min_ratio=0.25, max_ratio=4.0),
            anti_uav_module.BorderFilter(margin_px=6),
        ]

    def __call__(self, frame: np.ndarray):
        return self.detect(frame)

    def detect(self, frame: np.ndarray, roi: Optional[Sequence[float]] = None, prefer_roi: bool = False):
        candidates = []
        if roi is not None and self.enable_roi:
            roi_box = self.anti_uav._expand_bbox(roi, frame.shape, self.roi_expand)
            roi_candidates = self._predict_crop(frame, roi_box, "roi")
            candidates.append(roi_candidates)
            if prefer_roi and roi_candidates[2].size:
                return self._postprocess(frame, candidates)
        full_box = (0.0, 0.0, float(frame.shape[1]), float(frame.shape[0]))
        candidates.append(self._predict_crop(frame, full_box, "full_frame"))
        return self._postprocess(frame, candidates)

    def _predict_crop(self, frame: np.ndarray, crop_box: Sequence[float], source: str):
        x1, y1, x2, y2 = [int(value) for value in self.anti_uav._clip_bbox(crop_box, frame.shape)]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=object),
            )
        boxes, class_ids, scores = self.backend.infer(crop)
        if boxes.size:
            boxes = boxes.copy()
            boxes[:, [0, 2]] += float(x1)
            boxes[:, [1, 3]] += float(y1)
        sources = np.full((scores.shape[0],), source, dtype=object)
        return boxes, class_ids, scores, sources

    def _postprocess(self, frame: np.ndarray, detections):
        start = perf_counter()
        non_empty = [item for item in detections if item[2].size]
        if not non_empty:
            timer_add(self.timer, "detector_array_pack_ms", start)
            return []
        boxes = np.concatenate([item[0] for item in non_empty], axis=0)
        class_ids = np.concatenate([item[1] for item in non_empty], axis=0)
        scores = np.concatenate([item[2] for item in non_empty], axis=0)
        sources = np.concatenate([item[3] for item in non_empty], axis=0)

        if self.target_class_names:
            target_keep = []
            for class_id in class_ids:
                class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"class_{class_id}"
                target_keep.append(class_name.lower() in self.target_class_names)
            keep_mask = np.asarray(target_keep, dtype=bool)
            boxes, class_ids, scores, sources = boxes[keep_mask], class_ids[keep_mask], scores[keep_mask], sources[keep_mask]
            if not scores.size:
                timer_add(self.timer, "detector_array_pack_ms", start)
                return []
        timer_add(self.timer, "detector_array_pack_ms", start)

        start = perf_counter()
        if self.pre_nms_topk > 0 and scores.shape[0] > self.pre_nms_topk:
            keep = np.argpartition(scores, -self.pre_nms_topk)[-self.pre_nms_topk :]
            boxes, class_ids, scores, sources = boxes[keep], class_ids[keep], scores[keep], sources[keep]
        timer_add(self.timer, "detector_pre_nms_topk_ms", start)

        start = perf_counter()
        keep = apply_classwise_nms_numpy(boxes, class_ids, scores, self.nms_iou)
        if self.max_det > 0 and keep.shape[0] > self.max_det:
            keep = keep[: self.max_det]
        boxes, class_ids, scores, sources = boxes[keep], class_ids[keep], scores[keep], sources[keep]
        timer_add(self.timer, "detector_nms_ms", start)

        start = perf_counter()
        kept = []
        for box, class_id, score, source in zip(boxes, class_ids, scores, sources):
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"class_{class_id}"
            detection = self.anti_uav.Detection(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(score),
                class_id=int(class_id),
                class_name=class_name,
                source=str(source),
            )
            if all(rule.keep(detection, frame) for rule in self.filters):
                kept.append(detection)
        timer_add(self.timer, "detector_object_filter_ms", start)
        return kept

    def release(self) -> None:
        self.backend.release()


class TimedDetectorAdapter:
    """Wrap the board detector and collect per-call timings."""

    def __init__(self, detector, timer: TimingAccumulator):
        self.detector = detector
        self.timer = timer

    def __call__(self, frame: np.ndarray):
        return self.detect(frame)

    def detect(self, frame: np.ndarray, roi: Optional[Sequence[float]] = None, prefer_roi: bool = False):
        start = perf_counter()
        outputs = self.detector.detect(frame, roi=roi, prefer_roi=prefer_roi)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("detector_total_ms", elapsed_ms)
        self.timer.add("detector_call_ms", elapsed_ms)
        if roi is None:
            self.timer.add("detector_full_frame_ms", elapsed_ms)
        else:
            self.timer.add("detector_roi_ms", elapsed_ms)
        return outputs

    def release(self) -> None:
        self.detector.release()

    def __getattr__(self, name):
        return getattr(self.detector, name)


class NanoTrackRKNNLiteTracker:
    """Wrap the upstream NanoTrack RK3588 runtime with the AntiUAV tracker contract."""

    name = "nanotrack_rknn"

    def __init__(
        self,
        anti_uav_module,
        *,
        nanotrack_root: Path,
        config_path: Optional[Path] = None,
        tback_path: Optional[Path] = None,
        xback_path: Optional[Path] = None,
        head_path: Optional[Path] = None,
        score_threshold: float = 0.25,
    ):
        self.anti_uav = anti_uav_module
        self.nanotrack_root = nanotrack_root
        self.config_path = (config_path or nanotrack_root / "models" / "config" / "config.yaml").expanduser().resolve()
        self.tback_path = (tback_path or nanotrack_root / "weights" / "track_backbone_T.rknn").expanduser().resolve()
        self.xback_path = (xback_path or nanotrack_root / "weights" / "track_backbone_X.rknn").expanduser().resolve()
        self.head_path = (head_path or nanotrack_root / "weights" / "head.rknn").expanduser().resolve()
        self.score_threshold = float(score_threshold)
        self.initialized = False
        self._tracker = None
        self._load_modules()
        self._tracker = self._build_tracker()
        self.reset()

    def _load_modules(self) -> None:
        if str(self.nanotrack_root) not in sys.path:
            sys.path.insert(0, str(self.nanotrack_root))
        from core.config import cfg
        from models.rknnlite_rk3588_tracker import NnoTracker_RKNNLite

        self._cfg = cfg
        self._tracker_class = NnoTracker_RKNNLite

    def _build_tracker(self):
        self._cfg.merge_from_file(str(self.config_path))
        return self._tracker_class(str(self.tback_path), str(self.xback_path), str(self.head_path))

    def _rebuild_tracker(self) -> None:
        self._release_tracker_instance(self._tracker)
        self._tracker = self._build_tracker()

    def _clear_tracker_state(self) -> None:
        self.initialized = False
        if self._tracker is None:
            return
        for attr_name in ("center_pos", "size", "channel_average", "Toutput", "Xoutput"):
            if hasattr(self._tracker, attr_name):
                setattr(self._tracker, attr_name, None)

    @staticmethod
    def _release_tracker_instance(tracker) -> None:
        if tracker is None:
            return
        for attr_name in ("rknn_Tback", "rknn_Xback", "rknn_Head"):
            runtime = getattr(tracker, attr_name, None)
            release = getattr(runtime, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass

    def reset(self) -> None:
        self._clear_tracker_state()

    def _template_init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        if self._tracker is None:
            self._tracker = self._build_tracker()
        xyxy = self.anti_uav._clip_bbox(bbox, frame.shape)
        x1, y1, x2, y2 = xyxy
        width = max(float(x2 - x1), 10.0)
        height = max(float(y2 - y1), 10.0)
        tracker = self._tracker
        tracker.center_pos = np.array([x1 + (width - 1.0) / 2.0, y1 + (height - 1.0) / 2.0], dtype=np.float32)
        tracker.size = np.array([width, height], dtype=np.float32)

        w_z = tracker.size[0] + self._cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        h_z = tracker.size[1] + self._cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        s_z = round(np.sqrt(w_z * h_z))
        tracker.channel_average = np.mean(frame, axis=(0, 1))
        z_crop = tracker.get_subwindow(
            frame,
            tracker.center_pos,
            self._cfg.TRACK.EXEMPLAR_SIZE,
            s_z,
            tracker.channel_average,
        )
        back_T_in = z_crop.transpose((0, 2, 3, 1))
        tracker.Toutput = tracker.rknn_Tback.inference(inputs=[back_T_in])
        self.initialized = True

    def _apply_runtime_bbox(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        x1, y1, x2, y2 = self.anti_uav._clip_bbox(bbox, frame.shape)
        width = max(float(x2 - x1), 10.0)
        height = max(float(y2 - y1), 10.0)
        self._tracker.center_pos = np.array([x1 + (width - 1.0) / 2.0, y1 + (height - 1.0) / 2.0], dtype=np.float32)
        self._tracker.size = np.array([width, height], dtype=np.float32)
        self._tracker.channel_average = np.mean(frame, axis=(0, 1))
        self.initialized = True

    def init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        try:
            self._template_init(frame, bbox)
        except Exception:
            self._rebuild_tracker()
            self._template_init(frame, bbox)

    def correct_bbox(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        self._apply_runtime_bbox(frame, bbox)

    def reinit_from_detection(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        try:
            self._clear_tracker_state()
            self._template_init(frame, bbox)
        except Exception:
            self._rebuild_tracker()
            self._template_init(frame, bbox)

    def update(self, frame: np.ndarray):
        if not self.initialized:
            return False, None, 0.0
        outputs = self._tracker.track(frame)
        bbox = outputs.get("bbox")
        if bbox is None:
            return False, None, 0.0
        x, y, width, height = [float(value) for value in bbox]
        clipped = self.anti_uav._clip_bbox((x, y, x + width, y + height), frame.shape)
        score = float(outputs.get("best_score", 0.0))
        return score >= self.score_threshold, clipped, score

    def release(self) -> None:
        self._release_tracker_instance(self._tracker)
        self._tracker = None
        self.initialized = False


class TimedTracker:
    """Wrap a tracker and collect timings for each operation."""

    def __init__(self, tracker, timer: TimingAccumulator):
        self.tracker = tracker
        self.timer = timer
        self.name = getattr(tracker, "name", tracker.__class__.__name__)

    def reset(self) -> None:
        start = perf_counter()
        self.tracker.reset()
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("tracker_total_ms", elapsed_ms)
        self.timer.add("tracker_reset_ms", elapsed_ms)

    def init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        start = perf_counter()
        self.tracker.init(frame, bbox)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("tracker_total_ms", elapsed_ms)
        self.timer.add("tracker_init_ms", elapsed_ms)

    def correct_bbox(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        start = perf_counter()
        self.tracker.correct_bbox(frame, bbox)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("tracker_total_ms", elapsed_ms)
        self.timer.add("tracker_correct_ms", elapsed_ms)

    def reinit_from_detection(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        start = perf_counter()
        self.tracker.reinit_from_detection(frame, bbox)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("tracker_total_ms", elapsed_ms)
        self.timer.add("tracker_reinit_ms", elapsed_ms)

    def update(self, frame: np.ndarray):
        start = perf_counter()
        outputs = self.tracker.update(frame)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("tracker_total_ms", elapsed_ms)
        self.timer.add("tracker_update_ms", elapsed_ms)
        return outputs

    def release(self) -> None:
        release = getattr(self.tracker, "release", None)
        if callable(release):
            start = perf_counter()
            release()
            elapsed_ms = (perf_counter() - start) * 1000.0
            self.timer.add("tracker_total_ms", elapsed_ms)
            self.timer.add("tracker_release_ms", elapsed_ms)

    def __getattr__(self, name):
        return getattr(self.tracker, name)


class RKNNPairPresenceVerifier:
    """RKNN runtime for the pair-head presence verifier on RK3588."""

    name = "pair_head_edl"

    def __init__(
        self,
        anti_uav_module,
        checkpoint_path: Path,
        metadata_path: Optional[Path] = None,
        *,
        patch_scale: float = 1.2,
    ):
        self.anti_uav = anti_uav_module
        self.model_path = checkpoint_path.expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Presence-verifier RKNN not found: {self.model_path}")
        self.metadata_path = (
            metadata_path.expanduser().resolve() if metadata_path else self.model_path.with_suffix(".json")
        )
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Presence-verifier metadata JSON not found: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.patch_scale = max(1.0, float(patch_scale))
        self.patch_size = int(self.metadata.get("patch_size", 64))
        self.feature_names = tuple(self.metadata.get("feature_names", anti_uav_module.PRESENCE_FEATURE_NAMES))
        self.use_metadata = bool(self.metadata.get("use_metadata", True))
        self.loss_mode = str(self.metadata.get("loss_mode", "ce")).lower()

        from rknnlite.api import RKNNLite

        self.runtime = RKNNLite()
        ret = self.runtime.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"Presence-verifier load_rknn failed with code {ret}")
        ret = self.runtime.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Presence-verifier init_runtime failed with code {ret}")

        self.reference_patch = None
        self.previous_patch = None
        self.previous_bbox = None
        self.last_features: dict[str, float] = {}

    def reset(self) -> None:
        self.reference_patch = None
        self.previous_patch = None
        self.previous_bbox = None
        self.last_features = {}

    def release(self) -> None:
        release = getattr(self.runtime, "release", None)
        if callable(release):
            release()

    def on_init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        clipped = self.anti_uav._clip_bbox(bbox, frame.shape)
        patch = self._extract_feature_patch(frame, clipped)
        self.reference_patch = patch
        self.previous_patch = patch
        self.previous_bbox = clipped

    def on_soft_correction(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        clipped = self.anti_uav._clip_bbox(bbox, frame.shape)
        self.previous_patch = self._extract_feature_patch(frame, clipped)
        self.previous_bbox = clipped

    def on_hard_reinit(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        self.on_init(frame, bbox)

    def _extract_feature_patch(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
        patch = self.anti_uav._extract_patch(frame, bbox, self.patch_scale)
        if patch.size == 0:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        resized = cv2.resize(gray, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32)
        std = float(normalized.std())
        if std < 1e-6:
            return normalized / 255.0
        return (normalized - float(normalized.mean())) / std

    def _feature_vector(self, features: dict[str, float]) -> np.ndarray:
        return np.asarray([float(features.get(name, 0.0)) for name in self.feature_names], dtype=np.float32)

    def evaluate(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        track_score: float,
        *,
        previous_bbox: Optional[Sequence[float]] = None,
        context: Optional[dict[str, float]] = None,
    ):
        context = context or {}
        clipped = self.anti_uav._clip_bbox(bbox, frame.shape)
        current_patch = self._extract_feature_patch(frame, clipped)
        reference_bbox = previous_bbox or self.previous_bbox or clipped
        features = {
            "track_score": float(np.clip(track_score, 0.0, 1.0)),
            "reference_similarity": self.anti_uav._patch_similarity(self.reference_patch, current_patch),
            "previous_similarity": self.anti_uav._patch_similarity(self.previous_patch, current_patch),
            "motion_ratio": float(min(self.anti_uav._bbox_center_distance_ratio(clipped, reference_bbox), 10.0) / 10.0),
            "area_change": float(min(self.anti_uav._bbox_area_change_ratio(clipped, reference_bbox), 10.0) / 10.0),
            "aspect_change": float(min(self.anti_uav._bbox_aspect_change_ratio(clipped, reference_bbox), 10.0) / 10.0),
            "edge_ratio": self.anti_uav._bbox_edge_ratio(clipped, frame.shape),
            "detection_gap": float(
                min(
                    float(context.get("frames_since_detection", 0.0))
                    / max(float(context.get("detect_interval", 1.0)) * 3.0, 1.0),
                    1.0,
                )
            ),
            "contradiction_signal": float(
                min(
                    float(context.get("detector_contradiction_streak", 0.0))
                    / max(float(context.get("detector_contradiction_consensus_frames", 1.0)), 1.0),
                    1.0,
                )
            ),
            "requires_refresh": float(bool(context.get("requires_detector_refresh", False))),
            "assist_active": float(bool(context.get("assist_active", False))),
        }
        for name in self.feature_names:
            features.setdefault(name, 0.0)
        self.previous_patch = current_patch
        self.previous_bbox = clipped
        self.last_features = features

        if self.reference_patch is None or current_patch is None:
            return self.anti_uav.PresenceEstimate(score=0.0, features=features, uncertainty=1.0)

        image_pair = np.stack([self.reference_patch, current_patch], axis=0).astype(np.float32, copy=False)[None]
        inputs = [image_pair]
        if self.use_metadata:
            metadata = self._feature_vector(features)[None]
            inputs.append(metadata.astype(np.float32, copy=False))
        outputs = self.runtime.inference(inputs=inputs)
        logits = np.asarray(outputs[0]).reshape(-1)
        if logits.size != 2:
            raise RuntimeError(f"Unexpected verifier logits shape: {np.asarray(outputs[0]).shape}")
        if self.loss_mode == "edl":
            evidence = np.log1p(np.exp(logits))
            alpha = evidence + 1.0
            total = float(np.sum(alpha))
            probability = float(alpha[1] / max(total, 1e-6))
            uncertainty = min(float(2.0 / max(total, 1e-6)), 1.0)
        else:
            shifted = logits - np.max(logits)
            probs = np.exp(shifted)
            probs /= np.sum(probs)
            probability = float(probs[1])
            uncertainty = float(1.0 - abs(probability - 0.5) * 2.0)
        return self.anti_uav.PresenceEstimate(
            score=float(np.clip(probability, 0.0, 1.0)),
            features=features,
            uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
        )


class TimedPresenceVerifier:
    """Wrap a presence verifier and collect timings."""

    def __init__(self, verifier, timer: TimingAccumulator):
        self.verifier = verifier
        self.timer = timer
        self.name = getattr(verifier, "name", verifier.__class__.__name__)
        self.last_features = {}

    def reset(self) -> None:
        start = perf_counter()
        self.verifier.reset()
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("presence_total_ms", elapsed_ms)
        self.timer.add("presence_reset_ms", elapsed_ms)
        self.last_features = dict(getattr(self.verifier, "last_features", {}))

    def on_init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        start = perf_counter()
        self.verifier.on_init(frame, bbox)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("presence_total_ms", elapsed_ms)
        self.timer.add("presence_init_ms", elapsed_ms)
        self.last_features = dict(getattr(self.verifier, "last_features", {}))

    def on_soft_correction(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        start = perf_counter()
        self.verifier.on_soft_correction(frame, bbox)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("presence_total_ms", elapsed_ms)
        self.timer.add("presence_soft_ms", elapsed_ms)
        self.last_features = dict(getattr(self.verifier, "last_features", {}))

    def on_hard_reinit(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        start = perf_counter()
        self.verifier.on_hard_reinit(frame, bbox)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("presence_total_ms", elapsed_ms)
        self.timer.add("presence_hard_ms", elapsed_ms)
        self.last_features = dict(getattr(self.verifier, "last_features", {}))

    def evaluate(self, frame: np.ndarray, bbox: Sequence[float], track_score: float, *, previous_bbox=None, context=None):
        start = perf_counter()
        estimate = self.verifier.evaluate(frame, bbox, track_score, previous_bbox=previous_bbox, context=context)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timer.add("presence_total_ms", elapsed_ms)
        self.timer.add("presence_eval_ms", elapsed_ms)
        self.last_features = dict(getattr(self.verifier, "last_features", {}))
        return estimate

    def release(self) -> None:
        release = getattr(self.verifier, "release", None)
        if callable(release):
            start = perf_counter()
            release()
            elapsed_ms = (perf_counter() - start) * 1000.0
            self.timer.add("presence_total_ms", elapsed_ms)
            self.timer.add("presence_release_ms", elapsed_ms)

    def __getattr__(self, name):
        return getattr(self.verifier, name)


def iter_frames(source: str, repeat_image: int = 1) -> Iterator[tuple[str, np.ndarray]]:
    path = Path(source).expanduser()
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open camera index: {source}")
        index = 0
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                yield f"frame_{index:06d}", frame
        finally:
            capture.release()
        return

    resolved = path.resolve()
    if resolved.is_dir():
        images = [item for item in sorted(resolved.iterdir()) if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is not None:
                yield image_path.name, frame
        return

    if resolved.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(str(resolved))
        if frame is None:
            raise RuntimeError(f"Unable to read image: {resolved}")
        for index in range(max(1, repeat_image)):
            yield f"{resolved.stem}_{index + 1:03d}{resolved.suffix}", frame.copy()
        return

    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {resolved}")
    index = 0
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            yield f"frame_{index:06d}", frame
    finally:
        capture.release()


def open_writer(path: Path, frame_shape: tuple[int, int, int], fps: float = 30.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frame_shape[:2]
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio))))
    return float(ordered[index])


def main() -> None:
    args = parse_args()
    anti_uav = load_anti_uav_module()
    timer = TimingAccumulator()
    model_path = Path(args.model).expanduser().resolve()
    input_hw = parse_hw(args.input_size)
    class_names = parse_name_list(args.class_names)
    target_class_names = parse_name_list(args.target_class_names)

    detector = TimedDetectorAdapter(
        BoardYoloDetectionAdapter(
            anti_uav,
            model_path=model_path,
            input_hw=input_hw,
            class_names=class_names,
            target_class_names=target_class_names,
            conf=args.conf,
            nms_iou=args.nms_iou,
            pre_nms_topk=args.detector_pre_nms_topk,
            max_det=args.detector_max_det,
            postprocess_backend=args.detector_postprocess_backend,
            postprocess_lib=args.detector_postprocess_lib,
            enable_roi=not args.disable_roi_redetect,
            timer=timer,
        ),
        timer,
    )

    if args.tracker == "template_match":
        tracker_impl = anti_uav.TemplateMatchTracker(score_threshold=args.tracker_score_thresh)
    else:
        nanotrack_root = resolve_nanotrack_root(args.nanotrack_root)
        tracker_impl = NanoTrackRKNNLiteTracker(
            anti_uav,
            nanotrack_root=nanotrack_root,
            config_path=Path(args.nanotrack_config).expanduser().resolve() if args.nanotrack_config else None,
            tback_path=Path(args.nanotrack_tback).expanduser().resolve() if args.nanotrack_tback else None,
            xback_path=Path(args.nanotrack_xback).expanduser().resolve() if args.nanotrack_xback else None,
            head_path=Path(args.nanotrack_head).expanduser().resolve() if args.nanotrack_head else None,
            score_threshold=args.tracker_score_thresh,
        )
    tracker = TimedTracker(tracker_impl, timer)

    verifier_name = args.presence_verifier.strip().lower()
    presence_model_path = Path(args.presence_model).expanduser().resolve() if args.presence_model else None
    presence_metadata_path = Path(args.presence_metadata).expanduser().resolve() if args.presence_metadata else None
    if not verifier_name:
        default_presence = resolve_default_presence_model()
        if default_presence is not None and (default_presence.suffix.lower() == ".rknn" or torch_is_available()):
            verifier_name = "pair_head_edl"
            presence_model_path = default_presence
        elif default_presence is not None:
            print("Torch is unavailable on this RK3588 environment; skipping default pair_head_edl verifier.")
    if verifier_name == "none":
        verifier_name = ""
        presence_model_path = None
        presence_metadata_path = None
    presence_uncertainty_thresh = (
        None if args.presence_uncertainty_thresh < 0 else float(args.presence_uncertainty_thresh)
    )
    if verifier_name in {"mlp", "pair_head", "pair_head_edl"} and presence_model_path is None:
        raise ValueError("--presence-model is required when --presence-verifier is mlp/pair_head/pair_head_edl.")

    presence_verifier = None
    if verifier_name:
        suffix = presence_model_path.suffix.lower() if presence_model_path is not None else ""
        if suffix == ".rknn" and verifier_name in {"pair_head", "pair_head_edl"}:
            presence_verifier = TimedPresenceVerifier(
                RKNNPairPresenceVerifier(
                    anti_uav,
                    checkpoint_path=presence_model_path,
                    metadata_path=presence_metadata_path,
                ),
                timer,
            )
        else:
            presence_verifier = anti_uav.build_presence_verifier(
                verifier_name,
                checkpoint_path=presence_model_path,
                device=args.presence_device or None,
            )
            presence_verifier = TimedPresenceVerifier(presence_verifier, timer)

    system = anti_uav.AntiUAVSystem(
        detector,
        tracker=tracker,
        presence_verifier=presence_verifier,
        presence_score_thresh=args.presence_score_thresh,
        presence_uncertainty_thresh=presence_uncertainty_thresh,
        presence_refresh_streak=args.presence_refresh_streak,
        detector_assist_policy=args.detector_assist_policy,
        detect_interval=args.detect_interval,
        max_lost=args.max_lost,
        tracker_score_thresh=args.tracker_score_thresh,
        min_confidence=args.conf,
        roi_redetect=not args.disable_roi_redetect,
        manual_confirmation=not args.no_manual_confirmation,
        min_confirm_detections=args.min_confirm_detections,
    )
    recorder = anti_uav.AlertRecorder(
        state_path=args.state_log or None,
        alert_path=args.alert_log or None,
        crop_dir=args.alert_crops or None,
    )

    save_output = Path(args.save_output).expanduser().resolve() if args.save_output else None
    save_image = save_output is not None and save_output.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    writer = None
    last_annotated = None
    should_annotate = not args.disable_annotate or args.show or save_output is not None
    frame_times_ms: list[float] = []
    measured_frames = 0
    measured_totals_ms = defaultdict(float)
    measured_counts = defaultdict(int)

    try:
        for frame_index, (_, frame) in enumerate(iter_frames(args.source, repeat_image=args.repeat_image), start=1):
            if args.max_frames > 0 and frame_index > args.max_frames:
                break
            timer_before = dict(timer.totals_ms)
            counts_before = dict(timer.counts)
            start = perf_counter()
            state = system.step(frame)
            events = system.drain_alerts()
            recorder_start = perf_counter()
            recorder.record_state(state)
            recorder.record_events(frame, events)
            recorder_elapsed_ms = (perf_counter() - recorder_start) * 1000.0
            timer.add("recorder_total_ms", recorder_elapsed_ms)
            end = perf_counter()
            if frame_index > args.warmup_frames:
                frame_times_ms.append((end - start) * 1000.0)
                measured_frames += 1
                for key, value in timer.totals_ms.items():
                    measured_totals_ms[key] += value - timer_before.get(key, 0.0)
                for key, value in timer.counts.items():
                    measured_counts[key] += value - counts_before.get(key, 0)

            annotated = frame
            if should_annotate:
                annotated = system.annotate(frame, state)
                last_annotated = annotated

            if args.show:
                cv2.imshow("anti-uav-rk3588", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in {ord("c"), ord("C")}:
                    system.confirm_current_target(True, note="operator_confirmed")
                elif key in {ord("r"), ord("R")}:
                    system.confirm_current_target(False, note="operator_rejected")

            if save_output is not None and not save_image:
                if writer is None:
                    writer = open_writer(save_output, annotated.shape)
                writer.write(annotated)

        if save_output is not None and save_image and last_annotated is not None:
            save_output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_output), last_annotated)

        if args.benchmark_json:
            benchmark_path = Path(args.benchmark_json).expanduser().resolve()
            benchmark_path.parent.mkdir(parents=True, exist_ok=True)
            mean_ms = float(sum(frame_times_ms) / len(frame_times_ms)) if frame_times_ms else 0.0
            detector_total_ms = float(measured_totals_ms.get("detector_total_ms", 0.0))
            tracker_total_ms = float(measured_totals_ms.get("tracker_total_ms", 0.0))
            presence_total_ms = float(measured_totals_ms.get("presence_total_ms", 0.0))
            recorder_total = float(measured_totals_ms.get("recorder_total_ms", 0.0))
            other_total_ms = max(sum(frame_times_ms) - detector_total_ms - tracker_total_ms - presence_total_ms - recorder_total, 0.0)
            summary = {
                "source": args.source,
                "tracker": args.tracker,
                "detector_assist_policy": args.detector_assist_policy,
                "presence_verifier": verifier_name or "none",
                "presence_model": str(presence_model_path) if presence_model_path is not None else "",
                "model": str(model_path),
                "measured_frames": measured_frames,
                "warmup_frames": args.warmup_frames,
                "mean_ms": mean_ms,
                "p50_ms": percentile(frame_times_ms, 0.50),
                "p95_ms": percentile(frame_times_ms, 0.95),
                "max_ms": max(frame_times_ms) if frame_times_ms else 0.0,
                "fps_mean": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
                "detector_mean_ms_per_frame": detector_total_ms / max(measured_frames, 1),
                "detector_mean_ms_per_call": detector_total_ms / max(measured_counts.get("detector_call_ms", 0), 1),
                "detector_calls": int(measured_counts.get("detector_call_ms", 0)),
                "detector_full_frame_mean_ms": float(measured_totals_ms.get("detector_full_frame_ms", 0.0))
                / max(measured_counts.get("detector_full_frame_ms", 0), 1),
                "detector_full_frame_calls": int(measured_counts.get("detector_full_frame_ms", 0)),
                "detector_roi_mean_ms": float(measured_totals_ms.get("detector_roi_ms", 0.0))
                / max(measured_counts.get("detector_roi_ms", 0), 1),
                "detector_roi_calls": int(measured_counts.get("detector_roi_ms", 0)),
                "detector_detail_mean_ms": {
                    key.removeprefix("detector_").removesuffix("_ms"): float(value)
                    / max(measured_counts.get(key, 0), 1)
                    for key, value in sorted(measured_totals_ms.items())
                    if key.startswith("detector_")
                    and key.endswith("_ms")
                    and key
                    not in {
                        "detector_total_ms",
                        "detector_call_ms",
                        "detector_full_frame_ms",
                        "detector_roi_ms",
                    }
                },
                "detector_detail_calls": {
                    key.removeprefix("detector_").removesuffix("_ms"): int(measured_counts.get(key, 0))
                    for key in sorted(measured_counts)
                    if key.startswith("detector_")
                    and key.endswith("_ms")
                    and key
                    not in {
                        "detector_total_ms",
                        "detector_call_ms",
                        "detector_full_frame_ms",
                        "detector_roi_ms",
                    }
                },
                "tracker_mean_ms_per_frame": tracker_total_ms / max(measured_frames, 1),
                "tracker_init_mean_ms": float(measured_totals_ms.get("tracker_init_ms", 0.0)) / max(measured_counts.get("tracker_init_ms", 0), 1),
                "tracker_init_calls": int(measured_counts.get("tracker_init_ms", 0)),
                "tracker_reset_mean_ms": float(measured_totals_ms.get("tracker_reset_ms", 0.0)) / max(measured_counts.get("tracker_reset_ms", 0), 1),
                "tracker_reset_calls": int(measured_counts.get("tracker_reset_ms", 0)),
                "tracker_mean_ms_per_update": float(measured_totals_ms.get("tracker_update_ms", 0.0)) / max(measured_counts.get("tracker_update_ms", 0), 1),
                "tracker_update_calls": int(measured_counts.get("tracker_update_ms", 0)),
                "tracker_reinit_mean_ms": float(measured_totals_ms.get("tracker_reinit_ms", 0.0)) / max(measured_counts.get("tracker_reinit_ms", 0), 1),
                "tracker_reinit_calls": int(measured_counts.get("tracker_reinit_ms", 0)),
                "tracker_correct_mean_ms": float(measured_totals_ms.get("tracker_correct_ms", 0.0)) / max(measured_counts.get("tracker_correct_ms", 0), 1),
                "tracker_correct_calls": int(measured_counts.get("tracker_correct_ms", 0)),
                "presence_mean_ms_per_frame": presence_total_ms / max(measured_frames, 1),
                "presence_init_mean_ms": float(measured_totals_ms.get("presence_init_ms", 0.0)) / max(measured_counts.get("presence_init_ms", 0), 1),
                "presence_init_calls": int(measured_counts.get("presence_init_ms", 0)),
                "presence_reset_mean_ms": float(measured_totals_ms.get("presence_reset_ms", 0.0)) / max(measured_counts.get("presence_reset_ms", 0), 1),
                "presence_reset_calls": int(measured_counts.get("presence_reset_ms", 0)),
                "presence_eval_mean_ms": float(measured_totals_ms.get("presence_eval_ms", 0.0)) / max(measured_counts.get("presence_eval_ms", 0), 1),
                "presence_eval_calls": int(measured_counts.get("presence_eval_ms", 0)),
                "presence_hard_mean_ms": float(measured_totals_ms.get("presence_hard_ms", 0.0)) / max(measured_counts.get("presence_hard_ms", 0), 1),
                "presence_hard_calls": int(measured_counts.get("presence_hard_ms", 0)),
                "presence_soft_mean_ms": float(measured_totals_ms.get("presence_soft_ms", 0.0)) / max(measured_counts.get("presence_soft_ms", 0), 1),
                "presence_soft_calls": int(measured_counts.get("presence_soft_ms", 0)),
                "recorder_mean_ms_per_frame": recorder_total / max(measured_frames, 1),
                "other_mean_ms_per_frame": other_total_ms / max(measured_frames, 1),
                "annotated": should_annotate,
                "save_output": str(save_output) if save_output is not None else "",
            }
            benchmark_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        if writer is not None:
            writer.release()
        recorder.close()
        detector.release()
        tracker_release = getattr(tracker, "release", None)
        if callable(tracker_release):
            tracker_release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
