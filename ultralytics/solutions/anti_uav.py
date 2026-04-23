# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import importlib
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass
class Detection:
    """Detection payload used by the alerting-only anti-UAV pipeline."""

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int = -1
    class_name: str = "target"
    source: str = "full_frame"

    def as_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["bbox"] = [float(v) for v in self.bbox]
        return payload


@dataclass
class AlertEvent:
    """Alert lifecycle event emitted by the perception pipeline."""

    event_type: str
    frame_index: int
    alert_id: int
    status: str
    bbox: Optional[Tuple[float, float, float, float]]
    confidence: float
    track_score: float
    threat_score: float
    note: str = ""

    def as_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        if self.bbox is not None:
            payload["bbox"] = [float(v) for v in self.bbox]
        return payload


@dataclass
class TargetState:
    """Per-frame single-target perception status."""

    frame_index: int
    status: str
    bbox: Optional[Tuple[float, float, float, float]]
    confidence: float
    track_score: float
    presence_score: float
    presence_uncertainty: float
    lost_frames: int
    age: int
    frames_since_detection: int
    threat_score: float
    confirmation_state: str
    alert_id: int
    alert_active: bool
    alert_emitted: bool
    detector_mode: str
    confirmation_hits: int
    detection_hits: int
    class_id: int = -1
    class_name: str = "target"
    presence_features: Optional[Dict[str, float]] = None

    def as_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        if self.bbox is not None:
            payload["bbox"] = [float(v) for v in self.bbox]
        return payload


@dataclass
class PresenceEstimate:
    """Presence-verifier output for the active track hypothesis."""

    score: float
    features: Dict[str, float]
    uncertainty: float = 0.0


class BasePresenceVerifier(ABC):
    """Optional verifier that estimates whether the current tracker output still matches the target."""

    name = "base"
    feature_names: Tuple[str, ...] = ()

    def __init__(self):
        self.last_features: Dict[str, float] = {}

    def reset(self) -> None:
        """Forget any verifier-side runtime state."""
        self.last_features = {}

    def on_init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Capture verifier state when a fresh target is initialized."""
        del frame, bbox

    def on_soft_correction(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Update runtime verifier state after a detector-driven bbox correction."""
        del frame, bbox

    def on_hard_reinit(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Refresh verifier state after a hard tracker reinitialization."""
        self.on_init(frame, bbox)

    @abstractmethod
    def evaluate(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        track_score: float,
        *,
        previous_bbox: Optional[Sequence[float]] = None,
        context: Optional[Dict[str, float]] = None,
    ) -> PresenceEstimate:
        """Estimate whether the current tracker bbox still corresponds to the original target."""


PRESENCE_FEATURE_NAMES: Tuple[str, ...] = (
    "track_score",
    "reference_similarity",
    "previous_similarity",
    "motion_ratio",
    "area_change",
    "aspect_change",
    "edge_ratio",
    "detection_gap",
    "contradiction_signal",
    "requires_refresh",
    "assist_active",
)


class FeaturePresenceVerifier(BasePresenceVerifier):
    """Common feature extractor for lightweight tracker-presence verification."""

    name = "feature"
    feature_names = PRESENCE_FEATURE_NAMES

    def __init__(self, patch_scale: float = 1.2, patch_size: int = 32):
        super().__init__()
        self.patch_scale = max(1.0, float(patch_scale))
        self.patch_size = max(8, int(patch_size))
        self.reference_patch = None
        self.previous_patch = None
        self.previous_bbox = None

    def reset(self) -> None:
        """Clear verifier reference state."""
        super().reset()
        self.reference_patch = None
        self.previous_patch = None
        self.previous_bbox = None

    def on_init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Snapshot the detector-backed target appearance as the reference template."""
        clipped = _clip_bbox(bbox, frame.shape)
        patch = self._extract_feature_patch(frame, clipped)
        self.reference_patch = patch
        self.previous_patch = patch
        self.previous_bbox = clipped

    def on_soft_correction(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Refresh short-term motion state while preserving the original reference template."""
        clipped = _clip_bbox(bbox, frame.shape)
        self.previous_patch = self._extract_feature_patch(frame, clipped)
        self.previous_bbox = clipped

    def evaluate(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        track_score: float,
        *,
        previous_bbox: Optional[Sequence[float]] = None,
        context: Optional[Dict[str, float]] = None,
    ) -> PresenceEstimate:
        """Extract lightweight motion/appearance features for the current track hypothesis."""
        context = context or {}
        clipped = _clip_bbox(bbox, frame.shape)
        current_patch = self._extract_feature_patch(frame, clipped)
        reference_bbox = previous_bbox or self.previous_bbox or clipped
        features = {
            "track_score": float(np.clip(track_score, 0.0, 1.0)),
            "reference_similarity": _patch_similarity(self.reference_patch, current_patch),
            "previous_similarity": _patch_similarity(self.previous_patch, current_patch),
            "motion_ratio": float(min(_bbox_center_distance_ratio(clipped, reference_bbox), 10.0) / 10.0),
            "area_change": float(min(_bbox_area_change_ratio(clipped, reference_bbox), 10.0) / 10.0),
            "aspect_change": float(min(_bbox_aspect_change_ratio(clipped, reference_bbox), 10.0) / 10.0),
            "edge_ratio": _bbox_edge_ratio(clipped, frame.shape),
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
        return PresenceEstimate(score=self.score_from_features(features), features=features)

    def score_from_features(self, features: Dict[str, float]) -> float:
        """Convert a feature dict into a presence probability."""
        raise NotImplementedError

    def feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Vectorize features in a stable order for lightweight MLP inference."""
        return np.asarray([float(features.get(name, 0.0)) for name in self.feature_names], dtype=np.float32)

    def _extract_feature_patch(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
        """Extract a normalized grayscale patch used for cheap appearance comparisons."""
        patch = _extract_patch(frame, bbox, self.patch_scale)
        if patch.size == 0:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        resized = cv2.resize(gray, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32)
        std = float(normalized.std())
        if std < 1e-6:
            return normalized / 255.0
        return (normalized - float(normalized.mean())) / std


class HeuristicPresenceVerifier(FeaturePresenceVerifier):
    """Rule-based verifier used as the default lightweight implementation for Scheme A."""

    name = "heuristic"

    def score_from_features(self, features: Dict[str, float]) -> float:
        """Blend motion and appearance cues into a conservative presence score."""
        score = (
            0.34 * features["track_score"]
            + 0.23 * features["reference_similarity"]
            + 0.17 * features["previous_similarity"]
            + 0.09 * (1.0 - features["motion_ratio"])
            + 0.06 * (1.0 - min(max(features["area_change"] - 0.1, 0.0), 1.0))
            + 0.04 * (1.0 - min(max(features["aspect_change"] - 0.1, 0.0), 1.0))
            + 0.03 * (1.0 - features["edge_ratio"])
            + 0.02 * (1.0 - features["detection_gap"])
            + 0.01 * (1.0 - features["contradiction_signal"])
            + 0.01 * (1.0 - features["requires_refresh"])
        )
        return float(np.clip(score, 0.0, 1.0))


class PresenceMLP:
    """Tiny two-layer classifier used by the optional learned presence verifier."""

    def __init__(self, in_features: int, hidden_dim: int = 32):
        import torch.nn as nn

        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def state_dict(self):
        """Expose state dict for saving."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load learned weights."""
        self.model.load_state_dict(state_dict)


class MLPPresenceVerifier(FeaturePresenceVerifier):
    """Learned presence verifier over the lightweight Scheme A feature vector."""

    name = "mlp"

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        device: Optional[str] = None,
        patch_scale: float = 1.2,
        patch_size: int = 32,
    ):
        super().__init__(patch_scale=patch_scale, patch_size=patch_size)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Presence-verifier checkpoint not found: {checkpoint}")

        self._torch = importlib.import_module("torch")
        runtime_device = _parse_torch_device(self._torch, device)
        payload = self._torch.load(str(checkpoint), map_location=runtime_device)
        feature_names = tuple(payload.get("feature_names", self.feature_names))
        if feature_names != self.feature_names:
            raise ValueError(
                "Presence-verifier feature_names mismatch. "
                f"Expected {self.feature_names}, got {feature_names}"
            )
        hidden_dim = int(payload.get("hidden_dim", 32))
        network = PresenceMLP(len(self.feature_names), hidden_dim=hidden_dim)
        network.load_state_dict(payload["state_dict"])
        self._network = network.model.to(runtime_device).eval()
        self._runtime_device = runtime_device

    def score_from_features(self, features: Dict[str, float]) -> float:
        """Run the lightweight MLP and return the target-present probability."""
        vector = self.feature_vector(features)
        tensor = self._torch.from_numpy(vector).unsqueeze(0).to(self._runtime_device)
        with self._torch.no_grad():
            logits = self._network(tensor)
            probability = self._torch.softmax(logits, dim=1)[0, 1].item()
        return float(np.clip(probability, 0.0, 1.0))


class PairPresenceNet:
    """Small ROI verifier network over template/current patches plus optional metadata."""

    def __init__(self, in_channels: int = 2, metadata_dim: int = 0, hidden_dim: int = 64):
        import torch
        import torch.nn as nn

        class _PairPresenceNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.metadata_encoder = (
                    nn.Sequential(nn.Linear(metadata_dim, 32), nn.ReLU(inplace=True)) if metadata_dim > 0 else None
                )
                fused_dim = 64 + (32 if metadata_dim > 0 else 0)
                self.classifier = nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 2),
                )

            def forward(self, image_pair, metadata=None):
                image_features = self.image_encoder(image_pair).flatten(1)
                if self.metadata_encoder is not None and metadata is not None:
                    meta_features = self.metadata_encoder(metadata)
                    image_features = torch.cat([image_features, meta_features], dim=1)
                return self.classifier(image_features)

        self.model = _PairPresenceNet()

    def state_dict(self):
        """Expose state dict for saving."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load learned weights."""
        self.model.load_state_dict(state_dict)


class PairROIPresenceVerifier(FeaturePresenceVerifier):
    """Scheme B/C verifier: a small learned ROI head over template/current crops plus metadata."""

    name = "pair_head"

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        device: Optional[str] = None,
        patch_scale: float = 1.2,
        patch_size: Optional[int] = None,
    ):
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Presence-verifier checkpoint not found: {checkpoint}")

        torch_module = importlib.import_module("torch")
        runtime_device = _parse_torch_device(torch_module, device)
        payload = torch_module.load(str(checkpoint), map_location=runtime_device)
        stored_patch_size = int(payload.get("patch_size", patch_size or 64))
        super().__init__(patch_scale=patch_scale, patch_size=stored_patch_size)
        feature_names = tuple(payload.get("feature_names", self.feature_names))
        if feature_names != self.feature_names:
            raise ValueError(
                "Presence-verifier feature_names mismatch. "
                f"Expected {self.feature_names}, got {feature_names}"
            )

        self._torch = torch_module
        self._runtime_device = runtime_device
        self.loss_mode = str(payload.get("loss_mode", "ce")).lower()
        self.use_metadata = bool(payload.get("use_metadata", True))
        hidden_dim = int(payload.get("hidden_dim", 64))
        metadata_dim = len(self.feature_names) if self.use_metadata else 0
        network = PairPresenceNet(in_channels=2, metadata_dim=metadata_dim, hidden_dim=hidden_dim)
        network.load_state_dict(payload["state_dict"])
        self._network = network.model.to(runtime_device).eval()

    def evaluate(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        track_score: float,
        *,
        previous_bbox: Optional[Sequence[float]] = None,
        context: Optional[Dict[str, float]] = None,
    ) -> PresenceEstimate:
        """Run the pair-head verifier and optionally expose evidential uncertainty."""
        context = context or {}
        clipped = _clip_bbox(bbox, frame.shape)
        current_patch = self._extract_feature_patch(frame, clipped)
        reference_bbox = previous_bbox or self.previous_bbox or clipped
        features = {
            "track_score": float(np.clip(track_score, 0.0, 1.0)),
            "reference_similarity": _patch_similarity(self.reference_patch, current_patch),
            "previous_similarity": _patch_similarity(self.previous_patch, current_patch),
            "motion_ratio": float(min(_bbox_center_distance_ratio(clipped, reference_bbox), 10.0) / 10.0),
            "area_change": float(min(_bbox_area_change_ratio(clipped, reference_bbox), 10.0) / 10.0),
            "aspect_change": float(min(_bbox_aspect_change_ratio(clipped, reference_bbox), 10.0) / 10.0),
            "edge_ratio": _bbox_edge_ratio(clipped, frame.shape),
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
            return PresenceEstimate(score=0.0, features=features, uncertainty=1.0)

        image_pair = np.stack([self.reference_patch, current_patch], axis=0).astype(np.float32, copy=False)
        image_tensor = self._torch.from_numpy(image_pair).unsqueeze(0).to(self._runtime_device)
        metadata_tensor = None
        if self.use_metadata:
            metadata = self.feature_vector(features)
            metadata_tensor = self._torch.from_numpy(metadata).unsqueeze(0).to(self._runtime_device)

        with self._torch.no_grad():
            logits = self._network(image_tensor, metadata_tensor)
            if self.loss_mode == "edl":
                evidence = self._torch.nn.functional.softplus(logits)
                alpha = evidence + 1.0
                total_evidence = alpha.sum(dim=1, keepdim=True)
                probability = (alpha / total_evidence)[0, 1].item()
                uncertainty = min(float(2.0 / max(total_evidence.item(), 1e-6)), 1.0)
            else:
                probability = self._torch.softmax(logits, dim=1)[0, 1].item()
                uncertainty = float(1.0 - abs(probability - 0.5) * 2.0)

        return PresenceEstimate(
            score=float(np.clip(probability, 0.0, 1.0)),
            features=features,
            uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
        )


def build_presence_verifier(name: str = "heuristic", **kwargs) -> BasePresenceVerifier:
    """Instantiate a lightweight presence verifier."""
    key = name.strip().lower()
    if key == "heuristic":
        return HeuristicPresenceVerifier(**kwargs)
    if key == "mlp":
        if not kwargs.get("checkpoint_path"):
            raise ValueError("checkpoint_path is required for MLPPresenceVerifier.")
        return MLPPresenceVerifier(**kwargs)
    if key in {"pair_head", "pair_head_edl"}:
        if not kwargs.get("checkpoint_path"):
            raise ValueError("checkpoint_path is required for PairROIPresenceVerifier.")
        return PairROIPresenceVerifier(**kwargs)
    raise KeyError(f"Unknown presence verifier '{name}'. Available verifiers: heuristic, mlp, pair_head, pair_head_edl")


class BaseSingleTargetTracker(ABC):
    """Generic single-target tracker contract used by the alerting pipeline."""

    name = "base"

    @abstractmethod
    def init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Initialize the tracker on the current frame."""

    @abstractmethod
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float, float, float]], float]:
        """Advance tracker state and return (success, bbox, score)."""

    @abstractmethod
    def reset(self) -> None:
        """Forget the current track."""

    def correct_bbox(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """
        Apply a detector-driven bbox correction without necessarily rebuilding identity state.

        Trackers that do not distinguish soft correction from hard reinitialization can fall back to the
        legacy behavior and simply reinitialize from the detector box.
        """
        self.reinit_from_detection(frame, bbox)

    def reinit_from_detection(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Hard reinitialize tracker state from a detector-provided bbox."""
        self.reset()
        self.init(frame, bbox)


_TRACKER_REGISTRY: Dict[str, Callable[..., BaseSingleTargetTracker]] = {}


def register_tracker(name: str, factory: Callable[..., BaseSingleTargetTracker], *, overwrite: bool = False) -> None:
    """Register a tracker factory by name."""
    key = name.strip().lower()
    if key in _TRACKER_REGISTRY and not overwrite:
        raise ValueError(f"Tracker '{name}' is already registered")
    _TRACKER_REGISTRY[key] = factory


def available_trackers() -> List[str]:
    """Return the list of registered tracker backends."""
    return sorted(_TRACKER_REGISTRY)


def build_tracker(name: str = "template_match", **kwargs) -> BaseSingleTargetTracker:
    """Instantiate a tracker backend from the registry."""
    key = name.strip().lower()
    if key not in _TRACKER_REGISTRY:
        raise KeyError(f"Unknown tracker '{name}'. Available trackers: {', '.join(available_trackers())}")
    return _TRACKER_REGISTRY[key](**kwargs)


class TemplateMatchTracker(BaseSingleTargetTracker):
    """
    Lightweight local-search tracker used as the default safe fallback.

    The interface mirrors a NanoTrack-style init/update/reset API so production trackers can replace it without
    changing the perception state machine.
    """

    name = "template_match"

    def __init__(self, search_scale: float = 2.0, context_scale: float = 1.5, score_threshold: float = 0.2):
        self.search_scale = max(search_scale, 1.1)
        self.context_scale = max(context_scale, 1.0)
        self.score_threshold = score_threshold
        self.template = None
        self.bbox = None

    def reset(self) -> None:
        """Clear tracker state."""
        self.template = None
        self.bbox = None

    def init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Initialize tracker with the current target patch."""
        clipped_bbox = _clip_bbox(bbox, frame.shape)
        patch = _extract_patch(frame, clipped_bbox, self.context_scale)
        if patch.size == 0:
            self.reset()
            return
        self.template = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        self.bbox = clipped_bbox

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float, float, float]], float]:
        """Search a local region with normalized cross correlation."""
        if self.template is None or self.bbox is None:
            return False, None, 0.0

        search_bbox = _expand_bbox(self.bbox, frame.shape, self.search_scale)
        search = _extract_patch(frame, search_bbox, 1.0)
        if search.size == 0:
            self.reset()
            return False, None, 0.0

        search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        template_h, template_w = self.template.shape[:2]
        search_h, search_w = search_gray.shape[:2]
        if search_h < template_h or search_w < template_w:
            return False, self.bbox, 0.0

        result = cv2.matchTemplate(search_gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        if score < self.score_threshold:
            return False, self.bbox, float(score)

        x1 = search_bbox[0] + max_loc[0]
        y1 = search_bbox[1] + max_loc[1]
        x2 = x1 + template_w
        y2 = y1 + template_h
        self.bbox = _clip_bbox((x1, y1, x2, y2), frame.shape)
        return True, self.bbox, float(score)


class OpenCVTracker(BaseSingleTargetTracker):
    """Wrapper for OpenCV single-object trackers when available."""

    name = "opencv"

    def __init__(self, tracker_type: str = "csrt"):
        self.tracker_type = tracker_type.lower()
        self.tracker = None

    def reset(self) -> None:
        """Clear tracker state."""
        self.tracker = None

    def init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Instantiate and initialize the selected OpenCV tracker."""
        self.reset()
        self.tracker = _create_opencv_tracker(self.tracker_type)
        x1, y1, x2, y2 = _clip_bbox(bbox, frame.shape)
        self.tracker.init(frame, tuple(int(round(v)) for v in (x1, y1, x2 - x1, y2 - y1)))

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float, float, float]], float]:
        """Advance the OpenCV tracker state."""
        if self.tracker is None:
            return False, None, 0.0
        ok, bbox = self.tracker.update(frame)
        if not ok:
            return False, None, 0.0
        x, y, w, h = bbox
        return True, _clip_bbox((x, y, x + w, y + h), frame.shape), 1.0


_NANOTRACK_MODULE_CACHE = {}


class NanoTrackPyTracker(BaseSingleTargetTracker):
    """
    Thin adapter around a NanoTrack-style PyTorch implementation.

    This adapter is intentionally limited to perception-only replay. It does not expose any actuation or control API.
    The expected workspace is either the vendored `third_party/nanotrack_vendor` snapshot or a compatible upstream
    checkout passed explicitly through `nanotrack_root`.
    """

    name = "nanotrack"

    def __init__(
        self,
        nanotrack_root: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        snapshot_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        score_threshold: float = 0.25,
    ):
        self.nanotrack_root = _resolve_nanotrack_root(nanotrack_root)
        self.config_path = self._resolve_file(
            config_path,
            fallback=self.nanotrack_root / "models" / "config" / "configv2.yaml",
            label="NanoTrack config",
        )
        self.snapshot_path = self._resolve_file(
            snapshot_path,
            fallback=self.nanotrack_root / "models" / "pretrained" / "nanotrackv2.pth",
            label="NanoTrack snapshot",
        )
        self.score_threshold = float(score_threshold)
        self.device = device
        self.initialized = False
        self._modules = _load_nanotrack_modules(self.nanotrack_root)
        self._torch = importlib.import_module("torch")
        self._runtime_device = _parse_torch_device(self._torch, device)
        if self._runtime_device.type == "cuda":
            self._torch.cuda.set_device(self._runtime_device.index or 0)
        self._tracker = self._build_tracker()

    @staticmethod
    def _resolve_file(value: Optional[Union[str, Path]], fallback: Path, label: str) -> Path:
        path = Path(value).expanduser().resolve() if value else fallback.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path

    def _build_tracker(self):
        cfg = self._modules["cfg"]
        cfg.merge_from_file(str(self.config_path))
        cfg.CUDA = self._runtime_device.type == "cuda"

        model = self._modules["ModelBuilder"]()
        model = self._modules["load_pretrain"](model, str(self.snapshot_path))
        model = model.to(self._runtime_device).eval()
        return self._modules["build_tracker"](model)

    def reset(self) -> None:
        """Forget the current track state while keeping the loaded model in memory."""
        self.initialized = False

    def _apply_runtime_bbox(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Update NanoTrack search center/scale while preserving the existing template features."""
        x1, y1, x2, y2 = _clip_bbox(bbox, frame.shape)
        width = max(float(x2 - x1), 10.0)
        height = max(float(y2 - y1), 10.0)
        self._tracker.center_pos = np.array([x1 + (width - 1.0) / 2.0, y1 + (height - 1.0) / 2.0], dtype=np.float32)
        self._tracker.size = np.array([width, height], dtype=np.float32)
        self._tracker.channel_average = np.mean(frame, axis=(0, 1))
        self.initialized = True

    def init(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Initialize NanoTrack with an xyxy bbox."""
        x1, y1, x2, y2 = _clip_bbox(bbox, frame.shape)
        self._tracker.init(frame, [x1, y1, x2 - x1, y2 - y1])
        self.initialized = True

    def correct_bbox(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Apply detector correction without refreshing the NanoTrack template embedding."""
        self._apply_runtime_bbox(frame, bbox)

    def reinit_from_detection(self, frame: np.ndarray, bbox: Sequence[float]) -> None:
        """Refresh NanoTrack from a detector-provided box and rebuild template state."""
        self.reset()
        self.init(frame, bbox)

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float, float, float]], float]:
        """Advance NanoTrack and convert its xywh output back to xyxy coordinates."""
        if not self.initialized:
            return False, None, 0.0

        outputs = self._tracker.track(frame)
        bbox = outputs.get("bbox")
        if bbox is None:
            return False, None, 0.0

        x, y, w, h = [float(v) for v in bbox]
        clipped = _clip_bbox((x, y, x + w, y + h), frame.shape)
        score = float(outputs.get("best_score", 0.0))
        return score >= self.score_threshold, clipped, score


class DetectionFilter(ABC):
    """Base class for defensive false-positive filters."""

    @abstractmethod
    def keep(self, detection: Detection, frame: np.ndarray) -> bool:
        """Return True if the detection should remain in the candidate set."""


class AreaFilter(DetectionFilter):
    """Reject detections that are too small or too large for the configured scene."""

    def __init__(self, min_area_px: float = 9.0, max_area_ratio: float = 0.25):
        self.min_area_px = min_area_px
        self.max_area_ratio = max_area_ratio

    def keep(self, detection: Detection, frame: np.ndarray) -> bool:
        x1, y1, x2, y2 = detection.bbox
        area = max((x2 - x1) * (y2 - y1), 0.0)
        frame_area = max(float(frame.shape[0] * frame.shape[1]), 1.0)
        return area >= self.min_area_px and area / frame_area <= self.max_area_ratio


class AspectRatioFilter(DetectionFilter):
    """Reject detections with extreme aspect ratios that are unlikely to be UAVs."""

    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 10.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def keep(self, detection: Detection, frame: np.ndarray) -> bool:
        del frame
        x1, y1, x2, y2 = detection.bbox
        height = max(y2 - y1, 1.0)
        width = max(x2 - x1, 1.0)
        ratio = width / height
        return self.min_ratio <= ratio <= self.max_ratio


class BorderFilter(DetectionFilter):
    """Reject detections that sit almost entirely on the frame edge."""

    def __init__(self, margin_px: int = 1):
        self.margin_px = margin_px

    def keep(self, detection: Detection, frame: np.ndarray) -> bool:
        x1, y1, x2, y2 = detection.bbox
        height, width = frame.shape[:2]
        return not (
            x2 <= self.margin_px
            or y2 <= self.margin_px
            or x1 >= width - self.margin_px
            or y1 >= height - self.margin_px
        )


class PatchClassifierFilter(DetectionFilter):
    """
    Generic crop-level classifier filter.

    The classifier should accept a BGR crop and return either a string label or a (label, score) tuple.
    """

    def __init__(self, classifier: Callable[[np.ndarray], Union[str, Tuple[str, float]]], reject_labels: Iterable[str]):
        self.classifier = classifier
        self.reject_labels = {label.lower() for label in reject_labels}

    def keep(self, detection: Detection, frame: np.ndarray) -> bool:
        crop = _extract_patch(frame, detection.bbox, 1.0)
        if crop.size == 0:
            return False
        output = self.classifier(crop)
        label = output[0] if isinstance(output, tuple) else output
        return str(label).lower() not in self.reject_labels


class YOLODetectionAdapter:
    """
    Turn a YOLO model into a detector with optional tiling, ROI re-detection and IR/night preprocessing.

    This adapter intentionally only exposes perception-side functionality. It does not implement any actuation logic.
    """

    def __init__(
        self,
        model,
        *,
        class_names: Optional[Iterable[str]] = None,
        classes: Optional[Iterable[int]] = None,
        conf: float = 0.25,
        imgsz: int = 640,
        device: Optional[str] = None,
        max_det: int = 100,
        tile_size: Optional[Union[int, Tuple[int, int]]] = None,
        tile_overlap: float = 0.2,
        enable_full_frame: bool = True,
        enable_tiling: bool = False,
        enable_roi: bool = True,
        roi_expand: float = 2.5,
        preprocess_mode: str = "rgb",
        clahe: bool = False,
        filters: Optional[Iterable[DetectionFilter]] = None,
        nms_iou: float = 0.45,
    ):
        self.model = model
        self.class_names = {name.lower() for name in class_names} if class_names else None
        self.classes = list(classes) if classes is not None else None
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.tile_size = _normalize_tile_size(tile_size)
        self.tile_overlap = float(np.clip(tile_overlap, 0.0, 0.9))
        self.enable_full_frame = enable_full_frame
        self.enable_tiling = enable_tiling and self.tile_size is not None
        self.enable_roi = enable_roi
        self.roi_expand = max(roi_expand, 1.0)
        self.preprocess_mode = preprocess_mode
        self.clahe = clahe
        self.filters = list(filters or [])
        self.nms_iou = nms_iou

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        """Backward-compatible detector entrypoint."""
        return self.detect(frame)

    def detect(self, frame: np.ndarray, roi: Optional[Sequence[float]] = None, prefer_roi: bool = False) -> List[Detection]:
        """
        Run the configured detector stack.

        Detection order:
        1. ROI re-detection when requested and enabled
        2. Full-frame detection
        3. Tiled detection for tiny targets
        """
        prepared = _prepare_frame(frame, self.preprocess_mode, self.clahe)
        candidates = []

        if roi is not None and self.enable_roi:
            roi_box = _expand_bbox(roi, frame.shape, self.roi_expand)
            candidates.extend(self._predict_crop(prepared, roi_box, source="roi"))
            if prefer_roi and candidates:
                return self._postprocess(frame, candidates)

        if self.enable_full_frame:
            candidates.extend(self._predict_crop(prepared, (0.0, 0.0, frame.shape[1], frame.shape[0]), source="full_frame"))

        if self.enable_tiling:
            tile_h, tile_w = self.tile_size
            for tile_box in iter_tiles(frame.shape[:2], tile_size=(tile_h, tile_w), overlap=self.tile_overlap):
                candidates.extend(self._predict_crop(prepared, tile_box, source="tile"))

        return self._postprocess(frame, candidates)

    def _predict_crop(self, frame: np.ndarray, crop_box: Sequence[float], source: str) -> List[Detection]:
        """Run YOLO on a crop and map boxes back to original frame coordinates."""
        x1, y1, x2, y2 = [int(v) for v in _clip_bbox(crop_box, frame.shape)]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        results = self.model.predict(
            source=crop,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = results[0].names
        detections = []
        for xyxy, conf, cls in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
            class_id = int(cls)
            class_name = str(names[class_id])
            if self.class_names and class_name.lower() not in self.class_names:
                continue
            gx1 = float(x1 + xyxy[0])
            gy1 = float(y1 + xyxy[1])
            gx2 = float(x1 + xyxy[2])
            gy2 = float(y1 + xyxy[3])
            detections.append(
                Detection(
                    bbox=(gx1, gy1, gx2, gy2),
                    confidence=float(conf),
                    class_id=class_id,
                    class_name=class_name,
                    source=source,
                )
            )
        return detections

    def _postprocess(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        """Apply duplicate suppression and defensive false-positive filters."""
        kept = _nms_detections(detections, iou_threshold=self.nms_iou)
        if not self.filters:
            return kept
        return [detection for detection in kept if all(rule.keep(detection, frame) for rule in self.filters)]


class AlertRecorder:
    """JSONL recorder for pipeline state and alert events."""

    def __init__(
        self,
        *,
        state_path: Optional[Union[str, Path]] = None,
        alert_path: Optional[Union[str, Path]] = None,
        crop_dir: Optional[Union[str, Path]] = None,
    ):
        self.state_file = self._open_jsonl(state_path)
        self.alert_file = self._open_jsonl(alert_path)
        self.crop_dir = Path(crop_dir).expanduser().resolve() if crop_dir else None
        if self.crop_dir is not None:
            self.crop_dir.mkdir(parents=True, exist_ok=True)

    def _open_jsonl(self, path: Optional[Union[str, Path]]):
        if path is None or path == "":
            return None
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return open(resolved, "w", encoding="utf-8")

    def record_state(self, state: TargetState) -> None:
        """Write one frame state record."""
        if self.state_file is not None:
            self.state_file.write(json.dumps(state.as_dict(), ensure_ascii=False) + "\n")

    def record_events(self, frame: np.ndarray, events: Sequence[AlertEvent]) -> None:
        """Write alert events and save crops when requested."""
        for event in events:
            if self.alert_file is not None:
                self.alert_file.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
            if self.crop_dir is not None and event.bbox is not None:
                crop = _extract_patch(frame, event.bbox, 1.4)
                if crop.size != 0:
                    crop_path = self.crop_dir / f"alert_{event.alert_id:04d}_frame_{event.frame_index:06d}.jpg"
                    cv2.imwrite(str(crop_path), crop)

    def close(self) -> None:
        """Close all open files."""
        if self.state_file is not None:
            self.state_file.close()
        if self.alert_file is not None:
            self.alert_file.close()


class AntiUAVSystem:
    """
    Alerting-only single-target perception pipeline.

    Pipeline stages:
    detection -> tracking -> lost recovery -> manual confirmation -> alert event logging
    """

    def __init__(
        self,
        detector,
        *,
        tracker: Optional[Union[str, BaseSingleTargetTracker]] = None,
        presence_verifier: Optional[Union[str, BasePresenceVerifier]] = None,
        presence_model_path: Optional[Union[str, Path]] = None,
        presence_device: Optional[str] = None,
        presence_patch_scale: float = 1.2,
        presence_patch_size: int = 32,
        presence_score_thresh: float = 0.45,
        presence_uncertainty_thresh: Optional[float] = None,
        presence_refresh_streak: int = 2,
        detect_interval: int = 2,
        max_lost: int = 30,
        tracker_score_thresh: float = 0.4,
        min_confidence: float = 0.45,
        roi_redetect: bool = True,
        full_frame_fallback: bool = True,
        manual_confirmation: bool = True,
        pending_frames: int = 3,
        auto_confirm_frames: int = 4,
        min_confirm_detections: int = 1,
        association_min_iou: float = 0.05,
        association_max_center_ratio: float = 2.0,
        association_max_area_change: float = 6.0,
        association_max_aspect_change: float = 3.5,
        association_relaxed_multiplier: float = 2.0,
        assist_frames: int = 6,
        assist_hard_reinit_streak: int = 2,
        stale_search_lost_frames: int = 12,
        stale_search_low_score_streak: int = 8,
        stale_search_edge_margin_px: int = 8,
        refresh_override_confidence: float = 0.75,
        refresh_override_continue_confidence: Optional[float] = None,
        refresh_override_consensus_frames: int = 2,
        refresh_override_stability_iou: float = 0.25,
        refresh_override_min_center_ratio: float = 3.0,
        refresh_override_motion_center_ratio: float = 5.0,
        detector_contradiction_confidence: float = 0.75,
        detector_contradiction_continue_confidence: Optional[float] = None,
        detector_contradiction_consensus_frames: int = 2,
        detector_contradiction_min_center_ratio: float = 3.0,
        detector_contradiction_motion_center_ratio: float = 5.0,
        detector_contradiction_track_score_thresh: float = 0.85,
        detector_contradiction_edge_margin_px: int = 8,
        detector_contradiction_high_score_confidence: float = 0.90,
        detector_contradiction_high_score_continue_confidence: Optional[float] = None,
        suspect_score_thresh: Optional[float] = None,
    ):
        self.detector = detector
        if isinstance(tracker, str):
            tracker = (
                build_tracker(tracker, score_threshold=tracker_score_thresh)
                if tracker.strip().lower() == TemplateMatchTracker.name
                else build_tracker(tracker)
            )
        self.tracker = tracker or TemplateMatchTracker(score_threshold=tracker_score_thresh)
        if isinstance(presence_verifier, str):
            verifier_kwargs = {
                "patch_scale": presence_patch_scale,
                "patch_size": presence_patch_size,
            }
            if presence_verifier.strip().lower() == "mlp":
                verifier_kwargs["checkpoint_path"] = presence_model_path
                verifier_kwargs["device"] = presence_device
            elif presence_verifier.strip().lower() in {"pair_head", "pair_head_edl"}:
                verifier_kwargs["checkpoint_path"] = presence_model_path
                verifier_kwargs["device"] = presence_device
            presence_verifier = build_presence_verifier(presence_verifier, **verifier_kwargs)
        self.presence_verifier = presence_verifier
        self.presence_score_thresh = float(np.clip(presence_score_thresh, 0.0, 1.0))
        self.presence_uncertainty_thresh = (
            None if presence_uncertainty_thresh is None else float(np.clip(presence_uncertainty_thresh, 0.0, 1.0))
        )
        self.presence_refresh_streak = max(1, int(presence_refresh_streak))
        self.detect_interval = max(1, detect_interval)
        self.max_lost = max(1, max_lost)
        self.tracker_score_thresh = tracker_score_thresh
        self.min_confidence = min_confidence
        self.roi_redetect = roi_redetect
        self.full_frame_fallback = full_frame_fallback
        self.manual_confirmation = manual_confirmation
        self.pending_frames = max(1, pending_frames)
        self.auto_confirm_frames = max(1, auto_confirm_frames)
        self.min_confirm_detections = max(1, min_confirm_detections)
        self.association_min_iou = max(0.0, association_min_iou)
        self.association_max_center_ratio = max(0.0, association_max_center_ratio)
        self.association_max_area_change = max(1.0, association_max_area_change)
        self.association_max_aspect_change = max(1.0, association_max_aspect_change)
        self.association_relaxed_multiplier = max(1.0, association_relaxed_multiplier)
        self.assist_frames = max(0, int(assist_frames))
        self.assist_hard_reinit_streak = max(1, int(assist_hard_reinit_streak))
        self.stale_search_lost_frames = max(1, int(stale_search_lost_frames))
        self.stale_search_low_score_streak = max(1, int(stale_search_low_score_streak))
        self.stale_search_edge_margin_px = max(1, int(stale_search_edge_margin_px))
        self.refresh_override_confidence = float(refresh_override_confidence)
        default_continue_confidence = max(self.min_confidence, min(self.refresh_override_confidence, 0.60))
        self.refresh_override_continue_confidence = (
            default_continue_confidence
            if refresh_override_continue_confidence is None
            else max(float(refresh_override_continue_confidence), self.min_confidence)
        )
        self.refresh_override_consensus_frames = max(1, int(refresh_override_consensus_frames))
        self.refresh_override_stability_iou = float(np.clip(refresh_override_stability_iou, 0.0, 1.0))
        self.refresh_override_min_center_ratio = max(0.0, float(refresh_override_min_center_ratio))
        self.refresh_override_motion_center_ratio = max(0.0, float(refresh_override_motion_center_ratio))
        self.detector_contradiction_confidence = float(detector_contradiction_confidence)
        default_contradiction_continue = max(
            self.min_confidence, min(self.detector_contradiction_confidence, self.refresh_override_continue_confidence)
        )
        self.detector_contradiction_continue_confidence = (
            default_contradiction_continue
            if detector_contradiction_continue_confidence is None
            else max(float(detector_contradiction_continue_confidence), self.min_confidence)
        )
        self.detector_contradiction_consensus_frames = max(1, int(detector_contradiction_consensus_frames))
        self.detector_contradiction_min_center_ratio = max(0.0, float(detector_contradiction_min_center_ratio))
        self.detector_contradiction_motion_center_ratio = max(0.0, float(detector_contradiction_motion_center_ratio))
        self.detector_contradiction_track_score_thresh = max(
            self.tracker_score_thresh, float(detector_contradiction_track_score_thresh)
        )
        self.detector_contradiction_edge_margin_px = max(1, int(detector_contradiction_edge_margin_px))
        self.detector_contradiction_high_score_confidence = max(
            float(detector_contradiction_high_score_confidence), self.detector_contradiction_confidence
        )
        default_high_score_continue = max(
            self.detector_contradiction_continue_confidence,
            min(self.detector_contradiction_high_score_confidence, 0.90),
        )
        self.detector_contradiction_high_score_continue_confidence = (
            default_high_score_continue
            if detector_contradiction_high_score_continue_confidence is None
            else max(float(detector_contradiction_high_score_continue_confidence), self.detector_contradiction_continue_confidence)
        )
        default_suspect_thresh = max(self.tracker_score_thresh + 0.10, 0.55)
        self.suspect_score_thresh = (
            default_suspect_thresh if suspect_score_thresh is None else max(float(suspect_score_thresh), self.tracker_score_thresh)
        )
        self.reset()

    def reset(self) -> None:
        """Reset all runtime state."""
        self.frame_index = 0
        self.age = 0
        self.lost_frames = 0
        self.frames_since_detection = 0
        self.active_detection = None
        self.bbox = None
        self.confidence = 0.0
        self.track_score = 0.0
        self.presence_score = 0.0
        self.presence_uncertainty = 0.0
        self.status = "searching"
        self.confirmation_state = "idle"
        self.alert_active = False
        self.alert_counter = 0
        self.current_alert_id = 0
        self.pending_events = []
        self.last_detector_mode = "none"
        self.last_frame_shape = (1, 1, 3)
        self.last_threat_score = 0.0
        self.confirmation_hits = 0
        self.detection_hits = 0
        self.requires_detector_refresh = False
        self.assist_frames_remaining = 0
        self.assist_disagreement_streak = 0
        self.low_score_streak = 0
        self.presence_low_streak = 0
        self.last_presence_features = {}
        self.refresh_override_streak = 0
        self.refresh_override_candidate = None
        self.force_hard_reacquire = False
        self.detector_contradiction_streak = 0
        self.detector_contradiction_candidate = None
        self.tracker.reset()
        if self.presence_verifier is not None:
            self.presence_verifier.reset()

    def step(self, frame: np.ndarray) -> TargetState:
        """
        Advance the perception state machine by one frame.

        The output is intentionally safe: it only exposes perception status and alert lifecycle records.
        """
        self.frame_index += 1
        self.pending_events = []
        alert_emitted = False
        tracking_ok = False
        detection_accepted = False
        assist_active = self.assist_frames_remaining > 0
        suspect_tracking = False
        self.last_frame_shape = frame.shape
        self.force_hard_reacquire = False
        previous_bbox = self.bbox

        if self.bbox is not None:
            tracking_ok, tracked_bbox, track_score = self.tracker.update(frame)
            self.track_score = float(track_score)
            if tracking_ok and tracked_bbox is not None:
                self.bbox = _clip_bbox(tracked_bbox, frame.shape)
                self._update_presence_score(
                    frame,
                    self.bbox,
                    previous_bbox=previous_bbox,
                    assist_active=assist_active,
                )
                suspect_tracking = self.track_score < self.suspect_score_thresh
                suspect_tracking = suspect_tracking or self.presence_score < self.presence_score_thresh
                if self.presence_uncertainty_thresh is not None:
                    suspect_tracking = suspect_tracking or self.presence_uncertainty >= self.presence_uncertainty_thresh
                self.status = "tracking"
                self.lost_frames = 0
                self.age += 1
                self.low_score_streak = 0 if self.track_score >= self.tracker_score_thresh else self.low_score_streak + 1
                if self.presence_low_streak >= self.presence_refresh_streak:
                    self.requires_detector_refresh = True
            else:
                self.status = "lost"
                self.lost_frames += 1
                self.low_score_streak += 1
                self.presence_score = 0.0
                self.presence_uncertainty = 1.0
                self.presence_low_streak += 1
                if self.confirmation_state == "confirmed":
                    self.confirmation_state = "pending"
                    self.requires_detector_refresh = True

        if self._should_force_full_frame_search():
            self._drop_target(frame, note="force_full_frame_search")
            tracking_ok = False
            assist_active = False
            suspect_tracking = False

        should_detect = (
            self.bbox is None
            or not tracking_ok
            or self.frames_since_detection >= self.detect_interval
            or self.status == "searching"
            or assist_active
            or suspect_tracking
            or self.requires_detector_refresh
        )

        if should_detect:
            prefer_roi = tracking_ok and not assist_active and not suspect_tracking and not self.requires_detector_refresh
            detections = self._detect_candidates(frame, prefer_roi=prefer_roi)
            strict_association = tracking_ok and not assist_active and not suspect_tracking and not self.requires_detector_refresh
            detection = self._select_detection(detections, association_bbox=self.bbox, strict_association=strict_association)
            self.frames_since_detection = 0
            if detection is not None:
                previous_bbox = self.bbox
                handoff = (
                    "hard"
                    if self.force_hard_reacquire
                    else self._resolve_detection_handoff(previous_bbox, detection.bbox, tracking_ok, assist_active)
                )
                self._activate_detection(frame, detection, handoff=handoff)
                detection_accepted = True
                if previous_bbox is None:
                    self.status = "detected"
                    self.age = 1
                    self.confirmation_hits = 1
                    self.detection_hits = 1
                    self.assist_frames_remaining = 0
                elif tracking_ok and not self.force_hard_reacquire:
                    self.status = "redetected"
                    self.age += 1
                    self.confirmation_hits += 1
                    self.detection_hits += 1
                    self.assist_frames_remaining = self.assist_frames
                else:
                    self.status = "reacquired"
                    self.age += 1
                    self.confirmation_hits = 1
                    self.detection_hits = 1
                    self.assist_frames_remaining = self.assist_frames
                self.lost_frames = 0
                self.low_score_streak = 0
                self._reset_detector_contradiction()
                if self.requires_detector_refresh and self.alert_active:
                    self.confirmation_state = "confirmed"
                self.requires_detector_refresh = False
                self._reset_refresh_override()
            else:
                self.assist_disagreement_streak = 0
                if tracking_ok and self.bbox is not None:
                    self.status = "tracking"
                    self.lost_frames = 0
                elif self.bbox is None or self.lost_frames > self.max_lost:
                    self._drop_target(frame, note="target_lost")
                else:
                    self.status = "lost"
        else:
            self.frames_since_detection += 1
            self.last_detector_mode = "idle"

        if self.bbox is None:
            self.confirmation_hits = 0
            self.detection_hits = 0
            self.requires_detector_refresh = False
            self.assist_frames_remaining = 0
            self.assist_disagreement_streak = 0
            self.low_score_streak = 0
            self.presence_score = 0.0
            self.presence_uncertainty = 0.0
            self.presence_low_streak = 0
            self.last_presence_features = {}
            self._reset_detector_contradiction()
            self._reset_refresh_override()
        elif detection_accepted:
            pass
        elif tracking_ok and self.track_score >= self.tracker_score_thresh:
            self.confirmation_hits += 1
        else:
            self.confirmation_hits = 0

        threat_score = self._estimate_threat_score(frame.shape)
        self.last_threat_score = threat_score
        if self.bbox is not None:
            if self.requires_detector_refresh:
                self.confirmation_state = "pending"
            elif self.manual_confirmation:
                if (
                    self.confirmation_hits >= self.pending_frames
                    and self.detection_hits >= self.min_confirm_detections
                    and self.confirmation_state in {"idle", "pending"}
                ):
                    self.confirmation_state = "pending"
            elif (
                self.confirmation_hits >= self.auto_confirm_frames
                and self.detection_hits >= self.min_confirm_detections
                and self.confirmation_state != "confirmed"
            ):
                event = self._confirm_current_target(note="auto_confirmed")
                alert_emitted = event is not None
        else:
            self.confirmation_state = "idle"

        state = TargetState(
            frame_index=self.frame_index,
            status=self.status,
            bbox=self.bbox,
            confidence=self.confidence,
            track_score=self.track_score,
            presence_score=self.presence_score,
            presence_uncertainty=self.presence_uncertainty,
            lost_frames=self.lost_frames,
            age=self.age,
            frames_since_detection=self.frames_since_detection,
            threat_score=threat_score,
            confirmation_state=self.confirmation_state,
            alert_id=self.current_alert_id if self.alert_active else 0,
            alert_active=self.alert_active,
            alert_emitted=alert_emitted,
            detector_mode=self.last_detector_mode,
            confirmation_hits=self.confirmation_hits,
            detection_hits=self.detection_hits,
            class_id=self.active_detection.class_id if self.active_detection else -1,
            class_name=self.active_detection.class_name if self.active_detection else "target",
            presence_features=dict(self.last_presence_features) if self.last_presence_features else None,
        )
        if self.bbox is not None and self.assist_frames_remaining > 0 and not detection_accepted:
            self.assist_frames_remaining -= 1
        return state

    def confirm_current_target(self, accepted: bool, note: str = "manual_review") -> Optional[AlertEvent]:
        """
        Apply a human confirmation decision to the current target.

        Returns the newly emitted event, if any.
        """
        if self.bbox is None:
            return None
        if accepted:
            if self.requires_detector_refresh:
                self.confirmation_state = "pending"
                return None
            return self._confirm_current_target(note=note)

        self.confirmation_state = "rejected"
        event = self._emit_event("alert_rejected", note=note)
        self._drop_target(None, note="rejected_by_operator", emit_clear=False)
        return event

    def drain_alerts(self) -> List[AlertEvent]:
        """Return and clear queued alert lifecycle events."""
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def annotate(self, frame: np.ndarray, state: Optional[TargetState] = None) -> np.ndarray:
        """Draw a compact debug overlay for replay, review and demos."""
        if state is None:
            state = TargetState(
                frame_index=self.frame_index,
                status=self.status,
                bbox=self.bbox,
                confidence=self.confidence,
                track_score=self.track_score,
                presence_score=self.presence_score,
                presence_uncertainty=self.presence_uncertainty,
                lost_frames=self.lost_frames,
                age=self.age,
                frames_since_detection=self.frames_since_detection,
                threat_score=self._estimate_threat_score(frame.shape),
                confirmation_state=self.confirmation_state,
                alert_id=self.current_alert_id if self.alert_active else 0,
                alert_active=self.alert_active,
                alert_emitted=False,
                detector_mode=self.last_detector_mode,
                confirmation_hits=self.confirmation_hits,
                detection_hits=self.detection_hits,
                class_id=self.active_detection.class_id if self.active_detection else -1,
                class_name=self.active_detection.class_name if self.active_detection else "target",
                presence_features=dict(self.last_presence_features) if self.last_presence_features else None,
            )

        annotated = frame.copy()
        if state.bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in state.bbox]
            color = (0, 255, 0) if state.alert_active else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = (
                f"{state.class_name} {state.status} {state.confirmation_state} "
                f"det={state.confidence:.2f} trk={state.track_score:.2f} prs={state.presence_score:.2f} "
                f"unc={state.presence_uncertainty:.2f} "
                f"thr={state.threat_score:.2f}"
            )
            cv2.putText(
                annotated,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                annotated,
                "searching",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        hud_lines = [
            f"frame={state.frame_index}",
            f"mode={state.detector_mode}",
            f"lost={state.lost_frames}",
            f"alert={'on' if state.alert_active else 'off'}",
        ]
        for index, line in enumerate(hud_lines):
            cv2.putText(
                annotated,
                line,
                (20, 60 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _detect_candidates(self, frame: np.ndarray, prefer_roi: bool) -> List[Detection]:
        """Run the detector stack with ROI re-detection when possible."""
        self.last_detector_mode = "full_frame"
        detect_fn = getattr(self.detector, "detect", None)
        if detect_fn is None:
            return self.detector(frame)

        if self.bbox is not None and self.roi_redetect:
            detections = detect_fn(frame, roi=self.bbox, prefer_roi=prefer_roi)
            self.last_detector_mode = "roi" if detections else "roi_miss"
            if detections or not self.full_frame_fallback:
                return detections

        detections = detect_fn(frame, roi=None, prefer_roi=False)
        self.last_detector_mode = "full_frame"
        return detections

    def _update_presence_score(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        *,
        previous_bbox: Optional[Sequence[float]],
        assist_active: bool,
    ) -> None:
        """Refresh the lightweight presence estimate for the active tracker hypothesis."""
        if self.presence_verifier is None:
            self.presence_score = float(np.clip(self.track_score, 0.0, 1.0))
            self.presence_uncertainty = 0.0
            self.last_presence_features = {}
        else:
            estimate = self.presence_verifier.evaluate(
                frame,
                bbox,
                self.track_score,
                previous_bbox=previous_bbox,
                context={
                    "assist_active": assist_active,
                    "detect_interval": float(self.detect_interval),
                    "detector_contradiction_consensus_frames": float(self.detector_contradiction_consensus_frames),
                    "detector_contradiction_streak": float(self.detector_contradiction_streak),
                    "frames_since_detection": float(self.frames_since_detection),
                    "requires_detector_refresh": float(self.requires_detector_refresh),
                },
            )
            self.presence_score = float(np.clip(estimate.score, 0.0, 1.0))
            self.presence_uncertainty = float(np.clip(estimate.uncertainty, 0.0, 1.0))
            self.last_presence_features = dict(estimate.features)

        low_presence = self.presence_score < self.presence_score_thresh
        if self.presence_uncertainty_thresh is not None:
            low_presence = low_presence or self.presence_uncertainty >= self.presence_uncertainty_thresh
        if not low_presence:
            self.presence_low_streak = 0
        else:
            self.presence_low_streak += 1

    def _activate_detection(self, frame: np.ndarray, detection: Detection, handoff: str = "hard") -> None:
        """Promote a detector output into the active target with either soft correction or hard reinitialization."""
        self.active_detection = detection
        self.bbox = _clip_bbox(detection.bbox, frame.shape)
        self.confidence = float(detection.confidence)
        self.track_score = 1.0
        self.presence_score = 1.0
        self.presence_uncertainty = 0.0
        self.presence_low_streak = 0
        if self.confirmation_state == "rejected":
            self.confirmation_state = "idle"
        if handoff == "soft":
            self.tracker.correct_bbox(frame, self.bbox)
            if self.presence_verifier is not None:
                self.presence_verifier.on_soft_correction(frame, self.bbox)
        else:
            self.tracker.reinit_from_detection(frame, self.bbox)
            self.assist_disagreement_streak = 0
            if self.presence_verifier is not None:
                self.presence_verifier.on_hard_reinit(frame, self.bbox)
        self.last_presence_features = dict(getattr(self.presence_verifier, "last_features", {})) if self.presence_verifier else {}

    def _drop_target(self, frame: Optional[np.ndarray], note: str, emit_clear: bool = True) -> None:
        """Forget the current target and return to searching."""
        if self.alert_active and emit_clear:
            self._emit_event("alert_cleared", note=note)
        del frame
        self.active_detection = None
        self.bbox = None
        self.confidence = 0.0
        self.track_score = 0.0
        self.presence_score = 0.0
        self.presence_uncertainty = 0.0
        self.status = "searching"
        self.age = 0
        self.lost_frames = 0
        self.frames_since_detection = 0
        self.confirmation_hits = 0
        self.detection_hits = 0
        self.requires_detector_refresh = False
        self.assist_disagreement_streak = 0
        self.low_score_streak = 0
        self.presence_low_streak = 0
        self.last_presence_features = {}
        self._reset_detector_contradiction()
        self._reset_refresh_override()
        self.confirmation_state = "idle"
        self.alert_active = False
        self.current_alert_id = 0
        self.tracker.reset()
        if self.presence_verifier is not None:
            self.presence_verifier.reset()

    def _resolve_detection_handoff(
        self,
        previous_bbox: Optional[Sequence[float]],
        detection_bbox: Sequence[float],
        tracking_ok: bool,
        assist_active: bool,
    ) -> str:
        """
        Decide whether a detector refresh should only correct the live box or rebuild the tracker template.

        During the assist window we tolerate one detector/tracker disagreement to avoid overreacting to detector
        jitter. If the detector only passes the relaxed association gate for multiple consecutive assist frames,
        the current template is likely stale and we escalate to a hard reinitialization.
        """
        if previous_bbox is None or not tracking_ok:
            self.assist_disagreement_streak = 0
            return "hard"
        if not assist_active:
            self.assist_disagreement_streak = 0
            return "soft"

        detector_disagrees = not self._passes_association_gate(detection_bbox, previous_bbox, strict=True)
        if not detector_disagrees:
            self.assist_disagreement_streak = 0
            return "soft"

        self.assist_disagreement_streak += 1
        if self.assist_disagreement_streak >= self.assist_hard_reinit_streak:
            self.assist_disagreement_streak = 0
            return "hard"
        return "soft"

    def _select_detection(
        self,
        detections: Sequence[Detection],
        association_bbox: Optional[Sequence[float]] = None,
        strict_association: bool = False,
    ) -> Optional[Detection]:
        """Choose the most plausible single target, preferring consistency with the current target."""
        filtered = [d for d in detections if d.confidence >= self.min_confidence]
        if not filtered:
            self._reset_detector_contradiction()
            self._reset_refresh_override()
            return None

        if association_bbox is None:
            self._reset_detector_contradiction()
            self._reset_refresh_override()
            return max(filtered, key=lambda det: det.confidence)

        if self._allow_detector_contradiction(association_bbox):
            contradiction = self._select_detector_contradiction(filtered, association_bbox)
            if contradiction is not None and self.detector_contradiction_streak >= self.detector_contradiction_consensus_frames:
                self.requires_detector_refresh = True
                self._reset_refresh_override()
                self.force_hard_reacquire = True
                return contradiction
        else:
            self._reset_detector_contradiction()

        if self.requires_detector_refresh:
            refresh_override = self._select_refresh_override(filtered, association_bbox)
            if refresh_override is not None:
                if self.refresh_override_streak >= self.refresh_override_consensus_frames:
                    self.force_hard_reacquire = True
                    return refresh_override
                return None
        else:
            self._reset_refresh_override()

        gated = [d for d in filtered if self._passes_association_gate(d.bbox, association_bbox, strict_association)]
        if not gated:
            return None

        return max(gated, key=lambda det: det.confidence + 0.3 * _bbox_iou(det.bbox, association_bbox))

    def _allow_detector_contradiction(self, association_bbox: Sequence[float]) -> bool:
        """Enable contradiction handling when the tracker looks stale despite still returning success."""
        if self.requires_detector_refresh or association_bbox is None or self.bbox is None:
            return False
        return True

    def _select_detector_contradiction(
        self,
        detections: Sequence[Detection],
        association_bbox: Sequence[float],
    ) -> Optional[Detection]:
        """Track repeated far full-frame detections that contradict the active tracker hypothesis."""
        elevated_threshold = self._requires_high_score_detector_contradiction()
        if elevated_threshold:
            min_confidence = (
                self.detector_contradiction_high_score_confidence
                if self.detector_contradiction_candidate is None
                else self.detector_contradiction_high_score_continue_confidence
            )
        else:
            min_confidence = (
                self.detector_contradiction_confidence
                if self.detector_contradiction_candidate is None
                else self.detector_contradiction_continue_confidence
            )
        candidates = [
            detection
            for detection in detections
            if detection.source == "full_frame"
            and detection.confidence >= min_confidence
            and self._is_far_detector_contradiction_candidate(detection.bbox, association_bbox)
        ]
        if not candidates:
            self._reset_detector_contradiction()
            return None

        selected = max(
            candidates,
            key=lambda det: (
                1 if self._is_consistent_detector_contradiction_candidate(det.bbox) else 0,
                det.confidence,
            ),
        )
        if self._is_consistent_detector_contradiction_candidate(selected.bbox):
            self.detector_contradiction_streak += 1
        else:
            self.detector_contradiction_streak = 1
        self.detector_contradiction_candidate = selected
        return selected

    def _requires_high_score_detector_contradiction(self) -> bool:
        """Use a stricter contradiction threshold when the tracker still looks confident and is not edge-locked."""
        if self.bbox is None:
            return False
        if self.track_score <= self.detector_contradiction_track_score_thresh:
            return False
        return not _bbox_near_frame_edge(self.bbox, self.last_frame_shape, self.detector_contradiction_edge_margin_px)

    def _select_refresh_override(
        self,
        detections: Sequence[Detection],
        association_bbox: Sequence[float],
    ) -> Optional[Detection]:
        """Track a stable far full-frame candidate while detector refresh is required."""
        min_confidence = (
            self.refresh_override_confidence
            if self.refresh_override_candidate is None
            else self.refresh_override_continue_confidence
        )
        candidates = [
            detection
            for detection in detections
            if detection.source == "full_frame"
            and detection.confidence >= min_confidence
            and self._is_far_refresh_candidate(detection.bbox, association_bbox)
        ]
        if not candidates:
            self._reset_refresh_override()
            return None

        selected = max(
            candidates,
            key=lambda det: (
                1 if self._is_consistent_refresh_override_candidate(det.bbox) else 0,
                det.confidence,
            ),
        )
        if self._is_consistent_refresh_override_candidate(selected.bbox):
            self.refresh_override_streak += 1
        else:
            self.refresh_override_streak = 1
        self.refresh_override_candidate = selected
        return selected

    def _is_far_detector_contradiction_candidate(
        self,
        candidate_bbox: Sequence[float],
        reference_bbox: Sequence[float],
    ) -> bool:
        """Return True when a full-frame candidate strongly contradicts the current tracker hypothesis."""
        if self._passes_association_gate(candidate_bbox, reference_bbox, strict=True):
            return False
        return _bbox_center_distance_ratio(candidate_bbox, reference_bbox) >= self.detector_contradiction_min_center_ratio

    def _is_consistent_detector_contradiction_candidate(self, candidate_bbox: Sequence[float]) -> bool:
        """Allow a contradiction to build either from stable boxes or from smoothly moving distant detections."""
        if self.detector_contradiction_candidate is None:
            return False
        previous_bbox = self.detector_contradiction_candidate.bbox
        return self._is_motion_consistent_candidate(
            candidate_bbox,
            previous_bbox,
            self.refresh_override_stability_iou,
            self.detector_contradiction_motion_center_ratio,
        )

    def _is_far_refresh_candidate(self, candidate_bbox: Sequence[float], reference_bbox: Sequence[float]) -> bool:
        """Return True when a full-frame candidate is far enough to justify bypassing association gating."""
        if self._passes_association_gate(candidate_bbox, reference_bbox, strict=True):
            return False
        return _bbox_center_distance_ratio(candidate_bbox, reference_bbox) >= self.refresh_override_min_center_ratio

    def _is_consistent_refresh_override_candidate(self, candidate_bbox: Sequence[float]) -> bool:
        """Allow both stable and smoothly moving far candidates to build refresh consensus."""
        if self.refresh_override_candidate is None:
            return False

        previous_bbox = self.refresh_override_candidate.bbox
        return self._is_motion_consistent_candidate(
            candidate_bbox,
            previous_bbox,
            self.refresh_override_stability_iou,
            self.refresh_override_motion_center_ratio,
        )

    def _is_motion_consistent_candidate(
        self,
        candidate_bbox: Sequence[float],
        previous_bbox: Sequence[float],
        stability_iou: float,
        motion_center_ratio: float,
    ) -> bool:
        """Treat either overlapping or smoothly moving candidate boxes as a coherent detection trajectory."""
        if _bbox_iou(candidate_bbox, previous_bbox) >= stability_iou:
            return True

        center_ratio = _bbox_center_distance_ratio(candidate_bbox, previous_bbox)
        area_change = _bbox_area_change_ratio(candidate_bbox, previous_bbox)
        aspect_change = _bbox_aspect_change_ratio(candidate_bbox, previous_bbox)
        return (
            center_ratio <= motion_center_ratio
            and area_change <= self.association_max_area_change
            and aspect_change <= self.association_max_aspect_change
        )

    def _should_force_full_frame_search(self) -> bool:
        """Drop a stale edge-locked track so the detector can re-enter global search mode."""
        if self.bbox is None or not self.requires_detector_refresh:
            return False
        if self.lost_frames < self.stale_search_lost_frames:
            return False
        if self.low_score_streak < self.stale_search_low_score_streak:
            return False
        return _bbox_near_frame_edge(self.bbox, self.last_frame_shape, self.stale_search_edge_margin_px)

    def _reset_refresh_override(self) -> None:
        """Clear full-frame refresh override state."""
        self.refresh_override_streak = 0
        self.refresh_override_candidate = None
        self.force_hard_reacquire = False

    def _reset_detector_contradiction(self) -> None:
        """Clear detector-contradiction state accumulated while tracker still reports success."""
        self.detector_contradiction_streak = 0
        self.detector_contradiction_candidate = None

    def _passes_association_gate(
        self,
        candidate_bbox: Sequence[float],
        reference_bbox: Sequence[float],
        strict: bool,
    ) -> bool:
        """Reject detector refreshes that jump too far from the active target hypothesis."""
        iou = _bbox_iou(candidate_bbox, reference_bbox)
        center_ratio = _bbox_center_distance_ratio(candidate_bbox, reference_bbox)
        area_change = _bbox_area_change_ratio(candidate_bbox, reference_bbox)
        aspect_change = _bbox_aspect_change_ratio(candidate_bbox, reference_bbox)

        multiplier = 1.0 if strict else self.association_relaxed_multiplier
        min_iou = self.association_min_iou / multiplier
        max_center_ratio = self.association_max_center_ratio * multiplier
        max_area_change = self.association_max_area_change * multiplier
        max_aspect_change = self.association_max_aspect_change * multiplier

        spatially_consistent = iou >= min_iou or center_ratio <= max_center_ratio
        scale_consistent = area_change <= max_area_change and aspect_change <= max_aspect_change
        return spatially_consistent and scale_consistent

    def _confirm_current_target(self, note: str) -> Optional[AlertEvent]:
        """Mark the current target as human-validated and raise an alert if needed."""
        if self.bbox is None:
            return None
        self.confirmation_state = "confirmed"
        if self.alert_active:
            return None
        self.alert_active = True
        self.alert_counter += 1
        self.current_alert_id = self.alert_counter
        return self._emit_event("alert_raised", note=note)

    def _emit_event(self, event_type: str, note: str) -> AlertEvent:
        """Create and queue an alert event."""
        event = AlertEvent(
            event_type=event_type,
            frame_index=self.frame_index,
            alert_id=self.current_alert_id or (self.alert_counter + 1),
            status=self.status,
            bbox=self.bbox,
            confidence=self.confidence,
            track_score=self.track_score,
            threat_score=self.last_threat_score,
            note=note,
        )
        self.pending_events.append(event)
        return event

    def _estimate_threat_score(self, frame_shape: Tuple[int, ...]) -> float:
        """
        Compute a conservative alert score for operator review.

        This is a perception-side salience estimate only. It should not be interpreted as an engagement decision.
        """
        if self.bbox is None:
            return 0.0

        image_area = max(float(frame_shape[0] * frame_shape[1]), 1.0)
        box_area = max((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]), 0.0)
        area_ratio = min(box_area / image_area, 0.15) / 0.15
        score = 0.65 * self.confidence + 0.25 * self.track_score + 0.10 * area_ratio
        return float(np.clip(score, 0.0, 1.0))


def iter_tiles(
    shape_hw: Tuple[int, int],
    *,
    tile_size: Tuple[int, int],
    overlap: float = 0.2,
) -> List[Tuple[float, float, float, float]]:
    """Generate overlapping tiles that fully cover an image."""
    height, width = shape_hw
    tile_h, tile_w = tile_size
    stride_y = max(1, int(tile_h * (1.0 - overlap)))
    stride_x = max(1, int(tile_w * (1.0 - overlap)))
    y_starts = list(range(0, max(height - tile_h, 0) + 1, stride_y)) or [0]
    x_starts = list(range(0, max(width - tile_w, 0) + 1, stride_x)) or [0]
    if y_starts[-1] != max(height - tile_h, 0):
        y_starts.append(max(height - tile_h, 0))
    if x_starts[-1] != max(width - tile_w, 0):
        x_starts.append(max(width - tile_w, 0))
    return [(float(x), float(y), float(min(x + tile_w, width)), float(min(y + tile_h, height))) for y in y_starts for x in x_starts]


def _prepare_frame(frame: np.ndarray, preprocess_mode: str, clahe: bool) -> np.ndarray:
    """Prepare RGB/gray/IR-like inputs for detection."""
    mode = preprocess_mode.lower()
    if mode in {"rgb", "bgr"}:
        prepared = frame
    elif mode in {"gray", "ir", "infrared"}:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if clahe:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        prepared = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")
    return prepared


def _normalize_tile_size(tile_size: Optional[Union[int, Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    """Normalize tile size configuration."""
    if tile_size is None:
        return None
    if isinstance(tile_size, int):
        return tile_size, tile_size
    if len(tile_size) != 2:
        raise ValueError(f"tile_size must be int or a 2-tuple, got {tile_size}")
    return int(tile_size[0]), int(tile_size[1])


def _nms_detections(detections: Sequence[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Greedy NMS over Detection objects."""
    if not detections:
        return []
    remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [candidate for candidate in remaining if _bbox_iou(current.bbox, candidate.bbox) < iou_threshold]
    return kept


def _create_opencv_tracker(tracker_type: str):
    """Create an OpenCV tracker across cv2 and cv2.legacy namespaces."""
    tracker_type = tracker_type.lower()
    constructors = [
        f"Tracker{tracker_type.upper()}_create",
        f"Tracker{tracker_type.capitalize()}_create",
    ]
    namespaces = [cv2]
    if hasattr(cv2, "legacy"):
        namespaces.append(cv2.legacy)
    for namespace in namespaces:
        for name in constructors:
            constructor = getattr(namespace, name, None)
            if constructor is not None:
                return constructor()
    raise RuntimeError(f"OpenCV tracker '{tracker_type}' is not available in this build")


def _resolve_nanotrack_root(nanotrack_root: Optional[Union[str, Path]]) -> Path:
    """Resolve a NanoTrack workspace path."""
    candidates = []
    if nanotrack_root:
        candidates.append(Path(nanotrack_root))
    for env_name in ("NANOTRACK_ROOT",):
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(Path(raw))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "third_party" / "nanotrack_vendor")
    candidates.append(repo_root / "third_party" / "SiamTrackers" / "NanoTrack")

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "nanotrack").exists():
            return resolved
    hint = candidates[0].expanduser().resolve() if candidates else (repo_root / "third_party" / "nanotrack_vendor").resolve()
    raise FileNotFoundError(
        "NanoTrack workspace not found. Vendor the local snapshot or pass "
        f"nanotrack_root/NANOTRACK_ROOT. Last checked: {hint}"
    )


def _load_nanotrack_modules(nanotrack_root: Path) -> dict:
    """Import the upstream NanoTrack modules once per workspace path."""
    key = str(nanotrack_root.resolve())
    if key in _NANOTRACK_MODULE_CACHE:
        return _NANOTRACK_MODULE_CACHE[key]

    if key not in sys.path:
        sys.path.insert(0, key)

    modules = {
        "cfg": importlib.import_module("nanotrack.core.config").cfg,
        "ModelBuilder": importlib.import_module("nanotrack.models.model_builder").ModelBuilder,
        "build_tracker": importlib.import_module("nanotrack.tracker.tracker_builder").build_tracker,
        "load_pretrain": importlib.import_module("nanotrack.utils.model_load").load_pretrain,
    }
    _NANOTRACK_MODULE_CACHE[key] = modules
    return modules


def _parse_torch_device(torch_module, device: Optional[Union[str, int]]):
    """Parse a single-device torch target for NanoTrack."""
    if device is None or device == "":
        return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")
    if isinstance(device, int):
        return torch_module.device(f"cuda:{device}")

    value = str(device).strip()
    if "," in value:
        value = value.split(",", 1)[0].strip()
    if value.isdigit():
        return torch_module.device(f"cuda:{value}")
    return torch_module.device(value)


def _clip_bbox(bbox: Sequence[float], frame_shape: Tuple[int, ...]) -> Tuple[float, float, float, float]:
    """Clip bbox coordinates to image bounds."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    x2 = float(np.clip(x2, x1 + 1, width))
    y2 = float(np.clip(y2, y1 + 1, height))
    return x1, y1, x2, y2


def _expand_bbox(bbox: Sequence[float], frame_shape: Tuple[int, ...], scale: float) -> Tuple[float, float, float, float]:
    """Expand a bbox around its center."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = (x2 - x1) * scale
    height = (y2 - y1) * scale
    return _clip_bbox((cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0), frame_shape)


def _extract_patch(frame: np.ndarray, bbox: Sequence[float], scale: float) -> np.ndarray:
    """Crop a bbox-expanded patch from the frame."""
    x1, y1, x2, y2 = _expand_bbox(bbox, frame.shape, scale)
    return frame[int(y1):int(y2), int(x1):int(x2)]


def _bbox_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute IoU for target matching."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _bbox_center_distance_ratio(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Center distance normalized by the reference-box diagonal."""
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    distance = float(np.hypot(cx1 - cx2, cy1 - cy2))
    reference_diag = float(np.hypot(max(box2[2] - box2[0], 1.0), max(box2[3] - box2[1], 1.0)))
    return distance / max(reference_diag, 1.0)


def _bbox_area_change_ratio(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Symmetric area-change ratio between two boxes."""
    area1 = max((box1[2] - box1[0]) * (box1[3] - box1[1]), 1.0)
    area2 = max((box2[2] - box2[0]) * (box2[3] - box2[1]), 1.0)
    return max(area1 / area2, area2 / area1)


def _bbox_aspect_change_ratio(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Symmetric aspect-ratio change between two boxes."""
    aspect1 = max(box1[2] - box1[0], 1.0) / max(box1[3] - box1[1], 1.0)
    aspect2 = max(box2[2] - box2[0], 1.0) / max(box2[3] - box2[1], 1.0)
    return max(aspect1 / aspect2, aspect2 / aspect1)


def _bbox_edge_ratio(bbox: Sequence[float], frame_shape: Tuple[int, ...]) -> float:
    """Return how close a bbox sits to the nearest border, normalized to [0, 1]."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    distances = (
        max(float(x1), 0.0),
        max(float(y1), 0.0),
        max(float(width - x2), 0.0),
        max(float(height - y2), 0.0),
    )
    min_distance = min(distances)
    normalizer = max(min(width, height) * 0.1, 1.0)
    return float(np.clip(1.0 - min_distance / normalizer, 0.0, 1.0))


def _bbox_near_frame_edge(bbox: Sequence[float], frame_shape: Tuple[int, ...], margin_px: int) -> bool:
    """Return True when a bbox is anchored close to any image border."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    margin = max(float(margin_px), 0.0)
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def _patch_similarity(reference_patch: Optional[np.ndarray], current_patch: Optional[np.ndarray]) -> float:
    """Return a bounded similarity score between two normalized appearance patches."""
    if reference_patch is None or current_patch is None:
        return 0.0
    if reference_patch.shape != current_patch.shape:
        return 0.0
    reference = reference_patch.astype(np.float32, copy=False).reshape(-1)
    current = current_patch.astype(np.float32, copy=False).reshape(-1)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(current))
    if denominator < 1e-6:
        return 1.0 if np.allclose(reference, current) else 0.0
    similarity = float(np.dot(reference, current) / denominator)
    return float(np.clip((similarity + 1.0) * 0.5, 0.0, 1.0))


register_tracker(TemplateMatchTracker.name, TemplateMatchTracker, overwrite=True)
register_tracker(OpenCVTracker.name, OpenCVTracker, overwrite=True)
register_tracker(NanoTrackPyTracker.name, NanoTrackPyTracker, overwrite=True)


__all__ = (
    "AlertEvent",
    "AlertRecorder",
    "AntiUAVSystem",
    "AreaFilter",
    "AspectRatioFilter",
    "BasePresenceVerifier",
    "BaseSingleTargetTracker",
    "BorderFilter",
    "Detection",
    "DetectionFilter",
    "FeaturePresenceVerifier",
    "HeuristicPresenceVerifier",
    "MLPPresenceVerifier",
    "NanoTrackPyTracker",
    "OpenCVTracker",
    "PatchClassifierFilter",
    "PairPresenceNet",
    "PairROIPresenceVerifier",
    "PresenceEstimate",
    "PresenceMLP",
    "TargetState",
    "TemplateMatchTracker",
    "YOLODetectionAdapter",
    "available_trackers",
    "build_presence_verifier",
    "build_tracker",
    "iter_tiles",
    "register_tracker",
)
