"""Load unmodified public Dist tracker modules without the detector/PyTorch stack.

Only axis-aligned NumPy observations with ReID disabled are supported. The two
utility functions are extracted verbatim from the pinned upstream source AST.
This loader is intended for an isolated process, not an existing Ultralytics app.
"""
import ast
import hashlib
import importlib
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

CONFIG = dict(tracker_type="botsort", track_high_thresh=.03, track_low_thresh=.01,
              new_track_thresh=.10, track_buffer=30, match_thresh=.8,
              fuse_score=False, gmc_method="sparseOptFlow", proximity_thresh=.5,
              appearance_thresh=.25, with_reid=False)


def _module(name, directory=None):
    module = ModuleType(name)
    if directory is not None:
        module.__path__ = [str(directory)]
    sys.modules[name] = module
    return module


def _function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def load_dist(upstream):
    if "ultralytics" in sys.modules:
        raise RuntimeError("Run the Dist adapter in an isolated Python process")
    root = Path(upstream).resolve() / "ultralytics"
    _module("ultralytics", root)
    _module("ultralytics.trackers", root / "trackers")
    _module("ultralytics.trackers.utils", root / "trackers/utils")
    utils = _module("ultralytics.utils", root / "utils")
    utils.LOGGER = logging.getLogger("dist_public")
    ops = _module("ultralytics.utils.ops")
    # A non-instantiable tensor marker keeps the exact upstream NumPy branch.
    marker = type("UnsupportedTensor", (), {})
    ops.xywh2ltwh = _function(root / "utils/ops.py", "xywh2ltwh",
                             dict(np=np, torch=SimpleNamespace(Tensor=marker)))
    metrics = _module("ultralytics.utils.metrics")
    metrics.bbox_ioa = _function(root / "utils/metrics.py", "bbox_ioa", dict(np=np))

    def unsupported(*args, **kwargs):
        raise NotImplementedError("OBB / ReID / automatic installation is not supported")

    metrics.batch_probiou = unsupported
    checks = _module("ultralytics.utils.checks")
    checks.check_requirements = unsupported
    # Install lap explicitly; upstream must not attempt an implicit pip install.
    import lap
    assert lap.__version__
    bot = importlib.import_module("ultralytics.trackers.bot_sort")
    return bot.BOTSORT


def source_hashes(upstream):
    root = Path(upstream)
    files = list((root / "ultralytics/trackers").rglob("*.py"))
    files += [root / "ultralytics/utils/ops.py", root / "ultralytics/utils/metrics.py"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def as_results(boxes):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 5)
    if not np.isfinite(boxes).all() or not np.all(boxes[:, 2:4] > boxes[:, :2]):
        raise ValueError("Invalid detector observation")
    xywh = boxes[:, :4].copy()
    xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:4]) / 2
    xywh[:, 2:4] = boxes[:, 2:4] - boxes[:, :2]
    return SimpleNamespace(conf=boxes[:, 4], cls=np.zeros(len(boxes), np.float32), xywh=xywh)


def observations(tracker, boxes):
    result = []
    for track in tracker.tracked_stracks:
        if track.frame_id != tracker.frame_id or not track.is_activated:
            continue
        index = int(track.idx)
        if index != track.idx or not 0 <= index < len(boxes):
            raise RuntimeError("Invalid upstream detector association")
        box = boxes[index]
        if abs(float(track.score) - float(box[4])) >= 1e-6:
            raise RuntimeError("Detector score mismatch")
        result.append(dict(id=int(track.track_id), box=box[:4].tolist(),
                           score=float(box[4]), detection_index=index))
    return result
