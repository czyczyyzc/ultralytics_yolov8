"""Offline bridge for motion-aware C++; production inference has no Python dependency."""
import ctypes as ct
from pathlib import Path

import numpy as np

DEFAULTS = dict(high=.03, low=.01, birth=.10, expiry_seconds=1., nominal_fps=30.,
    localization_floor_px=1.5, acceleration_std_px_s2=180., unknown_gmc_speed_px_s=1500.,
    max_innovation_speed_px_s=3000., max_radius_px=240., nis_gate=16.,
    confirmation_hits=2, confirmation_window=3, ambiguity_margin=.03)


class NativeMotion:
    def __init__(self, library, fps=30., config=None):
        self.config = dict(DEFAULTS, nominal_fps=float(fps)) if config is None else dict(config)
        if set(self.config) != set(DEFAULTS):
            raise ValueError("Motion configuration must contain exactly the documented keys")
        values = np.ascontiguousarray([self.config[k] for k in DEFAULTS], dtype=np.float64)
        self.lib = ct.CDLL(str(Path(library).resolve()))
        self.lib.motion_create.argtypes = [ct.c_void_p, ct.c_int]
        self.lib.motion_create.restype = ct.c_void_p
        self.lib.motion_destroy.argtypes = [ct.c_void_p]
        self.lib.motion_error.restype = ct.c_char_p
        self.lib.motion_update.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p,
                                          ct.c_double, ct.c_double, ct.c_void_p, ct.c_int]
        self.lib.motion_update.restype = ct.c_int
        self.lib.motion_stats.argtypes = [ct.c_void_p, ct.c_void_p]
        self.handle = self.lib.motion_create(values.ctypes.data, len(values))
        if not self.handle:
            raise RuntimeError(self.lib.motion_error().decode())
        self.pairs = np.empty((100, 2), np.int32)

    def update(self, boxes, warp, quality, timestamp):
        boxes = np.ascontiguousarray(boxes, dtype=np.float32).reshape(-1, 5)
        warp = np.ascontiguousarray(warp, dtype=np.float64).reshape(2, 3)
        if len(boxes)>len(self.pairs):
            self.pairs = np.empty((len(boxes), 2), np.int32)
        n = self.lib.motion_update(self.handle, boxes.ctypes.data, len(boxes), warp.ctypes.data,
                                  quality, timestamp, self.pairs.ctypes.data, len(self.pairs))
        if n<0:
            raise RuntimeError(self.lib.motion_error().decode())
        shown = []
        for identity, index in self.pairs[:n]:
            index = int(index)
            if not 0<=index<len(boxes):
                raise RuntimeError("Invalid motion observation association")
            box = boxes[index]
            shown.append(dict(id=int(identity), detection_index=index,
                box=box[:4].tolist(), score=float(box[4])))
        return shown

    def stats(self):
        counts = np.empty(8, np.uint64)
        self.lib.motion_stats(self.handle, counts.ctypes.data)
        return dict(zip(("frames", "matches", "zero_iou_matches", "allocated_ids", "confirmations",
                         "ambiguous_columns", "expired_tracks", "live_tracks"), map(int, counts)))

    def close(self):
        if self.handle:
            self.lib.motion_destroy(self.handle)
            self.handle = None
