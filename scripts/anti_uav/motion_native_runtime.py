"""Offline bridge for motion-aware C++; production inference has no Python dependency."""
import ctypes as ct
from pathlib import Path

import numpy as np

DEFAULTS = dict(high=.03, low=.01, birth=.10, expiry_seconds=1., nominal_fps=30.,
    localization_floor_px=1.5, acceleration_std_px_s2=180., unknown_gmc_speed_px_s=1500.,
    max_innovation_speed_px_s=3000., max_radius_px=240., nis_gate=16.,
    confirmation_hits=3, confirmation_window=4, ambiguity_margin=.03)
CANDIDATE_DEFAULTS = dict(candidate_birth=.03, emit_tentative_ids=1)
CAUSAL_DEFAULTS = dict(DEFAULTS, **CANDIDATE_DEFAULTS)
OBSERVATION_STATUSES = ("unassigned", "pending", "confirmed", "ambiguous", "below_low")


class NativeMotion:
    def __init__(self, library, fps=30., config=None):
        self.config = dict(DEFAULTS, nominal_fps=float(fps)) if config is None else dict(config)
        if set(self.config) not in (set(DEFAULTS), set(CAUSAL_DEFAULTS)):
            raise ValueError("Motion configuration must contain exactly the documented keys")
        if abs(float(self.config["nominal_fps"])-float(fps))>1e-6:
            raise ValueError("Motion nominal FPS must match source FPS")
        order = DEFAULTS if set(self.config)==set(DEFAULTS) else CAUSAL_DEFAULTS
        values = np.ascontiguousarray([self.config[k] for k in order], dtype=np.float64)
        self.lib = ct.CDLL(str(Path(library).resolve()))
        self.lib.motion_create.argtypes = [ct.c_void_p, ct.c_int]
        self.lib.motion_create.restype = ct.c_void_p
        self.lib.motion_destroy.argtypes = [ct.c_void_p]
        self.lib.motion_error.restype = ct.c_char_p
        self.lib.motion_update.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p,
                                          ct.c_double, ct.c_double, ct.c_void_p, ct.c_int]
        self.lib.motion_update.restype = ct.c_int
        self.lib.motion_stats.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.motion_assignment_stats.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.motion_observations.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
        self.lib.motion_observations.restype = ct.c_int
        self.handle = self.lib.motion_create(values.ctypes.data, len(values))
        if not self.handle:
            raise RuntimeError(self.lib.motion_error().decode())
        self.pairs = np.empty((100, 2), np.int32)
        self._observations = []

    def observations(self):
        """Current measured boxes, including explicit identity-pending/ambiguous states."""
        if not self.handle:
            raise RuntimeError("Motion tracker is closed")
        return [dict(row, box=list(row["box"])) for row in self._observations]

    def update(self, boxes, warp, quality, timestamp):
        if not self.handle:
            raise RuntimeError("Motion tracker is closed")
        boxes = np.ascontiguousarray(boxes, dtype=np.float32).reshape(-1, 5)
        warp = np.ascontiguousarray(warp, dtype=np.float64).reshape(2, 3)
        if len(boxes)>len(self.pairs):
            self.pairs = np.empty((len(boxes), 2), np.int32)
        n = self.lib.motion_update(self.handle, boxes.ctypes.data, len(boxes), warp.ctypes.data,
                                  quality, timestamp, self.pairs.ctypes.data, len(self.pairs))
        if n<0:
            raise RuntimeError(self.lib.motion_error().decode())
        triples = np.empty((max(1, len(boxes)), 3), np.int32)
        count = self.lib.motion_observations(self.handle, triples.ctypes.data, len(triples))
        if count != len(boxes):
            raise RuntimeError("Native observation contract does not cover all detections")
        observations = []
        for identity, index, status in triples[:count]:
            index, status, identity = int(index), int(status), int(identity)
            if index != len(observations) or not 0<=status<len(OBSERVATION_STATUSES):
                raise RuntimeError("Invalid native observation index/status")
            allowed=status==2 or (status==1 and self.config.get("emit_tentative_ids",0)==1)
            if identity<0 or (identity>0 and not allowed) or (status==2 and identity==0):
                raise RuntimeError("Unconfirmed native observation exposes an ID")
            box = boxes[index]
            observations.append(dict(id=identity or None, detection_index=index,
                status=OBSERVATION_STATUSES[status], confirmed=status==2, predicted=False,
                box=box[:4].tolist(), score=float(box[4])))
        self._observations = observations
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
        if not self.handle:
            raise RuntimeError("Motion tracker is closed")
        counts = np.empty(8, np.uint64)
        self.lib.motion_stats(self.handle, counts.ctypes.data)
        result = dict(zip(("frames", "matches", "zero_iou_matches", "allocated_ids", "confirmations",
                         "ambiguous_columns", "expired_tracks", "live_tracks"), map(int, counts)))
        self.lib.motion_assignment_stats(self.handle, counts.ctypes.data)
        result["association"] = dict(zip(("frames", "admissible_edges", "components", "base_solves", "alternative_solves",
            "global_resolved_local_conflicts", "ambiguous_detections", "budget_abstentions"), map(int, counts)))
        return result

    def close(self):
        if self.handle:
            self.lib.motion_destroy(self.handle)
            self.handle = None
