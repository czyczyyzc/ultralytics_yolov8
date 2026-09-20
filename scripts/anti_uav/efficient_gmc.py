"""Bounded-cost causal sparse optical flow for background camera motion.

This is an explicit alternative to public Dist GMC, not an identical algorithm.
It never skips video frames and does not read target annotations.
"""
import time
import cv2
import numpy as np


class EfficientGMC:
    def __init__(self, max_width=480, max_corners=256, refresh=1):
        if max_width < 64 or max_corners < 8 or refresh < 1:
            raise ValueError("Invalid GMC budget")
        self.max_width, self.max_corners, self.refresh = max_width, max_corners, refresh
        self.previous = self.points = None
        self.index = 0
        self.counts = dict(frames=0, estimated=0, identity_fallback=0, refreshes=0)
        self.seconds = dict(preprocess=0., optical_flow=0., ransac=0., features=0.)

    def _features(self, gray):
        start = time.perf_counter()
        result = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners,
            qualityLevel=.01, minDistance=4, blockSize=3, useHarrisDetector=False, k=.04)
        self.seconds["features"] += time.perf_counter()-start
        self.counts["refreshes"] += 1
        return result

    def apply(self, raw_frame, detections=None):
        start = time.perf_counter()
        height, width = raw_frame.shape[:2]
        w = min(width, self.max_width)
        h = max(1, round(height*w/width))
        # Convert before resize to preserve the reference grayscale convention.
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if raw_frame.ndim == 3 else raw_frame
        if gray.shape != (h, w):
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
        self.seconds["preprocess"] += time.perf_counter()-start
        warp = np.eye(2, 3, dtype=np.float64)
        next_points = None
        accepted = False
        if self.previous is not None and self.previous.shape == gray.shape and self.points is not None and len(self.points) >= 6:
            start = time.perf_counter()
            moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None,
                winSize=(21,21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            self.seconds["optical_flow"] += time.perf_counter()-start
            if moved is not None and status is not None:
                valid = status.ravel().astype(bool) & np.isfinite(moved).all(axis=(1,2))
                valid &= (moved[:,0,0] >= 0) & (moved[:,0,0] < w) & (moved[:,0,1] >= 0) & (moved[:,0,1] < h)
                before, after = self.points[valid], moved[valid]
                if len(after) >= 6:
                    start = time.perf_counter()
                    # Six original-image pixels, matching the reference's 3 px at 1/2 scale.
                    small, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
                        ransacReprojThreshold=6*w/width, maxIters=2000, confidence=.99, refineIters=10)
                    self.seconds["ransac"] += time.perf_counter()-start
                    if small is not None and np.isfinite(small).all() and inliers is not None:
                        scale = float(np.hypot(small[0,0], small[1,0]))
                        mask = inliers.ravel().astype(bool)
                        if mask.sum() >= 6 and mask.mean() >= .3 and .5 <= scale <= 2:
                            sx, sy = width/w, height/h
                            warp[:,:2] = np.diag([sx,sy]) @ small[:,:2] @ np.diag([1/sx,1/sy])
                            warp[:,2] = small[:,2] * [sx,sy]
                            next_points, accepted = after[mask].copy(), True
        if accepted:
            self.counts["estimated"] += 1
        elif self.index:
            self.counts["identity_fallback"] += 1
        if self.index % self.refresh == 0 or next_points is None or len(next_points) < max(6,self.max_corners//2):
            next_points = self._features(gray)
        self.previous, self.points = gray, next_points
        self.index += 1
        self.counts["frames"] += 1
        return warp


class SuppliedWarp:
    """One-use warp produced by the live, ordered GMC worker for this same frame."""
    def __init__(self):
        self.warp = None

    def put(self, warp):
        if self.warp is not None or np.shape(warp) != (2,3) or not np.isfinite(warp).all():
            raise RuntimeError("Invalid or unconsumed live GMC warp")
        self.warp = np.asarray(warp).copy()

    def apply(self, image, detections=None):
        if self.warp is None:
            raise RuntimeError("Missing live GMC result")
        warp, self.warp = self.warp, None
        return warp
