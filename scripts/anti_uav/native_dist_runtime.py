"""ctypes bridge for the equivalent native public Dist association path."""
import ctypes as ct
from pathlib import Path
import numpy as np
from dist_numpy_runtime import CONFIG


class NativeDist:
    def __init__(self, library, fps=100, config=None):
        config = CONFIG if config is None else config
        if config["with_reid"] or config["fuse_score"]:
            raise ValueError("Native adapter implements no-ReID/no-score-fusion only")
        self.lib = ct.CDLL(str(Path(library).resolve()))
        self.lib.dist_create.argtypes = [ct.c_double]*5 + [ct.c_int]
        self.lib.dist_create.restype = ct.c_void_p
        self.lib.dist_destroy.argtypes = [ct.c_void_p]
        self.lib.dist_error.restype = ct.c_char_p
        self.lib.dist_update.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int]
        self.lib.dist_update.restype = ct.c_int
        self.lib.dist_snapshot.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int]
        self.lib.dist_snapshot.restype = ct.c_int
        self.handle = self.lib.dist_create(fps,config["track_high_thresh"],config["track_low_thresh"],
            config["new_track_thresh"],config["match_thresh"],config["track_buffer"])
        if not self.handle:
            raise RuntimeError(self.lib.dist_error().decode())
        self.pairs = np.empty((100,2),np.int32)

    def update(self, boxes, warp):
        boxes = np.ascontiguousarray(boxes,dtype=np.float32).reshape(-1,5)
        warp = np.ascontiguousarray(warp,dtype=np.float64).reshape(2,3)
        if not np.isfinite(boxes).all() or not np.isfinite(warp).all() or not np.all(boxes[:,2:4]>boxes[:,:2]):
            raise ValueError("Invalid detector observation or GMC")
        if len(boxes)>len(self.pairs):
            self.pairs = np.empty((len(boxes),2),np.int32)
        n = self.lib.dist_update(self.handle,boxes.ctypes.data,len(boxes),warp.ctypes.data,
                                 self.pairs.ctypes.data,len(self.pairs))
        if n<0:
            raise RuntimeError(self.lib.dist_error().decode())
        result = []
        for tid, index in self.pairs[:n]:
            index = int(index)
            if not 0<=index<len(boxes):
                raise RuntimeError("Invalid native observation index")
            b = boxes[index]
            result.append(dict(id=int(tid),box=b[:4].tolist(),score=float(b[4]),detection_index=index))
        return result

    def snapshot(self):
        rows = np.empty((4096,79),np.float64)
        n = self.lib.dist_snapshot(self.handle,rows.ctypes.data,len(rows))
        if n<0:
            raise RuntimeError("Debug state capacity exceeded")
        return rows[:n].copy()

    def close(self):
        if self.handle:
            self.lib.dist_destroy(self.handle)
            self.handle = None

