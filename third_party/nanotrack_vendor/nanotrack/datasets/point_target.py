from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from nanotrack.core.config import cfg
from nanotrack.utils.bbox import corner2center
from nanotrack.utils.point import Point


class PointTarget:
    def __init__(self):
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)

    def __call__(self, target, size, neg=False):
        cls = -1 * np.ones((size, size), dtype=np.int64)
        delta = np.zeros((4, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position
            selected = np.arange(num)
            np.random.shuffle(selected)
            selected = selected[:keep_num]
            return tuple(p[selected] for p in position)

        tcx, tcy, tw, th = corner2center(target)
        points = self.points.points
        if neg:
            neg_pos = np.where(np.square(tcx - points[0]) / np.square(tw / 4) + np.square(tcy - points[1]) / np.square(th / 4) < 1)
            cls[select(neg_pos, cfg.TRAIN.NEG_NUM)] = 0
            return cls, delta

        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2] - points[0]
        delta[3] = target[3] - points[1]
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / 4) + np.square(tcy - points[1]) / np.square(th / 4) < 1)
        neg_pos = np.where(np.square(tcx - points[0]) / np.square(tw / 2) + np.square(tcy - points[1]) / np.square(th / 2) > 1)
        cls[select(pos, cfg.TRAIN.POS_NUM)] = 1
        cls[select(neg_pos, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)] = 0
        return cls, delta
