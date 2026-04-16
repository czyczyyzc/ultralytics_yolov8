# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple

import numpy as np

Corner = namedtuple("Corner", "x1 y1 x2 y2")
BBox = Corner
Center = namedtuple("Center", "x y w h")


def corner2center(corner):
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1)
    x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1


def center2corner(center):
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    x, y, w, h = center[0], center[1], center[2], center[3]
    return x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5


def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        area1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        area2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(area1 / area2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x, y, w, h = region[0], region[1], region[2], region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h
