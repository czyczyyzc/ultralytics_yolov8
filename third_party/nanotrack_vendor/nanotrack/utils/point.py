import numpy as np


class Point:
    def __init__(self, stride, size, image_center):
        self.stride = stride
        self.size = size
        self.image_center = image_center
        self.points = self.generate_points(stride, size, image_center)

    @staticmethod
    def generate_points(stride, size, image_center):
        origin = image_center - (size // 2) * stride
        x, y = np.meshgrid([origin + stride * dx for dx in range(size)], [origin + stride * dy for dy in range(size)])
        return np.stack([x.astype(np.float32), y.astype(np.float32)])
