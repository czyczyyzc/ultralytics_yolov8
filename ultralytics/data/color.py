"""Explicit, geometry-preserving model input color transforms."""

import cv2
import numpy as np


class GrayscaleThreeChannel:
    """Convert final augmented BGR pixels to equal channels before tensor formatting."""

    def __call__(self, labels):
        image = labels["img"]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("gray3 requires uint8 HWC BGR input")
        result = dict(labels)
        result["img"] = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return result


def input_color_transform(mode):
    if mode == "source":
        return None
    if mode == "gray3":
        return GrayscaleThreeChannel()
    raise ValueError(f"Unknown input_color mode: {mode}")
