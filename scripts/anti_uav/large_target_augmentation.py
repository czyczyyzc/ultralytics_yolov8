"""Context-preserving zoom augmentation with explicit box geometry and quality limits."""

from __future__ import annotations

import cv2
import numpy as np


def context_zoom(image, boxes_xyxy, rng, area_range=(.25, .65), max_upscale=4., partial=False, full_frame=False):
    """Return a real-image crop, updated boxes and provenance, or None if unsuitable."""
    height, width = image.shape[:2]
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    if len(boxes) != 1:
        return None  # Do not silently drop other visible objects in multi-target crops.
    x1, y1, x2, y2 = boxes[0]
    bw, bh = x2-x1, y2-y1
    if min(bw, bh) < 48 or max(bw, bh) < 96:
        return None
    ratio = 960/544
    target_area = rng.uniform(*area_range)
    crop_w = int(round(np.sqrt(bw*bh*ratio/target_area)))
    crop_h = int(round(crop_w/ratio))
    if full_frame:
        crop_w = int(min(bw, bh*ratio)*.85)
        crop_h = int(round(crop_w/ratio))
        partial = True
    if not (1 <= crop_w <= width and 1 <= crop_h <= height):
        return None
    if max(960/crop_w, 544/crop_h) > max_upscale:
        return None
    if not partial and (crop_w < bw or crop_h < bh):
        return None
    cx, cy = (x1+x2)/2, (y1+y2)/2
    ox = int(round(cx-crop_w*rng.uniform(.35, .65)))
    oy = int(round(cy-crop_h*rng.uniform(.35, .65)))
    if full_frame:
        ox, oy = int(round(cx-crop_w/2)), int(round(cy-crop_h/2))
    elif partial:
        if rng.random() < .5:
            ox = int(round(x1 + bw*rng.uniform(.1, .35)))
        else:
            oy = int(round(y1 + bh*rng.uniform(.1, .35)))
    else:
        ox = int(np.clip(ox, np.ceil(x2-crop_w), np.floor(x1)))
        oy = int(np.clip(oy, np.ceil(y2-crop_h), np.floor(y1)))
    ox, oy = int(np.clip(ox, 0, width-crop_w)), int(np.clip(oy, 0, height-crop_h))
    clipped = boxes - [ox, oy, ox, oy]
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, crop_w)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, crop_h)
    retained = np.prod(clipped[0, 2:]-clipped[0, :2])/(bw*bh)
    if retained < (.5 if partial else .999):
        return None
    resized = cv2.resize(image[oy:oy+crop_h, ox:ox+crop_w], (960, 544), interpolation=cv2.INTER_LINEAR)
    updated = clipped * [960/crop_w, 544/crop_h, 960/crop_w, 544/crop_h]
    fraction = float(np.prod(updated[0, 2:]-updated[0, :2])/(960*544))
    if not area_range[0] <= fraction <= area_range[1]:
        return None
    return resized, updated, dict(crop_xywh=[ox, oy, crop_w, crop_h],
                                  source_box=boxes[0].tolist(), output_box=updated[0].tolist(),
                                  area_fraction=fraction, retained_fraction=float(retained),
                                  upscale=max(960/crop_w, 544/crop_h), partial=retained < .999)


def yolo_rows(boxes, width=960, height=544):
    rows = []
    for x1, y1, x2, y2 in boxes:
        rows.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    return "\n".join(rows)+"\n"


def mild_capture_effects(image, rng):
    """Model grayscale capture and mild blur, never synthesize missing object detail."""
    result = image.copy()
    gray = rng.random() < .7
    if gray:
        result = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    blur = rng.choice(["none", "gaussian", "motion"], p=[.65, .2, .15])
    if blur == "gaussian":
        result = cv2.GaussianBlur(result, (5, 5), float(rng.uniform(.35, .9)))
    elif blur == "motion":
        kernel = np.zeros((5, 5), dtype=np.float32)
        kernel[2, :] = .2
        result = cv2.filter2D(result, -1, kernel)
    return result, dict(grayscale=gray, blur=blur)
