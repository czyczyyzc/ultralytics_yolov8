"""Append-only, native-resolution random context crops for gray UAV training."""

from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np


AREA_BINS = ((.02, .10), (.10, .25), (.25, .50), (.50, .80), (.80, 1.001))


def eligible_donors(labels):
    """Pixel thresholds select extra views only, never remove original examples."""
    donors = {}
    for row in labels:
        path = Path(row["im_file"])
        if path.parent.name == "rgb" or {"zoom_train", "zoom_val", "gray_val", "Video00004"} & set(path.parts):
            continue
        boxes = np.asarray(row["bboxes"]).reshape(-1, 4)
        if len(boxes) != 1 or not row.get("normalized", True) or row.get("bbox_format", "xywh") != "xywh":
            continue
        h, w = row["shape"]
        wh = boxes[0, 2:]*[w, h]
        if np.isfinite(boxes).all() and min(wh) >= 48 and max(wh) >= 96:
            donors[str(path)] = row
    return [donors[key] for key in sorted(donors)]


def pixel_box(row):
    h, w = row["shape"]
    x, y, bw, bh = np.asarray(row["bboxes"])[0]*[w, h, w, h]
    return np.array([x-bw/2, y-bh/2, x+bw/2, y+bh/2], dtype=np.float64)


def sample_context_geometry(image_hw, box, rng, max_upscale=4., partial_probability=.15, max_attempts=48):
    """Sample integer crop geometry, with a full-object fallback for valid donors."""
    if max_upscale < 1 or not 0 <= partial_probability <= 1:
        raise ValueError("Invalid crop settings")
    h, w = image_hw
    box = np.asarray(box, dtype=np.float64).copy()
    if not np.isfinite(box).all() or max(-box[0], -box[1], box[2]-w, box[3]-h) > .01:
        raise ValueError("Source GT must lie within the image (0.01px serialization tolerance)")
    # YOLO decimal serialization can move a border coordinate by about 0.001px.
    box[[0, 2]] = box[[0, 2]].clip(0, w)
    box[[1, 3]] = box[[1, 3]].clip(0, h)
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    if min(bw, bh) < 48-.01 or max(bw, bh) < 96-.01:
        raise ValueError("Not an eligible native-resolution donor")
    ratio, area = 960/544, bw*bh
    max_cw = min(w, int(h*ratio))
    min_cw = int(np.ceil(max(960/max_upscale, (544/max_upscale+.5)*ratio)))
    # A fractional GT interval may need one more integer pixel than ceil(box size).
    span_w = int(np.ceil(x2))-int(np.floor(x1))
    span_h = int(np.ceil(y2))-int(np.floor(y1))
    full_min = max(min_cw, span_w, int(np.ceil((span_h+.5)*ratio)))
    if full_min > max_cw:
        raise ValueError("No full-object crop fits; refuse to silently drop the GT")
    request_partial = bool(rng.random() < partial_probability)
    for allow_partial in ((True, False) if request_partial else (False,)):
        low_cw = min_cw if allow_partial else full_min
        min_area = area/(max_cw*round(max_cw/ratio))
        max_area = min(1., area/(low_cw*round(low_cw/ratio)))
        feasible = [(max(lo, min_area), min(hi, max_area)) for lo, hi in AREA_BINS
                    if max(lo, min_area) < min(hi, max_area)]
        if not feasible:
            feasible = [(min_area, max_area)]
        # Choose a bin once so failed geometry does not always bias toward smaller sizes.
        interval = feasible[int(rng.integers(len(feasible)))]
        for _ in range(max_attempts):
            desired = float(np.exp(rng.uniform(np.log(interval[0]), np.log(interval[1]))))
            cw = int(np.clip(round(np.sqrt(area*ratio/desired)), low_cw, max_cw))
            if allow_partial:
                # After truncation, visible box area is smaller than source GT area.
                # Explore tighter crops rather than treating source area as visible area.
                upper = min(max_cw, int(np.ceil(np.sqrt(area*ratio/interval[0]))))
                cw = int(np.clip(round(np.exp(rng.uniform(np.log(low_cw), np.log(max(low_cw, upper))))), low_cw, max_cw))
            ch = int(round(cw/ratio))
            if max(960/cw, 544/ch) > max_upscale:
                continue
            left, right = max(0, int(np.ceil(x2-cw))), min(w-cw, int(np.floor(x1)))
            top, bottom = max(0, int(np.ceil(y2-ch))), min(h-ch, int(np.floor(y1)))
            if allow_partial:
                ox = int(np.clip((x1+x2-cw)/2+rng.uniform(-.3, .3)*bw, 0, w-cw))
                oy = int(np.clip((y1+y2-ch)/2+rng.uniform(-.3, .3)*bh, 0, h-ch))
            elif left <= right and top <= bottom:
                ox, oy = int(rng.integers(left, right+1)), int(rng.integers(top, bottom+1))
            else:
                continue
            clipped = box-[ox, oy, ox, oy]
            clipped[[0, 2]] = clipped[[0, 2]].clip(0, cw)
            clipped[[1, 3]] = clipped[[1, 3]].clip(0, ch)
            retained = float(np.prod(clipped[2:]-clipped[:2])/area)
            fraction = float(np.prod(clipped[2:]-clipped[:2])/(cw*ch))
            if retained < (.6 if allow_partial else .999) or not interval[0]*.99 <= fraction <= interval[1]*1.01:
                continue
            updated = clipped*[960/cw, 544/ch, 960/cw, 544/ch]
            return dict(crop_xywh=[ox, oy, cw, ch], output_box=updated.tolist(),
                        area_fraction=fraction, retained_fraction=retained,
                        upscale=max(960/cw, 544/ch), partial=retained < .999,
                        requested_partial=request_partial, geometry_fallback=False)
    # Rejection sampling must never terminate a long run or silently drop its GT.
    cw, ch = max_cw, int(round(max_cw/ratio))
    left, right = max(0, int(np.ceil(x2-cw))), min(w-cw, int(np.floor(x1)))
    top, bottom = max(0, int(np.ceil(y2-ch))), min(h-ch, int(np.floor(y1)))
    if left > right or top > bottom or max(960/cw, 544/ch) > max_upscale:
        raise ValueError("Donor cannot fit the configured aspect ratio and pixel budget")
    ox, oy = (left+right)//2, (top+bottom)//2
    updated = (box-[ox, oy, ox, oy])*[960/cw, 544/ch, 960/cw, 544/ch]
    return dict(crop_xywh=[ox, oy, cw, ch], output_box=updated.tolist(), area_fraction=float(area/(cw*ch)),
                retained_fraction=1., upscale=max(960/cw, 544/ch), partial=False,
                requested_partial=request_partial, geometry_fallback=True)


def random_context_crop(image, box, rng, max_upscale=4., partial_probability=.15):
    """Read native pixels only after selecting and verifying the crop geometry."""
    meta = sample_context_geometry(image.shape[:2], box, rng, max_upscale, partial_probability)
    ox, oy, cw, ch = meta["crop_xywh"]
    output = cv2.resize(image[oy:oy+ch, ox:ox+cw], (960, 544), interpolation=cv2.INTER_LINEAR)
    return output, np.asarray(meta["output_box"], dtype=np.float64), meta


def gray_capture_jitter(image, rng):
    """Mild sensor variation for extra large-object views, not 4px originals."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gain, offset = float(rng.uniform(.9, 1.1)), float(rng.uniform(-8, 8))
    gray = (gray-gray.mean())*gain+gray.mean()+offset
    if rng.random() < .2:
        gray += rng.normal(0, rng.uniform(.3, 1.5), gray.shape).astype(np.float32)
    if rng.random() < .15:
        gray = cv2.GaussianBlur(gray, (3, 3), float(rng.uniform(.3, .6)))
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class OnlineScaleDataset:
    """Keep every base slot; append equal dynamic-view slots for every eligible donor."""

    def __init__(self, base, config):
        from ultralytics.data.augment import Format
        if base.rect or tuple(base.imgsz) != (544, 960) or not base.augment:
            raise ValueError("Online scale requires augmented, non-rect 544x960 detection training")
        if base.use_segments or base.use_keypoints or base.use_obb:
            raise ValueError("This adapter is for axis-aligned detection only")
        self.base, self.config = base, dict(config)
        self.donors = eligible_donors(base.labels)
        expected = json.loads(Path(config["donor_manifest"]).read_text())["donors"]
        actual = [{"im_file": r["im_file"], "shape": list(r["shape"]), "bboxes": np.asarray(r["bboxes"]).tolist()}
                  for r in self.donors]
        if expected != actual:
            raise ValueError("Donor data changed since the coverage audit")
        self.views = int(config.get("views_per_donor", 4))
        if self.views < 1 or not self.donors:
            raise ValueError("No augmentation slots configured")
        self.labels = base.labels+self.donors*self.views
        self.im_files = [r["im_file"] for r in self.labels]
        self.rect, self.mosaic = False, False
        self.collate_fn = base.collate_fn
        self.format = Format(bbox_format="xywh", normalize=True, batch_idx=True)

    def __len__(self):
        return len(self.labels)

    def close_mosaic(self, hyp):
        self.base.close_mosaic(hyp)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.base):
            return self.base[index]
        row = self.donors[(index-len(self.base)) % len(self.donors)]
        image = cv2.imread(row["im_file"])
        if image is None or tuple(image.shape[:2]) != tuple(row["shape"]):
            raise ValueError(f"Source image changed: {row['im_file']}")
        # DataLoader seeds numpy per worker; each visit gets a fresh reproducible draw.
        rng = np.random.default_rng(int(np.random.randint(0, 2**32-1)))
        image, box, _ = random_context_crop(image, pixel_box(row), rng,
                                           self.config.get("max_upscale", 4.), self.config.get("partial_probability", .15))
        image = gray_capture_jitter(image, rng)
        if rng.random() < .5:
            image = np.ascontiguousarray(image[:, ::-1])
            box[[0, 2]] = 960-box[[2, 0]]
        labels = deepcopy(row)
        labels.pop("shape", None)
        labels.update(img=image, ori_shape=(544, 960), resized_shape=(544, 960),
                      bboxes=np.asarray([box], dtype=np.float32), bbox_format="xyxy", normalized=False)
        result = self.format(self.base.update_labels_info(labels))
        # YOLODataset.collate_fn depends on consistent key order across native / extra samples.
        keys = ("im_file", "ori_shape", "resized_shape", "img", "cls", "bboxes", "batch_idx")
        if set(result) != set(keys):
            raise ValueError(f"Unexpected sample keys: {list(result)}")
        return {key: result[key] for key in keys}
