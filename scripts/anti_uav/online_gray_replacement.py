"""Training-only cutout replacement using audited, precomputed background ROIs."""
from collections import Counter, OrderedDict
from copy import deepcopy
from fractions import Fraction
import io
import json
import math
import os
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from scripts.anti_uav.build_gray_replacement_batch import load_assets
from scripts.anti_uav.synthesize_gray_drone_replacements import (
    SkipSample, normalize_box, parse_boxes, render_prepared, sha256, stable_seed,
)


def encode_prepared(prepared):
    stream = io.BytesIO()
    np.savez_compressed(stream, patch=prepared["patch"], clean=prepared["clean"], erase=prepared["erase"],
                        meta=np.frombuffer(json.dumps(prepared["meta"]).encode(), dtype=np.uint8))
    return stream.getvalue()


def decode_prepared(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        result = {k: data[k] for k in ("patch", "clean", "erase")}
        result["meta"] = json.loads(data["meta"].tobytes())
    side = result["meta"]["side"]
    if any(result[k].shape != (side, side) for k in ("patch", "clean", "erase")):
        raise ValueError("Invalid cached ROI shape")
    if not np.isfinite(result["clean"]).all():
        raise ValueError("Invalid cached background")
    return result


def select_variant(seed, identity, epoch, occurrence, asset_count, probability=.5):
    """Balanced original/replacement slots and a deterministic per-frame asset cycle.

    At p=.5 and one exposure/epoch, 106 epochs visit 53 distinct assets and 53
    originals. A short run does NOT promise to expose every frame to all assets.
    """
    if not 0 <= probability <= 1 or epoch < 0 or asset_count < 1:
        raise ValueError("Invalid online replacement schedule")
    if probability == 0:
        return None
    fraction = Fraction(str(probability)).limit_denominator(100)
    n, d = fraction.numerator, fraction.denominator
    phase = stable_seed(seed, identity, occurrence, "phase") % d
    cycle, slot = divmod(epoch+phase, d)
    if slot >= n:
        return None
    order = np.random.default_rng(stable_seed(seed, identity, "assets")).permutation(asset_count)
    return int(order[(cycle*n+slot+occurrence) % asset_count])


def compose_cached(image, prepared, rgba):
    """Return native-size BGR + updated normalized bbox; never mutate the source."""
    m = prepared["meta"]
    if tuple(image.shape[:2]) != tuple(m["image_hw"]):
        raise ValueError("Source dimensions changed")
    x, y, side = (m[k] for k in ("left", "top", "side"))
    gray_roi = cv2.cvtColor(image[y:y+side, x:x+side], cv2.COLOR_BGR2GRAY)
    if not np.array_equal(gray_roi, prepared["patch"]):
        raise ValueError("Source pixels differ from the audited cached ROI")
    patch, mask, box, metrics = render_prepared(prepared, rgba)
    if not np.array_equal(patch[mask == 0], gray_roi[mask == 0]):
        raise AssertionError("Unmasked ROI pixels changed")
    result = image.copy()
    roi = result[y:y+side, x:x+side]
    # Assign only the edit mask; even an original's slightly unequal RGB channels
    # outside the repaired foreground stay bit-for-bit untouched.
    roi[mask > 0] = patch[mask > 0, None]
    h, w = image.shape[:2]
    text = normalize_box(box, w, h)
    if not np.allclose(parse_boxes(text, w, h)[0], box, atol=1e-4, rtol=0):
        raise AssertionError("Online label roundtrip failed")
    return result, np.asarray([float(v) for v in text.split()[1:]], dtype=np.float32), metrics


class OnlineReplacementDataset:
    """Keep sample count, original transforms and negative sampling unchanged."""
    def __init__(self, base, config):
        if base.rect or not base.augment or tuple(base._imgsz_hw()) != (544, 960):
            raise ValueError("Online replacement requires non-rect augmented 544x960 detection")
        if base.use_segments or base.use_keypoints or base.use_obb:
            raise ValueError("Only axis-aligned box detection is supported")
        self.base, self.config = base, dict(config)
        self.root = Path(config["cache"])
        summary = json.loads((self.root / "summary.json").read_text())
        if summary["stage"] != "complete" or (summary["is_smoke_subset"] and not config.get("allow_smoke_cache", False)):
            raise ValueError("Background cache is incomplete or smoke-only")
        manifest_path = self.root / "index.json"
        if sha256(manifest_path) != summary["index_sha256"]:
            raise ValueError("Background cache index changed")
        index = json.loads(manifest_path.read_text())
        self.index = index["images"]
        self.blocked = set(index["heldout_sha256"])
        if any(r["video_sha256"] in self.blocked for r in self.index.values()):
            raise ValueError("Held-out video found in augmentation cache")
        catalog = Path(index["catalog"])
        if sha256(catalog) != index["catalog_sha256"]:
            raise ValueError("Cutout catalog changed")
        self.assets = load_assets(catalog, {"24", "25"})
        if [a["id"] for a, _ in self.assets] != index["asset_ids"]:
            raise ValueError("Cutout set changed")
        self.probability = float(config.get("replacement_probability", .5))
        if not 0 <= self.probability <= 1:
            raise ValueError("Replacement probability must be in [0,1]")
        self.seed, self.epoch = int(config.get("seed", 20260918)), 0
        self.labels, self.im_files = base.labels, base.im_files
        self.collate_fn, self.rect, self.mosaic = base.collate_fn, False, False
        counts = Counter()
        self.occurrences = []
        for row in self.labels:
            path = row["im_file"]
            self.occurrences.append(counts[path])
            counts[path] += 1
            if path not in self.index:
                continue
            info = self.index[path]
            h, w = row["shape"]
            boxes = np.asarray(row["bboxes"])
            if (len(row["cls"]) != 1 or float(np.asarray(row["cls"]).ravel()[0]) != 0
                    or boxes.shape != (1, 4) or not row.get("normalized", True)
                    or row.get("bbox_format", "xywh") != "xywh"):
                raise ValueError("Cached positive is not a single normalized drone box")
            x, y, bw, bh = boxes[0]*[w, h, w, h]
            if not np.allclose([x-bw/2, y-bh/2, bw, bh], info["box"], atol=.25, rtol=0):
                raise ValueError("Training label differs from the reviewed cache annotation")
        self._pid, self._dbs, self._verified = None, OrderedDict(), set()

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def close_mosaic(self, hyp):
        self.base.close_mosaic(hyp)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_pid=None, _dbs=OrderedDict(), _verified=set())
        return state

    def cached(self, info):
        if self._pid != os.getpid():
            for db in self._dbs.values():
                db.close()
            self._pid, self._dbs, self._verified = os.getpid(), OrderedDict(), set()
        name = info["pack"]
        if name not in self._dbs:
            if len(self._dbs) >= 4:
                self._dbs.popitem(last=False)[1].close()
            self._dbs[name] = sqlite3.connect(f"file:{(self.root/name).resolve()}?mode=ro", uri=True)
        self._dbs.move_to_end(name)
        row = self._dbs[name].execute("SELECT payload, payload_sha FROM backgrounds WHERE frame=? AND state='ready'",
                                     (info["frame"],)).fetchone()
        if row is None:
            raise ValueError("Ready cache entry disappeared")
        import hashlib
        if hashlib.sha256(row[0]).hexdigest() != row[1] or row[1] != info["payload_sha"]:
            raise ValueError("Background payload checksum mismatch")
        return decode_prepared(row[0])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.labels[index]
        path = row["im_file"]
        info = self.index.get(path)
        if info is None:
            return self.base[index]
        choice = select_variant(self.seed, f"{info['video_sha256']}:{info['frame']}",
                                self.epoch, self.occurrences[index], len(self.assets), self.probability)
        if choice is None:
            return self.base[index]
        prepared = self.cached(info)
        stat = Path(path).stat()
        fingerprint = (path, stat.st_size, stat.st_mtime_ns)
        if fingerprint not in self._verified:
            if sha256(Path(path)) != info["source_sha256"]:
                raise ValueError("Source image changed since background preparation")
            self._verified.add(fingerprint)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        try:
            image, bbox, _ = compose_cached(image, prepared, self.assets[choice][1])
        except SkipSample:
            # An unsuitable/downsampled cutout must not produce an empty target.
            return self.base[index]
        h0, w0 = image.shape[:2]
        th, tw = self.base._imgsz_hw()
        ratio = min(th/h0, tw/w0)
        if ratio != 1:
            image = cv2.resize(image, (min(math.ceil(w0*ratio), tw), min(math.ceil(h0*ratio), th)),
                               interpolation=cv2.INTER_LINEAR)
        label = deepcopy(row)
        label.pop("shape", None)
        label.update(img=image, ori_shape=(h0, w0), resized_shape=image.shape[:2],
                     ratio_pad=(image.shape[0]/h0, image.shape[1]/w0), bboxes=bbox[None])
        return self.base.transforms(self.base.update_labels_info(label))
