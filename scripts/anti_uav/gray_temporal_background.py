"""Audited neighboring-frame backgrounds for offline grayscale synthesis."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


class BackgroundUnavailable(ValueError):
    pass


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exclusion_mask(shape, boxes, scale=1.0):
    mask = np.zeros(shape, np.uint8)
    for x, y, w, h in boxes:
        pad = max(5.0, max(w, h)*0.25)
        x0, y0 = max(0, int((x-pad)*scale)), max(0, int((y-pad)*scale))
        x1 = min(shape[1], int(math.ceil((x+w+pad)*scale)))
        y1 = min(shape[0], int(math.ceil((y+h+pad)*scale)))
        mask[y0:y1, x0:x1] = 255
    return mask


def register_neighbor(reference, donor, ref_boxes, donor_boxes):
    """Estimate donor->reference affine; ignore target features on both frames."""
    cv2.setRNGSeed(0)
    scale = min(1.0, 960/reference.shape[1])
    size = (round(reference.shape[1]*scale), round(reference.shape[0]*scale))
    ref = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
    other = cv2.resize(donor, size, interpolation=cv2.INTER_AREA)
    ref_mask = exclusion_mask(ref.shape, ref_boxes, scale)
    other_mask = exclusion_mask(ref.shape, donor_boxes, scale)
    points = cv2.goodFeaturesToTrack(ref, maxCorners=800, qualityLevel=0.005,
                                    minDistance=10, mask=255-ref_mask, blockSize=7)
    matrix, detail = None, {}
    if points is not None and len(points) >= 20:
        moved, status, _ = cv2.calcOpticalFlowPyrLK(ref, other, points, None,
                                                  winSize=(31, 31), maxLevel=4)
        if moved is None or status is None:
            raise BackgroundUnavailable("optical_flow_unavailable")
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(other, ref, moved, None,
                                                       winSize=(31, 31), maxLevel=4)
        if back is None or back_status is None:
            raise BackgroundUnavailable("backward_optical_flow_unavailable")
        src, dst = moved[:, 0], points[:, 0]
        q = np.round(src).astype(int)
        bounds = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < ref.shape[1]) & (q[:, 1] < ref.shape[0])
        safe = np.zeros(len(points), bool)
        safe[bounds] = other_mask[q[bounds, 1], q[bounds, 0]] == 0
        good = status.ravel().astype(bool) & back_status.ravel().astype(bool) & safe
        good &= np.linalg.norm(back[:, 0]-dst, axis=1) < 0.8
        if good.sum() >= 20:
            candidate, inliers = cv2.estimateAffinePartial2D(src[good], dst[good],
                method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=2000, confidence=0.995)
            if candidate is not None and inliers is not None:
                keep = inliers.ravel() > 0
                predicted = src[good] @ candidate[:, :2].T + candidate[:, 2]
                error = np.median(np.linalg.norm(predicted[keep]-dst[good][keep], axis=1)) / scale
                if keep.sum() >= 20 and keep.mean() >= 0.5 and error < 1.5:
                    matrix = candidate
                    detail = dict(registration="LK_forward_backward_RANSAC", inliers=int(keep.sum()),
                                  inlier_fraction=float(keep.mean()), median_error_px=float(error))
    if matrix is None:
        # Low-texture sky often has too few stable corners. ECC is only a fallback;
        # its result must also pass an independent local background agreement gate.
        valid = 255-cv2.bitwise_or(ref_mask, other_mask)
        initial = np.eye(2, 3, dtype=np.float32)
        try:
            score, backward = cv2.findTransformECC(
                cv2.GaussianBlur(ref, (5, 5), 1).astype(np.float32)/255,
                cv2.GaussianBlur(other, (5, 5), 1).astype(np.float32)/255,
                initial, cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-5), valid, 5)
        except cv2.error as e:
            raise BackgroundUnavailable("registration_failed") from e
        if score < 0.92:
            raise BackgroundUnavailable("low_ECC_correlation")
        matrix = cv2.invertAffineTransform(backward)
        detail = dict(registration="masked_ECC_euclidean", ecc=float(score))
    zoom = math.sqrt(abs(float(np.linalg.det(matrix[:, :2]))))
    if not 0.94 <= zoom <= 1.06 or abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))) > 8:
        raise BackgroundUnavailable("implausible_camera_transform")
    matrix[:, 2] /= scale
    return matrix, detail


def recover_contour(reference, donor, box):
    """Use temporal residuals to mask only connected foreground/rotor/codec halos."""
    x, y, w, h = box
    yy, xx = np.mgrid[:reference.shape[0], :reference.shape[1]]
    pad = max(5, int(math.ceil(max(w, h)*0.25)))
    search = (xx >= x-pad) & (xx < x+w+pad) & (yy >= y-pad) & (yy < y+h+pad)
    inside = (xx >= x) & (xx < x+w) & (yy >= y) & (yy < y+h)
    residual = reference.astype(np.float32)-donor
    noise = max(0.7, float(np.median(np.abs(residual[~search]-np.median(residual[~search]))))*1.4826)
    weak = (np.abs(residual) > max(3.0, 2.5*noise)) & search
    strong = (np.abs(residual) > max(8.0, 5*noise)) & inside
    if strong.sum() < 3:
        raise BackgroundUnavailable("no_reliable_foreground_contour")
    # Closing connects small breaks in a rotor's compression halo, not the GT box.
    weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, components = cv2.connectedComponents(weak)
    ids = np.unique(components[strong])
    ids = ids[ids != 0]
    core = np.isin(components, ids).astype(np.uint8)
    radius = max(2, min(5, int(math.ceil(max(w, h)*0.035))))
    hard = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2))
    soft = cv2.GaussianBlur(hard.astype(np.float32), (7, 7), 1.0)
    soft[hard > 0] = 1.0
    soft[soft < 0.005] = 0
    return soft, dict(residual_noise_mad=noise, foreground_core_pixels=int(core.sum()),
                      repair_support_pixels=int((soft > 0).sum()))


def boundary_metrics(reference, clean, matte):
    changed = (matte > 0).astype(np.uint8)
    outer = cv2.dilate(changed, np.ones((5, 5), np.uint8)).astype(bool) & ~changed.astype(bool)
    inner = changed.astype(bool) & ~cv2.erode(changed, np.ones((5, 5), np.uint8)).astype(bool)
    # A local high-pass ratio helps reject a smooth/white-noise replacement patch.
    hp = clean-cv2.GaussianBlur(clean, (5, 5), 1)
    denom = float(np.std(hp[outer])) if outer.any() else 0
    ratio = float(np.std(hp[inner]))/max(0.1, denom) if inner.any() else 0
    return dict(repair_boundary_mean_abs_change=float(np.mean(np.abs(clean-reference)[inner])) if inner.any() else 0,
                inner_outer_highpass_std_ratio=ratio)


class TemporalBackgrounds:
    """Registry of exact original videos and their approved COCO annotations."""
    def __init__(self, registry: Path, blocked=("Video00004",), offsets=(10, -10, 20, -20, 40, -40, 80, -80)):
        self.records, self.cache, self.offsets = {}, {}, offsets
        raw = json.loads(Path(registry).read_text())
        for entry in raw["videos"]:
            def path(k):
                return (Path(registry).parent / entry[k]).resolve()
            video, approved, coco = path("video"), path("approved_manifest"), path("coco")
            meta = json.loads(approved.read_text())
            name = Path(meta["video"]["name"]).stem
            if any(b.lower() in name.lower() for b in blocked):
                raise ValueError(f"Held-out temporal donor rejected: {name}")
            if name in self.records:
                raise ValueError(f"Duplicate temporal video: {name}")
            if file_hash(video) != meta["video"]["sha256"]:
                raise ValueError(f"Temporal video hash mismatch: {video}")
            ann_file = next(f for f in meta["files"] if f["path"] == "coco/annotations.json")
            if file_hash(coco) != ann_file["sha256"]:
                raise ValueError(f"Temporal annotation hash mismatch: {coco}")
            data = json.loads(coco.read_text())
            frame_by_id = {im["id"]: int(im["frame_index"]) for im in data["images"]}
            boxes = defaultdict(list)
            for a in data["annotations"]:
                boxes[frame_by_id[a["image_id"]]].append(a["bbox"])
            frame_meta = meta["frames"]
            if frame_meta["indexBase"] != 0:
                raise ValueError("Temporal frame indices must be zero based")
            eligible = set(frame_meta["includedFrameIndices"])
            eligible -= set(frame_meta.get("excludedUncertainFrameIndices", []))
            eligible -= set(frame_meta.get("excludedUnreviewedFrameIndices", []))
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise ValueError(f"Cannot open temporal video: {video}")
            self.records[name] = dict(video=video, metadata=meta, boxes=boxes,
                capture=capture, eligible=eligible, video_sha256=meta["video"]["sha256"],
                coco_sha256=ann_file["sha256"], approved_manifest_sha256=file_hash(approved))

    def close(self):
        for r in self.records.values():
            r["capture"].release()

    def read(self, record, frame):
        cap = record["capture"]
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
            raise BackgroundUnavailable("video_seek_failed")
        ok, image = cap.read()
        if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(frame+1)) > 0.5:
            raise BackgroundUnavailable("video_frame_index_mismatch")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def reconstruct(self, gray, box, video_id, frame, roi):
        cache_key = (video_id, frame, tuple(roi), tuple(box))
        if cache_key in self.cache:
            return self.cache[cache_key]
        if video_id not in self.records:
            raise BackgroundUnavailable("no_registered_temporal_video")
        r = self.records[video_id]
        if frame not in r["eligible"]:
            raise BackgroundUnavailable("source_frame_not_approved")
        if len(r["boxes"][frame]) != 1 or not np.allclose(r["boxes"][frame][0], box, atol=0.25, rtol=0):
            raise BackgroundUnavailable("source_GT_differs_from_approved_temporal_annotation")
        decoded = self.read(r, frame)
        if decoded.shape != gray.shape:
            raise BackgroundUnavailable("video_shape_mismatch")
        source_error = float(np.mean(np.abs(gray.astype(np.float32)-decoded)))
        if source_error > 3.0:
            raise BackgroundUnavailable("source_image_does_not_match_video_frame")
        left, top, side = roi
        ref = gray[top:top+side, left:left+side].astype(np.float32)
        local_box = [box[0]-left, box[1]-top, box[2], box[3]]
        reference_mask = exclusion_mask(ref.shape, [local_box])
        yy, xx = np.mgrid[:side, :side].astype(np.float32)
        design = np.stack((np.ones_like(xx), xx/side, yy/side), axis=-1)
        best, errors = None, []
        for offset in self.offsets:
            donor_frame = frame+offset
            if donor_frame not in r["eligible"]:
                continue
            try:
                donor = self.read(r, donor_frame)
                matrix, registration = register_neighbor(decoded, donor, r["boxes"][frame], r["boxes"][donor_frame])
                local_matrix = matrix.copy()
                local_matrix[:, 2] -= [left, top]
                aligned = cv2.warpAffine(donor, local_matrix, (side, side), flags=cv2.INTER_LINEAR).astype(np.float32)
                invalid = exclusion_mask(gray.shape, r["boxes"][donor_frame])
                # Borders and every annotated donor target are excluded before use.
                donor_valid = cv2.warpAffine(255-invalid, local_matrix, (side, side),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 254
                ring = (reference_mask == 0) & donor_valid
                if ring.sum() < 0.5*ref.size:
                    raise BackgroundUnavailable("insufficient_valid_background_ring")
                valid = ring.copy()
                difference = ref-aligned
                for _ in range(3):
                    coef = np.linalg.lstsq(design[valid], difference[valid], rcond=None)[0]
                    residual = difference-design @ coef
                    std = max(0.7, float(np.median(np.abs(residual[valid])))*1.4826)
                    valid = ring & (np.abs(residual) < 3*std)
                adjusted = aligned+design @ coef
                mae = float(np.mean(np.abs((ref-adjusted)[ring])))
                p95 = float(np.percentile(np.abs((ref-adjusted)[ring]), 95))
                if mae > 3.0 or p95 > 8.0:
                    raise BackgroundUnavailable("local_background_agreement_failed")
                matte, contour = recover_contour(ref, adjusted, local_box)
                if not np.all(donor_valid[matte > 0]):
                    raise BackgroundUnavailable("donor_target_or_border_overlaps_repair")
                clean = ref*(1-matte)+adjusted*matte
                boundary = boundary_metrics(ref, clean, matte)
                if boundary["repair_boundary_mean_abs_change"] > 1.5:
                    raise BackgroundUnavailable("repair_boundary_change_too_large")
                metrics = dict(background_method="registered_real_neighbor_plus_foreground_contour",
                    donor_frame=donor_frame, donor_offset=offset, donor_video_sha256=r["video_sha256"],
                    approved_coco_sha256=r["coco_sha256"], approved_manifest_sha256=r["approved_manifest_sha256"],
                    source_decode_mae=source_error, ring_mae=mae, ring_p95_abs_error=p95,
                    donor_to_reference_affine=matrix.tolist(), donor_exclusion_coverage=1.0,
                    **registration, **contour, **boundary)
                score = mae + 0.001*abs(offset)
                if best is None or score < best[0]:
                    best = score, clean, matte, metrics
            except (BackgroundUnavailable, cv2.error) as e:
                errors.append(dict(offset=offset, reason=str(e)))
        if best is None:
            raise BackgroundUnavailable("no_safe_temporal_donor:" + json.dumps(errors))
        _, clean, matte, metrics = best
        metrics["rejected_candidates"] = errors
        metrics["donor_candidates_checked"] = len(self.offsets)
        self.cache[cache_key] = clean, matte, metrics
        return clean, matte, metrics
