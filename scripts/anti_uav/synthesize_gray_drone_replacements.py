#!/usr/bin/env python3
"""Conservative, deterministic target replacement in annotated grayscale frames.

No model downloads, diffusion, GPU use, or training-list edits. A catalog command
extracts anchored Excel pictures. Only usable native-alpha cutouts are eligible;
opaque product photographs are retained for later segmentation, not guessed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import posixpath
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.gray_temporal_background import BackgroundUnavailable, TemporalBackgrounds


NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "d": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_path(root: Path, relative: str) -> Path:
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {relative}")
    return p


def fresh_directory(path: Path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.mkdir(parents=True)


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "big")


def relationships(z: zipfile.ZipFile, part: str) -> dict:
    rel = posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")
    if rel not in z.namelist():
        return {}
    return {r.attrib["Id"]: posixpath.normpath(posixpath.join(posixpath.dirname(part),
            r.attrib["Target"])) if not r.attrib["Target"].startswith("/") else r.attrib["Target"].lstrip("/")
            for r in ET.fromstring(z.read(rel)) if r.attrib.get("TargetMode") != "External"}


def workbook_pictures(path: Path, sheet_name: str):
    """Read actual drawing relationships; media numbering need not match row order."""
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        sheet = next(s for s in wb.find("s:sheets", NS) if s.attrib["name"] == sheet_name)
        part = relationships(z, "xl/workbook.xml")[sheet.attrib[f"{{{NS['r']}}}id"]]
        rels = relationships(z, part)
        for drawing in ET.fromstring(z.read(part)).findall("s:drawing", NS):
            dp = rels[drawing.attrib[f"{{{NS['r']}}}id"]]
            dr = relationships(z, dp)
            for anchor in ET.fromstring(z.read(dp)):
                row = anchor.find("d:from/d:row", NS)
                blip = anchor.find("d:pic/d:blipFill/a:blip", NS)
                if row is None or blip is None:
                    continue
                media = dr[blip.attrib[f"{{{NS['r']}}}embed"]]
                yield int(row.text) + 1, media, z.read(media)


def native_cutout(image: np.ndarray):
    if image.ndim != 3 or image.shape[2] != 4 or image[:, :, 3].min() == 255:
        return None, "opaque_requires_segmentation", {}
    alpha = image[:, :, 3]
    hard = (alpha > 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hard)
    if count < 2:
        return None, "empty_alpha", {}
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.argmax()) + 1
    main_fraction = float(areas.max() / max(1, areas.sum()))
    # Separate accessory silhouettes cannot be reliably identified without semantics.
    if main_fraction < 0.90:
        return None, "multiple_components_requires_review", {"main_fraction": main_fraction}
    if np.count_nonzero(hard[0]) + np.count_nonzero(hard[-1]) + np.count_nonzero(hard[:, 0]) + np.count_nonzero(hard[:, -1]):
        return None, "foreground_touches_image_edge", {"main_fraction": main_fraction}
    halo = cv2.dilate((labels == largest).astype(np.uint8), np.ones((7, 7), np.uint8))
    matte = alpha * halo
    yy, xx = np.where(matte > 8)
    if not len(xx):
        return None, "empty_alpha", {}
    bounds = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    x0, y0, x1, y1 = bounds
    crop = image[y0:y1, x0:x1].copy()
    crop[:, :, 3] = matte[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 16:
        return None, "cutout_too_small", {"bounds": bounds}
    return crop, "native_alpha_candidate", {"bounds": bounds, "main_fraction": main_fraction}


def font(size: int):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def label(canvas: Image.Image, xy, text: str, size=17, color=(225, 230, 235)):
    ImageDraw.Draw(canvas).text(xy, text, fill=color, font=font(size))


def asset_grid(records: list[dict], root: Path):
    eligible = [r for r in records if r.get("cutout")]
    for start in range(0, len(eligible), 24):
        page = eligible[start:start + 24]
        canvas = Image.new("RGB", (1200, math.ceil(len(page) / 4) * 220), (28, 32, 38))
        for k, r in enumerate(page):
            x, y = (k % 4) * 300, (k // 4) * 220
            im = Image.open(root / r["cutout"]).convert("RGBA")
            im.thumbnail((280, 155))
            tile = Image.new("RGB", (280, 160), (170, 175, 180))
            tile.paste(im, ((280-im.width)//2, (160-im.height)//2), im)
            canvas.paste(tile, (x + 10, y + 8))
            label(canvas, (x + 10, y + 173), f"ID {r['id']}  {r['company']}"[:33], 16)
            label(canvas, (x + 10, y + 195), str(r["model"])[:36], 13)
        canvas.save(root / f"catalog_preview_{start//24 + 1:02d}.jpg", quality=95)


def catalog(args):
    import openpyxl
    book = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True)
    sheet = book[args.sheet] if args.sheet else book.worksheets[0]
    rows = list(sheet.values)
    pictures = list(workbook_pictures(args.xlsx, sheet.title))
    book.close()
    fresh_directory(args.output)
    (args.output / "raw").mkdir()
    (args.output / "cutouts").mkdir()
    records, seen = [], set()
    for row, media, data in pictures:
        values = rows[row - 1]
        if len(values) < 4 or not values[3]:
            raise ValueError(f"Picture at Excel row {row} has no model metadata")
        uid = str(values[0])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", uid) or uid in seen:
            raise ValueError(f"Invalid/duplicate catalog ID at row {row}: {uid}")
        seen.add(uid)
        raw = f"raw/{uid}.png"
        im = Image.open(io.BytesIO(data)).convert("RGBA")
        im.save(args.output / raw)
        cutout, status, detail = native_cutout(np.asarray(im))
        r = dict(id=uid, excel_row=row, company=values[1], category=values[2],
                 model=values[3], physical_size=values[4], raw=raw, media=media,
                 embedded_sha256=hashlib.sha256(data).hexdigest(), status=status, **detail)
        if cutout is not None:
            r["cutout"] = f"cutouts/{uid}.png"
            Image.fromarray(cutout).save(args.output / r["cutout"])
            r["cutout_sha256"] = sha256(args.output / r["cutout"])
        records.append(r)
    result = dict(xlsx=str(args.xlsx.resolve()), xlsx_sha256=sha256(args.xlsx), records=records,
                  note="Native-alpha candidates still require semantic/pose/license review; not all catalog rows are usable.")
    dump(args.output / "catalog.json", result)
    asset_grid(records, args.output)
    print(json.dumps({"pictures": len(records), "native_alpha_candidates": sum("cutout" in r for r in records),
                      "output": str(args.output.resolve())}), flush=True)


def parse_boxes(text: str, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.split()]
        if len(v) != 5 or not all(math.isfinite(x) for x in v):
            raise ValueError(f"Invalid YOLO label: {line}")
        cls, cx, cy, w, h = v
        if cls != 0 or min(w, h) <= 0 or min(cx-w/2, cy-h/2) < -1e-6 or max(cx+w/2, cy+h/2) > 1+1e-6:
            raise ValueError(f"Invalid normalized drone box: {line}")
        boxes.append([(cx-w/2)*width, (cy-h/2)*height, w*width, h*height])
    return boxes


def normalize_box(box, width, height) -> str:
    x, y, w, h = box
    return f"0 {(x+w/2)/width:.9f} {(y+h/2)/height:.9f} {w/width:.9f} {h/height:.9f}\n"


class SkipSample(ValueError):
    pass


def prepare_target(gray: np.ndarray, box: list[float], seed: int,
                   max_ring_std=6.0, blur_sigma: float | None = None,
                   temporal=None, video_id=None, frame_index=None):
    """Compute the asset-independent background once, for offline or online use."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected uint8 grayscale frame")
    x, y, bw, bh = box
    edge = max(bw, bh)
    if min(bw, bh) < 3 or edge > 160:
        raise SkipSample("target_size_outside_3_to_160px_prototype_range")
    # Image-plane blur is not proportional to the target's size. A deliberately
    # modest fixed default avoids overblurring larger silhouettes; calibrate it
    # against the camera when an actual PSF/exposure estimate becomes available.
    sigma = float(blur_sigma if blur_sigma is not None else 0.65)
    if not math.isfinite(sigma) or not 0 < sigma <= 5:
        raise ValueError("blur_sigma must be in (0, 5]")
    pad = max(3, int(math.ceil(3*sigma)), int(math.ceil(edge*0.2)))
    radius = int(math.ceil(edge/2)) + pad + max(12, int(edge*0.35))
    cx, cy = x+bw/2, y+bh/2
    left, top = int(math.floor(cx))-radius, int(math.floor(cy))-radius
    side = 2*radius + 1
    if min(left, top) < 0 or left+side > gray.shape[1] or top+side > gray.shape[0]:
        raise SkipSample("insufficient_context_at_frame_edge")
    patch = gray[top:top+side, left:left+side]
    xx0, yy0 = int(math.floor(x-left)), int(math.floor(y-top))
    xx1, yy1 = int(math.ceil(x+bw-left)), int(math.ceil(y+bh-top))
    erase = np.zeros(patch.shape, np.uint8)
    erase[yy0-pad:yy1+pad, xx0-pad:xx1+pad] = 255
    ring = erase == 0
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    design = np.stack((np.ones_like(xx), xx/side, yy/side), axis=-1)
    valid = ring.copy()
    for _ in range(3):
        coef = np.linalg.lstsq(design[valid], patch[valid].astype(np.float32), rcond=None)[0]
        residual = patch.astype(np.float32) - design @ coef
        robust_std = max(0.5, float(np.median(np.abs(residual[valid])))*1.4826)
        valid = ring & (np.abs(residual) < 3*robust_std)
    ring_std = float(np.std(residual[ring]))
    if ring_std > max_ring_std:
        raise SkipSample(f"textured_background_ring_std_{ring_std:.2f}")
    target_residual = residual[yy0:yy1, xx0:xx1]
    lo, hi = np.percentile(target_residual, [5, 95])
    sign = -1 if abs(lo) >= abs(hi) else 1
    contrast = float(np.percentile(target_residual * sign, 90))
    fg = target_residual[(target_residual * sign) > max(2.0, 2*ring_std, contrast*0.3)]
    if len(fg) < 3 or contrast < max(4.0, 3*ring_std):
        raise SkipSample("foreground_contrast_not_reliable")
    flo, fhi = np.percentile(fg, [5, 95])
    background_metrics = dict(background_method="legacy_sky_plane_plus_independent_noise")
    if temporal is not None:
        try:
            clean, matte, background_metrics = temporal.reconstruct(
                gray, box, video_id, frame_index, (left, top, side))
        except BackgroundUnavailable as e:
            raise SkipSample(str(e)) from e
        erase = (matte > 0).astype(np.uint8)*255
    else:
        # Explicit legacy ablation only. The CLI now requires temporal donors by
        # default; a missing/unsafe donor must not silently fall back to this fill.
        rng = np.random.default_rng(seed)
        noise = rng.choice(residual[valid] - np.mean(residual[valid]), size=patch.shape)
        distance = cv2.distanceTransform(erase, cv2.DIST_L2, 5)
        blend = np.clip(distance/min(3, max(1, pad-1)), 0, 1)
        clean = patch.astype(np.float32)*(1-blend) + (design @ coef + noise)*blend
    return dict(patch=patch.copy(), clean=clean, erase=erase,
        meta=dict(box=box, left=left, top=top, side=side, cx=cx, cy=cy, edge=edge,
                  sigma=sigma, ring_std=ring_std, contrast=contrast, sign=sign,
                  flo=float(flo), fhi=float(fhi), image_hw=list(gray.shape),
                  background_metrics=background_metrics))


def render_prepared(prepared, rgba):
    """Render only a small ROI. No video decoding or registration in a train worker."""
    patch, clean, erase = (prepared[k] for k in ("patch", "clean", "erase"))
    m = prepared["meta"]
    box, left, top, side, cx, cy, edge, sigma, ring_std, contrast, sign, flo, fhi = (
        m[k] for k in ("box", "left", "top", "side", "cx", "cy", "edge", "sigma",
                       "ring_std", "contrast", "sign", "flo", "fhi"))
    alpha = rgba[:, :, 3].astype(np.float32)/255
    lum = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    values = lum[alpha > 0.8]
    if len(values) < 10:
        raise SkipSample("insufficient_solid_asset_pixels")
    qlo, qhi = np.percentile(values, [5, 95])
    shade = np.clip((lum-qlo)/max(1.0, qhi-qlo), 0, 1)
    foreground_delta = flo + shade*(fhi-flo)
    ah, aw = alpha.shape
    scale = edge / max(aw, ah)
    tw, th = aw*scale, ah*scale
    # Supersample with premultiplied alpha; transparent RGB must never leak halos.
    ss = 4
    matrix = np.array([[scale*ss, 0, (cx-left-tw/2)*ss + scale*ss/2 - 0.5],
                       [0, scale*ss, (cy-top-th/2)*ss + scale*ss/2 - 0.5]], np.float32)
    def project(a):
        high = cv2.warpAffine(a, matrix, (side*ss, side*ss), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        low = cv2.resize(high, (side, side), interpolation=cv2.INTER_AREA)
        kernel = 2*int(math.ceil(3*sigma))+1
        return cv2.GaussianBlur(low, (kernel, kernel), sigma)
    a = project(alpha)
    delta = project(foreground_delta * alpha)
    if a.max() < 0.25:
        raise SkipSample("asset_disappeared_after_downsampling")
    core = a > a.max()*0.55
    rendered_contrast = float(np.percentile(delta[core] * sign, 90))
    gain = contrast/max(0.01, rendered_contrast)
    if not 0.4 <= gain <= 3.0:
        raise SkipSample("contrast_gain_out_of_range")
    # A contrast residual is added relative to the locally restored background.
    composed = np.clip(np.rint(clean + delta*gain), 0, 255).astype(np.uint8)
    changed = (erase > 0) | (a > 0)
    composed[~changed] = patch[~changed]
    mask = changed.astype(np.uint8)*255
    outside = int(np.count_nonzero(composed[~changed] != patch[~changed]))
    if outside:
        raise AssertionError("Pixels outside the edit mask changed")
    visible = a > 0.15
    vy, vx = np.where(visible)
    newbox = [int(vx.min())+left, int(vy.min())+top,
              int(vx.max()-vx.min()+1), int(vy.max()-vy.min()+1)]
    measured = float(np.percentile((composed.astype(np.float32)-clean)[core]*sign, 90))
    if abs(measured-contrast) > max(2.0, contrast*0.2):
        raise SkipSample("contrast_matching_failed_after_clipping")
    metrics = dict(original_box_xywh=box, new_box_xywh=newbox,
                   placement_center_xy=[cx, cy], geometric_size_wh=[tw, th],
                   long_edge_error_px=abs(max(tw, th)-edge), blur_sigma_px=sigma,
                   blur_method="fixed_sigma_approximation_not_measured_PSF", ring_std=ring_std,
                   source_contrast=contrast, output_contrast=measured, contrast_gain=gain,
                   foreground_residual_p05_p95=[float(flo), float(fhi)],
                   changed_pixels=int(np.count_nonzero(composed != patch)),
                   outside_mask_changed_pixels=outside,
                   label_policy="bbox of blurred alpha > 0.15; may differ from geometric size")
    metrics.update(m["background_metrics"])
    return composed, mask, newbox, metrics


def replace_target(gray: np.ndarray, box: list[float], rgba: np.ndarray, seed: int,
                   max_ring_std=6.0, blur_sigma: float | None = None,
                   temporal=None, video_id=None, frame_index=None):
    """Approximate a smooth-sky replacement; reject hard cases, do not hallucinate."""
    prepared = prepare_target(gray, box, seed, max_ring_std, blur_sigma, temporal, video_id, frame_index)
    composed, local_mask, newbox, metrics = render_prepared(prepared, rgba)
    left, top, side = (prepared["meta"][k] for k in ("left", "top", "side"))
    result, mask = gray.copy(), np.zeros_like(gray)
    result[top:top+side, left:left+side] = composed
    mask[top:top+side, left:left+side] = local_mask
    return result, mask, newbox, metrics


def validate_samples(manifest: dict, root: Path, blocked: list[str]):
    seen = set()
    for sample in manifest["samples"]:
        # Dataset roots may be named strict_holdout_Video00004 even when their
        # images belong to Video00001. Check video identities, not that root name.
        token = " ".join((str(sample["video"]), Path(sample["image"]).parent.name,
                          Path(sample.get("source_image", "")).parent.name))
        if any(b.lower() in token.lower() for b in blocked):
            raise ValueError(f"Held-out source rejected: {token}")
        image = safe_path(root, sample["image"])
        lab = safe_path(root, sample["label"])
        if image in seen:
            raise ValueError(f"Duplicate source image: {image}")
        seen.add(image)
        if not image.is_file() or not lab.is_file():
            raise FileNotFoundError(image if not image.is_file() else lab)
        if "sha256" not in sample or sha256(image) != sample["sha256"]:
            raise ValueError(f"Missing/mismatched source image SHA256: {image}")


def draw_corner_box(image: Image.Image, box, scale_xy=(1.0, 1.0), origin_xy=(0.0, 0.0),
                    color=(40, 210, 255)):
    """Draw after resizing so the visualization outline stays one display pixel."""
    x, y, w, h = box
    sx, sy = scale_xy
    ox, oy = origin_xy
    x0, y0 = round((x-ox)*sx), round((y-oy)*sy)
    x1, y1 = max(x0, round((x+w-ox)*sx)-1), max(y0, round((y+h-oy)*sy)-1)
    length = max(1, min(8, min(x1-x0, y1-y0)//4))
    draw = ImageDraw.Draw(image)
    for px, py, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                           (x0, y1, 1, -1), (x1, y1, -1, -1)):
        draw.line([(px+dx*min(length, x1-x0), py), (px, py),
                   (px, py+dy*min(length, y1-y0))], fill=color, width=1)


def preview_card(original: np.ndarray, result: np.ndarray, box, info: dict,
                 asset: Image.Image, new_box=None):
    new_box = info["metrics"]["new_box_xywh"] if new_box is None else new_box
    panels = (("Original", original, box, (85, 220, 135)),
              ("Replaced", result, new_box, (40, 210, 255)))
    canvas = Image.new("RGB", (1200, 780), (25, 29, 35))
    label(canvas, (20, 10), f"{info['video'][:70]} / frame {info['frame']} / asset {info['asset_id']}", 19)
    label(canvas, (20, 40), f"{info['asset_model'][:90]} | GRAYSCALE SYNTHESIS / NOT A REAL CAPTURE", 16)
    for k, (name, array, bbox, color) in enumerate(panels):
        full = Image.fromarray(array).convert("RGB")
        full.thumbnail((580, 326))
        draw_corner_box(full, bbox, (full.width/array.shape[1], full.height/array.shape[0]), color=color)
        canvas.paste(full, (10+k*600, 96))
        label(canvas, (15+k*600, 70), name.upper() + " / " + ("original bbox" if k == 0 else "updated bbox"), 18, color)
    x, y, w, h = box
    span = max(48, int(math.ceil(max(w, h)*2.0)))
    l = max(0, min(original.shape[1]-span, int(x+w/2-span/2)))
    t = max(0, min(original.shape[0]-span, int(y+h/2-span/2)))
    for k, (name, array, bbox, color) in enumerate(panels):
        crop = Image.fromarray(array[t:t+span, l:l+span]).convert("RGB")
        crop = crop.resize((288, 288), Image.Resampling.NEAREST)
        draw_corner_box(crop, bbox, (288/span, 288/span), (l, t), color)
        canvas.paste(crop, (12+k*302, 462))
        label(canvas, (12+k*302, 434), f"{name} / {288/span:.1f}x pixel view", 16, color)
    asset = asset.copy().convert("RGBA")
    asset.thumbnail((260, 190))
    tile = Image.new("RGB", (270, 205), (160, 165, 170))
    tile.paste(asset, ((270-asset.width)//2, (205-asset.height)//2), asset)
    canvas.paste(tile, (620, 462))
    label(canvas, (620, 434), "Source cutout / enlarged", 16)
    m = info["metrics"]
    lines = ["Geometry long edge:", f"{max(w,h):.2f} px -> {max(m['geometric_size_wh']):.2f} px",
             "Label bbox W x H:", f"{w:.1f} x {h:.1f} -> {new_box[2]:.1f} x {new_box[3]:.1f}",
             "Foreground contrast:", f"{m['source_contrast']:.1f} -> {m['output_contrast']:.1f}",
             f"Blur sigma: {m['blur_sigma_px']:.2f} px", "Outside mask: 0 changes",
             "1px corners / no cross"]
    for k, line in enumerate(lines):
        label(canvas, (910, 463+k*29), line, 16)
    if "donor_frame" in m:
        label(canvas, (620, 690), f"BG: real frame {m['donor_frame']} ({m['donor_offset']:+d})", 15)
        label(canvas, (620, 716), "Contour repair / no independent noise fill", 14)
    else:
        label(canvas, (620, 690), "BG: legacy rectangle + independent noise", 14)
        label(canvas, (620, 716), "Review only / not seamless background", 14)
    return canvas


def preview_overview(root: Path, previews: list[str]):
    if not previews:
        return
    thumb = Image.new("RGB", (1200, math.ceil(len(previews)/2)*390), (25, 29, 35))
    for n, p in enumerate(previews):
        card = Image.open(root / p).resize((600, 390), Image.Resampling.LANCZOS)
        thumb.paste(card, ((n % 2)*600, (n//2)*390))
    thumb.save(root / "preview_contact_sheet.jpg", quality=95)


def render_previews(args):
    """Render existing saved labels/images without regenerating training data."""
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    if sha256(args.catalog) != manifest["catalog_sha256"]:
        raise ValueError("Catalog differs from the one used for synthesis")
    assets = {a["id"]: a for a in json.loads(args.catalog.read_text())["records"]}
    candidates = [s for s in manifest["samples"] if s["status"] == "replaced"]
    selected = [s for s in candidates if s.get("preview")] or candidates
    if args.limit < 1:
        raise ValueError("limit must be positive")
    fresh_directory(args.output)
    records, previews = [], []
    for info in selected[:args.limit]:
        source, output = Path(info["source_image"]), safe_path(root, info["image"])
        target_label = safe_path(root, info["label"])
        if sha256(source) != info["source_sha256"] or sha256(output) != info["output_sha256"]:
            raise ValueError("Source/output image changed since synthesis")
        original = np.asarray(Image.open(source).convert("L"))
        result = np.asarray(Image.open(output).convert("L"))
        boxes = parse_boxes(target_label.read_text(), result.shape[1], result.shape[0])
        if len(boxes) != 1 or not np.allclose(boxes[0], info["metrics"]["new_box_xywh"], atol=1e-4, rtol=0):
            raise ValueError(f"Saved YOLO label differs from synthesis manifest: {target_label}")
        asset = assets[info["asset_id"]]
        cutout = safe_path(args.catalog.parent, asset["cutout"])
        if sha256(cutout) != asset["cutout_sha256"]:
            raise ValueError("Cutout checksum changed")
        card = preview_card(original, result, info["metrics"]["original_box_xywh"], info,
                            Image.open(cutout), new_box=boxes[0])
        name = f"{len(previews)+1:02d}_{output.stem}_bbox.png"
        card.save(args.output / name)
        previews.append(name)
        records.append(dict(preview=name, saved_label=str(target_label.resolve()),
                            label_sha256=sha256(target_label), rendered_box_xywh=boxes[0],
                            original_box_xywh=info["metrics"]["original_box_xywh"]))
    preview_overview(args.output, previews)
    dump(args.output / "preview_manifest.json", dict(source_manifest=str(args.manifest.resolve()),
         source_manifest_sha256=sha256(args.manifest), training_files_modified=False,
         style="1px corners after resize; green original bbox, cyan saved synthetic bbox; no cross",
         previews=records))
    print(json.dumps({"previews": len(previews), "training_files_modified": False}), flush=True)


def synthesize(args):
    if args.variants < 1 or args.preview_count < 0 or args.max_ring_std <= 0:
        raise ValueError("Invalid variants/preview_count/max_ring_std")
    source = json.loads(args.source_manifest.read_text())
    root = args.source_root or args.source_manifest.parent
    validate_samples(source, root, ["Video00004"] + args.exclude_video)
    mode = getattr(args, "background_mode", "temporal")
    if mode == "temporal" and not getattr(args, "temporal_registry", None):
        raise ValueError("Temporal background mode requires --temporal-registry; no synthetic-fill fallback")
    temporal = TemporalBackgrounds(args.temporal_registry, ["Video00004"] + args.exclude_video) if mode == "temporal" else None
    catalog_data = json.loads(args.catalog.read_text())
    eligible = {a["id"]: a for a in catalog_data["records"] if a.get("cutout")}
    ids = args.asset_ids.split(",") if args.asset_ids else sorted(eligible, key=lambda s: int(s))
    if not ids or len(set(ids)) != len(ids) or any(i not in eligible for i in ids):
        raise ValueError("asset-ids must be unique eligible catalog IDs")
    assets = []
    for uid in ids:
        a = eligible[uid]
        path = safe_path(args.catalog.parent, a["cutout"])
        if sha256(path) != a["cutout_sha256"]:
            raise ValueError(f"Cutout checksum changed: {path}")
        assets.append((a, np.asarray(Image.open(path).convert("RGBA"))))
    fresh_directory(args.output)
    for directory in ("images", "labels", "masks", "previews"):
        (args.output / directory).mkdir()
    samples, previews, per_video, assigned = [], [], {}, {}
    started = time.perf_counter()
    for sample in source["samples"]:
        image_path = safe_path(root, sample["image"])
        label_path = safe_path(root, sample["label"])
        gray = np.asarray(Image.open(image_path).convert("L"))
        h, w = gray.shape
        text = label_path.read_text()
        boxes = parse_boxes(text, w, h)
        video = str(sample["video"])
        frame = image_path.stem
        key = hashlib.sha256(f"{video}/{sample['image']}".encode()).hexdigest()[:16]
        offset = stable_seed(args.seed, video) % len(assets)
        for variant in range(args.variants):
            a, rgba = assets[(offset+variant) % len(assets)]
            assigned.setdefault(video, {})[str(variant)] = a["id"]
            stem = f"{key}_v{variant:02d}"
            info = dict(video=video, frame=frame, variant=variant, asset_id=a["id"],
                        asset_model=a["model"], source_image=str(image_path.resolve()),
                        source_sha256=sample["sha256"], source_label_sha256=sha256(label_path),
                        image=f"images/{stem}.png", label=f"labels/{stem}.txt",
                        edit_mask=f"masks/{stem}.png")
            output, mask, output_label = gray, np.zeros_like(gray), text
            try:
                if not boxes:
                    info["status"] = "negative_preserved"
                elif len(boxes) != 1:
                    raise SkipSample("multiple_targets_not_supported_in_prototype")
                else:
                    output, mask, newbox, metrics = replace_target(
                        gray, boxes[0], rgba, stable_seed(args.seed, key, variant),
                        args.max_ring_std, args.blur_sigma, temporal, video, int(frame))
                    output_label = normalize_box(newbox, w, h)
                    info.update(status="replaced", metrics=metrics)
            except SkipSample as e:
                info.update(status="positive_preserved", skip_reason=str(e))
            Image.fromarray(output).save(args.output / info["image"])
            Image.fromarray(mask).save(args.output / info["edit_mask"])
            (args.output / info["label"]).write_text(output_label)
            info["output_sha256"] = sha256(args.output / info["image"])
            # Read back the actual PNG, not just an in-memory calculation.
            decoded = np.asarray(Image.open(args.output / info["image"]))
            if decoded.shape != gray.shape or not np.array_equal(decoded[mask == 0], gray[mask == 0]):
                raise AssertionError("Saved PNG changed outside the edit mask")
            info["saved_png_outside_mask_changed_pixels"] = 0
            samples.append(info)
            if info["status"] == "replaced" and len(previews) < args.preview_count and per_video.get(video, 0) < 2 and variant == 0:
                saved_boxes = parse_boxes((args.output / info["label"]).read_text(), w, h)
                card = preview_card(gray, output, boxes[0], info, Image.fromarray(rgba), new_box=saved_boxes[0])
                pp = f"previews/{len(previews)+1:02d}_{stem}.png"
                card.save(args.output / pp)
                previews.append(pp)
                per_video[video] = per_video.get(video, 0) + 1
                info["preview"] = pp
        print(json.dumps({"frame": frame, "video": video, "outputs": len(samples)}), flush=True)
    counts = {s: sum(r["status"] == s for r in samples)
              for s in ("replaced", "positive_preserved", "negative_preserved")}
    if temporal is not None:
        temporal.close()
    result = dict(source_manifest=str(args.source_manifest.resolve()),
                  source_manifest_sha256=sha256(args.source_manifest), catalog_sha256=sha256(args.catalog),
                  seed=args.seed, variants=args.variants, assigned_assets=assigned, counts=counts,
                  seconds=time.perf_counter()-started, previews=previews, samples=samples,
                  training_lists_modified=False, eligible_asset_ids=ids, background_mode=mode,
                  temporal_registry_sha256=sha256(args.temporal_registry) if temporal is not None else None,
                  limitations=["Frame prototype, not optical-flow/video-consistent synthesis.",
                               "Hidden background and blur are approximated, not measured ground truth.",
                               "Native alpha does not certify asset identity, viewpoint, or licensing.",
                               "Holdout name guard and input hashes do not detect renamed/overlapping videos.",
                               "Outputs are review candidates; do not concatenate all variants into training without rebalancing."])
    dump(args.output / "manifest.json", result)
    preview_overview(args.output, previews)
    print(json.dumps({"counts": counts, "previews": previews, "seconds": result["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("catalog", help="Extract embedded pictures and conservative native-alpha candidates")
    c.add_argument("--xlsx", type=Path, required=True)
    c.add_argument("--sheet")
    c.add_argument("--output", type=Path, required=True)
    c.set_defaults(func=catalog)
    s = sub.add_parser("synthesize", help="Generate lossless frame variants and visual QA")
    s.add_argument("--catalog", type=Path, required=True)
    s.add_argument("--source-manifest", type=Path, required=True)
    s.add_argument("--source-root", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--asset-ids", help="Comma-separated eligible IDs; default all native-alpha candidates")
    s.add_argument("--variants", type=int, default=1)
    s.add_argument("--preview-count", type=int, default=8)
    s.add_argument("--seed", type=int, default=20260914)
    s.add_argument("--exclude-video", action="append", default=[])
    s.add_argument("--max-ring-std", type=float, default=6.0)
    s.add_argument("--blur-sigma", type=float, help="Override 0.65px approximate blur, in original-frame pixels")
    s.add_argument("--background-mode", choices=["temporal", "legacy-plane"], default="temporal")
    s.add_argument("--temporal-registry", type=Path, help="Original videos + approved manifest/COCO paths")
    s.set_defaults(func=synthesize)
    p = sub.add_parser("preview", help="Draw original/updated bboxes from an existing synthesis run")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=render_previews)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.func(args)


if __name__ == "__main__":
    main()
