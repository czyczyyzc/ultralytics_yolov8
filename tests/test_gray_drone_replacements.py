import argparse
import json
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.drawing.image import Image as XLImage
from PIL import Image
import pytest

from scripts.anti_uav.synthesize_gray_drone_replacements import (
    SkipSample, draw_corner_box, native_cutout, normalize_box, parse_boxes, replace_target,
    render_previews, safe_path, sha256, stable_seed, synthesize, validate_samples, workbook_pictures,
)


def asset():
    rgba = np.zeros((64, 96, 4), np.uint8)
    rgba[24:40, 8:88] = [60, 60, 60, 255]
    rgba[12:52, 38:58] = [90, 90, 90, 255]
    return rgba


def scene():
    rng = np.random.default_rng(12)
    gray = np.clip(150 + rng.normal(0, 0.7, (240, 320)), 0, 255).astype(np.uint8)
    gray[112:128, 148:172] = 100
    return gray, [148.0, 112.0, 24.0, 16.0]


def test_drawing_maps_by_anchor_not_media_number(tmp_path):
    im = tmp_path / "a.png"
    Image.fromarray(asset()).save(im)
    wb = openpyxl.Workbook()
    wb.active.title = "drones"
    wb.active.add_image(XLImage(im), "F9")
    wb.active.add_image(XLImage(im), "F3")
    path = tmp_path / "a.xlsx"
    wb.save(path)
    assert [r[0] for r in workbook_pictures(path, "drones")] == [9, 3]


def test_native_alpha_is_conservative():
    crop, status, _ = native_cutout(asset())
    assert status == "native_alpha_candidate"
    assert crop.shape[:2] == (40, 80)
    opaque = asset()
    opaque[:, :, 3] = 255
    assert native_cutout(opaque)[1] == "opaque_requires_segmentation"
    multi = asset()
    multi[1:20, 1:30] = [50, 50, 50, 255]
    assert native_cutout(multi)[1] == "multiple_components_requires_review"


def test_replacement_preserves_geometry_background_and_contrast():
    gray, box = scene()
    rgba = native_cutout(asset())[0]
    result, mask, newbox, m = replace_target(gray, box, rgba, 123)
    assert result.dtype == np.uint8 and result.shape == gray.shape
    assert np.array_equal(result[mask == 0], gray[mask == 0])
    assert np.any(result != gray)
    assert m["long_edge_error_px"] < 1e-9
    assert abs(m["source_contrast"]-m["output_contrast"]) < 2
    assert abs(newbox[0]+newbox[2]/2-160) <= 1
    assert abs(newbox[1]+newbox[3]/2-120) <= 1
    assert np.array_equal(result, replace_target(gray, box, rgba, 123)[0])
    parsed = parse_boxes(normalize_box(newbox, 320, 240), 320, 240)[0]
    assert np.max(np.abs(np.array(parsed)-newbox)) < 1e-5


def test_transparent_color_cannot_leak():
    gray, box = scene()
    one = native_cutout(asset())[0]
    two = one.copy()
    two[two[:, :, 3] == 0, :3] = 255
    assert np.array_equal(replace_target(gray, box, one, 1)[0], replace_target(gray, box, two, 1)[0])


def test_reject_unreliable_context():
    gray, box = scene()
    rgba = native_cutout(asset())[0]
    with pytest.raises(SkipSample, match="foreground_contrast"):
        replace_target(np.full_like(gray, 150), box, rgba, 0)
    with pytest.raises(SkipSample, match="frame_edge"):
        replace_target(gray, [0, 0, 8, 8], rgba, 0)
    with pytest.raises(SkipSample, match="textured"):
        replace_target(np.random.default_rng(1).integers(0, 255, gray.shape, dtype=np.uint8), box, rgba, 0)


@pytest.mark.parametrize("text", ["0 nan .5 .1 .1", "1 .5 .5 .1 .1", "0 .99 .5 .2 .2", "0 .5 .5 0 .1"])
def test_bad_labels_rejected(text):
    with pytest.raises(ValueError):
        parse_boxes(text, 320, 240)


def test_source_guard_and_determinism(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_path(tmp_path, "../outside")
    with pytest.raises(ValueError, match="Held-out"):
        validate_samples({"samples": [{"video": "Video00004", "image": "a", "label": "b"}]}, tmp_path, ["Video00004"])
    assert stable_seed(2, "video", 1) == stable_seed(2, "video", 1)
    assert stable_seed(2, "video", 1) != stable_seed(2, "video", 2)


def test_corners_draw_after_scaling_without_cross():
    im = Image.new("RGB", (160, 120), (100, 100, 100))
    color = (40, 210, 255)
    draw_corner_box(im, [20, 30, 16, 12], scale_xy=(3, 3), origin_xy=(10, 20), color=color)
    data = np.asarray(im)
    assert tuple(data[30, 30]) == color
    assert tuple(data[65, 77]) == color
    assert tuple(data[48, 54]) == (100, 100, 100)
    assert tuple(data[31, 34]) == (100, 100, 100)  # not a 3px enlarged line
    assert tuple(data[30, 54]) == (100, 100, 100)  # corners, not a closed rectangle


def test_end_to_end_outputs_and_negative_preservation(tmp_path):
    gray, box = scene()
    sources = []
    for index in range(2):
        p = tmp_path / f"{index}.png"
        Image.fromarray(gray if index == 0 else np.full_like(gray, 150)).save(p)
        lab = tmp_path / f"{index}.txt"
        lab.write_text(normalize_box(box, 320, 240) if index == 0 else "")
        sources.append(dict(video="Video00001", image=p.name, label=lab.name, sha256=sha256(p),
                            source_image=f"/data/strict_holdout_Video00004/images/Video00001/{p.name}"))
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(dict(samples=sources)))
    a = tmp_path / "asset.png"
    Image.fromarray(native_cutout(asset())[0]).save(a)
    c = tmp_path / "catalog.json"
    c.write_text(json.dumps(dict(records=[dict(id="1", model="test", cutout=a.name, cutout_sha256=sha256(a))])))
    args = argparse.Namespace(source_manifest=manifest, source_root=None, catalog=c,
                              output=tmp_path/"out", variants=2, preview_count=0,
                              max_ring_std=6.0, blur_sigma=None, seed=1, exclude_video=[], asset_ids="1")
    synthesize(args)
    result = json.loads((args.output/"manifest.json").read_text())
    assert result["counts"] == dict(replaced=2, positive_preserved=0, negative_preserved=2)
    assert all(s["saved_png_outside_mask_changed_pixels"] == 0 for s in result["samples"])
    for s in result["samples"]:
        if s["status"] == "negative_preserved":
            assert not (args.output/s["label"]).read_text()
            assert np.all(np.asarray(Image.open(args.output/s["image"])) == 150)
    with pytest.raises(FileExistsError):
        synthesize(args)
    before = {p: sha256(p) for p in args.output.rglob("*") if p.is_file()}
    pa = argparse.Namespace(manifest=args.output/"manifest.json", catalog=c,
                            output=tmp_path/"bbox_previews", limit=2)
    render_previews(pa)
    rendered = json.loads((pa.output/"preview_manifest.json").read_text())
    assert len(rendered["previews"]) == 2
    assert not rendered["training_files_modified"]
    assert before == {p: sha256(p) for p in before}
    assert Image.open(pa.output/rendered["previews"][0]["preview"]).size == (1200, 780)
    (args.output/result["samples"][0]["label"]).write_text("0 .5 .5 .1 .1\n")
    pa.output = tmp_path/"mismatched_label_previews"
    with pytest.raises(ValueError, match="Saved YOLO label differs"):
        render_previews(pa)
