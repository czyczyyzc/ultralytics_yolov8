import json

import numpy as np
import pytest

from scripts.anti_uav.online_random_scale import eligible_donors, random_context_crop, gray_capture_jitter


def test_donor_selection_is_deduplicated_and_excludes_val_and_tiny():
    def row(path, width=.1, height=.1):
        return dict(im_file=path, shape=(1080, 1920), bboxes=np.array([[.5, .5, width, height]]))
    records = [row('/images/train/1.jpg'), row('/images/train/1.jpg'), row('/images/train/2.jpg'),
               row('/images/gray_val/3.jpg'), row('/images/rgb/4.jpg'), row('/images/train/5.jpg', .003, .003)]
    donors = eligible_donors(records)
    assert [r['im_file'] for r in donors] == ['/images/train/1.jpg', '/images/train/2.jpg']
    assert len(records) == 6


def test_context_randomizes_but_keeps_geometry_and_pixel_budget():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    box = np.array([700., 300., 1020., 580.])
    rng = np.random.default_rng(7)
    draws = [random_context_crop(image, box, rng) for _ in range(100)]
    assert len({tuple(r[2]['crop_xywh']) for r in draws}) > 90
    assert any(r[2]['area_fraction'] >= .8 for r in draws)
    for output, updated, meta in draws:
        assert output.shape == (544, 960, 3)
        assert meta['upscale'] <= 4 and meta['retained_fraction'] >= .6
        x, y, w, h = meta['crop_xywh']
        expected = box-[x, y, x, y]
        expected[[0, 2]] = expected[[0, 2]].clip(0, w)
        expected[[1, 3]] = expected[[1, 3]].clip(0, h)
        np.testing.assert_allclose(updated, expected*[960/w, 544/h, 960/w, 544/h])
        json.dumps(meta)


def test_small_eligible_donor_is_not_forced_to_fill_frame():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for seed in range(20):
        a = random_context_crop(image, [500, 500, 596, 548], np.random.default_rng(seed))
        b = random_context_crop(image, [500, 500, 596, 548], np.random.default_rng(seed))
        assert a[2] == b[2]
        assert a[2]['area_fraction'] <= 96*48*16/(960*544)


def test_intact_and_border_targets_are_not_dropped():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for seed in range(20):
        _, _, meta = random_context_crop(image, [0, 0, 220, 120], np.random.default_rng(seed), partial_probability=0)
        assert meta['retained_fraction'] == 1
    with pytest.raises(ValueError):
        random_context_crop(image, [0, 0, 8, 4], np.random.default_rng(0))


def test_gray_jitter_preserves_channels_shape_and_dtype():
    result = gray_capture_jitter(np.full((100, 100, 3), 120, dtype=np.uint8), np.random.default_rng(5))
    assert result.dtype == np.uint8 and result.shape == (100, 100, 3)
    np.testing.assert_array_equal(result[:, :, 0], result[:, :, 2])
