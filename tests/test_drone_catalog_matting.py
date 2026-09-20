import numpy as np
import pytest

from scripts.anti_uav.matte_drone_catalog import pack_alpha


def test_rgb_and_thin_parts_preserved():
    rgb = np.random.default_rng(1).integers(0, 256, (30, 40, 3), dtype=np.uint8)
    a = np.zeros((30, 40), np.uint8)
    a[8:20, 10:30] = 255
    a[6:8, 20] = 160
    rgba, bounds, quality = pack_alpha(rgb, a)
    assert np.array_equal(rgba[..., :3], rgb)
    assert np.array_equal(rgba[..., 3], a)
    assert bounds == [10, 6, 30, 20]
    assert quality["significant_components"] == 1


def test_empty_and_low_alpha():
    rgba, bounds, quality = pack_alpha(np.zeros((10, 10, 3), np.uint8), np.full((10, 10), 3, np.uint8))
    assert bounds is None
    assert not rgba.any()
    assert "very_small_or_empty_mask" in quality["flags"]


def test_multiple_objects_flagged_not_deleted():
    a = np.zeros((100, 100), np.uint8)
    a[10:30, 10:30] = 255
    a[50:80, 60:80] = 255
    rgba, _, quality = pack_alpha(np.zeros((100, 100, 3), np.uint8), a)
    assert np.array_equal(rgba[..., 3], a)
    assert "multiple_components_check_accessories" in quality["flags"]


def test_invalid_size_rejected():
    with pytest.raises(ValueError):
        pack_alpha(np.zeros((10, 10, 3), np.uint8), np.zeros((5, 5), np.uint8))
