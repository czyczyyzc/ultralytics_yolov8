import numpy as np
import pytest

from scripts.anti_uav.import_fixedwing_assets import reviewed_alpha_cleanup, support_envelope


def test_opt_in_envelope_removes_long_thread_without_filling_airframe_holes():
    mask = np.zeros((100, 120), np.uint8)
    mask[35:75, 25:100] = 255
    mask[50:55, 55:60] = 0
    mask[:35, 60] = 255
    before = mask.copy()
    config = dict(opening_kernel=7, padding_px=2, max_removed_alpha_fraction=.08)
    result = reviewed_alpha_cleanup(mask, [], [[30, 40, 50, 70]], envelope=config)
    assert result[2, 60] == 0
    assert np.array_equal(result[35:75, 25:100], before[35:75, 25:100])
    assert not result[50:55, 55:60].any()
    assert (result <= before).all() and np.array_equal(mask, before)


def test_multiple_solid_objects_or_empty_core_require_rejection():
    mask = np.zeros((80, 100), np.uint8)
    mask[10:35, 5:30] = 255
    mask[45:70, 70:95] = 255
    with pytest.raises(ValueError, match='too much'):
        support_envelope(mask, dict(opening_kernel=7, padding_px=2, max_removed_alpha_fraction=.08))
    with pytest.raises(ValueError, match='No solid'):
        support_envelope(np.zeros_like(mask), dict(opening_kernel=7, padding_px=2, max_removed_alpha_fraction=.08))
    with pytest.raises(ValueError, match='configuration'):
        support_envelope(mask, dict(opening_kernel=4, padding_px=2, max_removed_alpha_fraction=.08))


def test_body_shaped_gate_removes_thread_in_concave_airframe_gap():
    mask = np.zeros((120, 120), np.uint8)
    mask[20:100, 50:70] = 255
    mask[20:35, 25:95] = 255
    mask[65:85, 10:110] = 255
    mask[35:65, 35] = 255
    result = support_envelope(mask, dict(mode='dilated_core', opening_kernel=7,
                                        padding_px=2, max_removed_alpha_fraction=.08))
    assert result[50, 35] == 0
    assert np.array_equal(result[40:90, 52:68], mask[40:90, 52:68])
    assert (result <= mask).all()
