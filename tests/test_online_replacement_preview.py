import pytest

from scripts.anti_uav.preview_online_gray_replacements import random_schedule


def test_random_unique_sources_and_full_cutout_coverage():
    a, picks = random_schedule(list(range(500)), 53, 100, 123)
    assert len(a) == len(set(a)) == 500
    assert len(picks) == 100
    assert set(picks[:53]) == set(range(53))
    assert set(picks) == set(range(53))
    assert max(picks.count(i) for i in range(53)) == 2
    assert (a, picks) == random_schedule(list(range(500)), 53, 100, 123)
    assert a != random_schedule(list(range(500)), 53, 100, 124)[0]


@pytest.mark.parametrize("size,assets,count", [(10, 53, 11), (100, 0, 10), (100, 53, 0)])
def test_invalid_preview_request(size, assets, count):
    with pytest.raises(ValueError):
        random_schedule(list(range(size)), assets, count, 0)
