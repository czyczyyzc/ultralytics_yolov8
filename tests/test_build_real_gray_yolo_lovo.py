from pathlib import Path

import pytest

from scripts.anti_uav.build_real_gray_yolo_lovo import select_negatives


def test_select_negatives_prioritizes_hard_examples_and_hits_fraction() -> None:
    candidates = [Path(f"negative_{index}.jpg") for index in range(40)]
    hard = [candidates[7], candidates[3], Path("not_in_pool.jpg")]

    selected, hard_count = select_negatives(candidates, hard, 80, 0.20, seed=123)

    assert len(selected) == 20
    assert len(set(selected)) == 20
    assert selected[:2] == hard[:2]
    assert hard_count == 2


def test_select_negatives_is_deterministic() -> None:
    candidates = [Path(f"negative_{index}.jpg") for index in range(20)]

    first, _ = select_negatives(candidates, [], 80, 0.10, seed=456)
    second, _ = select_negatives(candidates, [], 80, 0.10, seed=456)

    assert first == second
    assert len(first) == 9


def test_select_negatives_rejects_unavailable_quota() -> None:
    with pytest.raises(RuntimeError, match="only 2 are available"):
        select_negatives([Path("a.jpg"), Path("b.jpg")], [], 80, 0.20, seed=1)
