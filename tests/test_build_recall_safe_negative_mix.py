from pathlib import Path

import pytest

from scripts.anti_uav.build_recall_safe_negative_mix import (
    filter_gray_negatives,
    parse_gray_frame,
    select_stratified_negatives,
)


def gray(sequence: str, frame: int) -> Path:
    return Path(f"/dataset/images/gray/{sequence}/{frame:06d}.jpg")


def test_parse_gray_frame() -> None:
    assert parse_gray_frame(gray("Video00004", 123)) == ("Video00004", 123)
    assert parse_gray_frame(Path("/dataset/images/rgb/frame.jpg")) is None


def test_filter_excludes_guard_and_thins_regular_negatives() -> None:
    sequence = "Video00001"
    candidates = [gray(sequence, frame) for frame in range(20) if frame != 10]
    hard = [gray(sequence, 7)]
    filtered, safe_hard, counts = filter_gray_negatives(
        candidates,
        hard,
        {sequence: [frame == 10 for frame in range(20)]},
        {sequence: 10.0},
        guard_seconds=0.2,
        temporal_sample_seconds=0.3,
        seed=7,
    )

    assert all(abs(parse_gray_frame(path)[1] - 10) > 2 for path in filtered)
    assert hard[0] in filtered
    assert safe_hard == hard
    assert counts["guard_excluded"] == 4
    assert counts["temporal_thinning_excluded"] > 0


def test_filter_rejects_positive_in_negative_pool() -> None:
    with pytest.raises(ValueError, match="positive frame"):
        filter_gray_negatives(
            [gray("Video00001", 1)],
            [],
            {"Video00001": [False, True]},
            {"Video00001": 10.0},
            guard_seconds=0.0,
            temporal_sample_seconds=0.0,
            seed=1,
        )


def test_stratified_selection_is_exact_deterministic_and_caps_hard_negatives() -> None:
    candidates = [
        *(gray("Video00001", frame) for frame in range(20)),
        *(gray("Video00002", frame) for frame in range(20)),
        *(Path(f"/dataset/images/rgb/{frame:06d}.jpg") for frame in range(20)),
    ]
    hard = candidates[:10]
    first, hard_count, groups = select_stratified_negatives(
        candidates,
        hard,
        positive_count=80,
        negative_fraction=0.20,
        hard_negative_max_fraction=0.25,
        seed=11,
    )
    second, _, _ = select_stratified_negatives(
        candidates,
        hard,
        positive_count=80,
        negative_fraction=0.20,
        hard_negative_max_fraction=0.25,
        seed=11,
    )

    assert len(first) == 20
    assert len(set(first)) == 20
    assert hard_count == 5
    assert first == second
    assert set(groups) == {"Video00001", "Video00002", "rgb"}
