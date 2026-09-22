from copy import deepcopy
from collections import Counter
from pathlib import Path

import pytest

from scripts.anti_uav.build_gray_pair_synthetic_test import (
    EvaluationTemporalBackgrounds, assign_assets, label_for,
)


def test_eval_registry_requires_explicit_non_training_schema():
    for plan in ({}, {"schema": "online_gray_background_cache.v1", "training_allowed": False},
                 {"schema": "evaluation_only_gray_replacement.v1", "training_allowed": True}):
        with pytest.raises(ValueError):
            EvaluationTemporalBackgrounds(plan)


def test_asset_schedule_deterministic_balanced_and_leaves_negatives_alone():
    rows = [dict(frame=i, box=[1, 2, 3, 4] if i < 17 else [], asset_id=None) for i in range(20)]
    first = assign_assets(deepcopy(rows), ["1", "2", "3"], 123)
    assert first == assign_assets(deepcopy(rows), ["1", "2", "3"], 123)
    assert first != assign_assets(deepcopy(rows), ["1", "2", "3"], 456)
    assert all(r["asset_id"] is None for r in first[17:])
    counts = Counter(r["asset_id"] for r in first[:17])
    assert max(counts.values())-min(counts.values()) == 1
    assert len(first) == len(rows)
    with pytest.raises(ValueError):
        assign_assets(rows, [], 123)
    with pytest.raises(ValueError):
        assign_assets(rows, ['1', '1'], 123)


def test_label_resolution_and_portable_unchanged_symlink(tmp_path):
    assert label_for('/data/images/gray/Video00004/1.jpg') == Path('/data/labels/gray/Video00004/1.txt')
    with pytest.raises(ValueError):
        label_for('/data/images/images/1.jpg')
    original = tmp_path/'original/images/Video00004/1.jpg'
    result = tmp_path/'synthetic/images/Video00004/1.jpg'
    original.parent.mkdir(parents=True)
    result.parent.mkdir(parents=True)
    original.write_bytes(b'unchanged image bytes')
    result.symlink_to('../../../original/images/Video00004/1.jpg')
    assert result.read_bytes() == original.read_bytes()
