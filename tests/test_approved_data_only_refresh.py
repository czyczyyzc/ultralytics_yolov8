from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.run_approved_data_only_refresh import (
    choose_idle_gpu, recipe_difference, select_approved_additions, verify_frozen_scope,
)


def task(digest="new", status="approved", revision=3, approved_revision=3):
    return dict(task_id=digest, source_sha256=digest, status=status, revision=revision,
                approved_revision=approved_revision, annotations=[dict(frame_index=1, status="confirmed")])


BASE = dict(train_video_hashes=["old"], test_sha256="test", validation_sha256="val")


def test_only_approved_nonheldout_new_sources():
    rows = [task("old"), task("test"), task("val"), task("pending", "submitted"),
            task("unfinished", "in_progress"), task()]
    assert select_approved_additions(rows, BASE, 1) == [task()]


def test_current_revision_must_be_reviewed():
    with pytest.raises(ValueError, match="unapproved current revision"):
        select_approved_additions([task(revision=4)], BASE, 1)


def test_duplicate_sha_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        select_approved_additions([task(), task()], BASE, 2)


def test_count_cannot_silently_expand():
    with pytest.raises(ValueError, match="Expected 15"):
        select_approved_additions([task()], BASE)


def test_annotation_change_during_preparation_is_rejected():
    before = [task()]
    after = deepcopy(before)
    after[0]["annotations"][0]["status"] = "absent"
    with pytest.raises(ValueError, match="annotations changed"):
        verify_frozen_scope(before, after)


def test_revision_change_during_preparation_is_rejected():
    with pytest.raises(ValueError, match="annotations changed"):
        verify_frozen_scope([task()], [task(revision=4, approved_revision=4)])


def test_identical_snapshot_is_accepted():
    verify_frozen_scope([task()], [task()])


def test_recipe_allows_data_and_count_changes_only():
    old = dict(method="same", seed=1, epochs=dict(foundation=30), data="old", training_videos=52)
    new = dict(old, data="new", training_videos=67)
    assert recipe_difference(old, new) == {}
    new["seed"] = 2
    assert set(recipe_difference(old, new)) == {"seed"}


def test_augmentation_change_is_not_data_only():
    old = dict(augmentation=dict(probability=.5, scale=.2))
    new = dict(augmentation=dict(probability=.5, scale=.3))
    assert set(recipe_difference(old, new)) == {"augmentation"}


def test_idle_gpu_selection_never_shares_memory():
    text = "0, 0, 0\n2, 1200, 0\n6, 0, 0\n"
    assert choose_idle_gpu(text, [2, 6]) == 6
    assert choose_idle_gpu("2, 0, 90\n6, 500, 0\n", [2, 6]) is None
