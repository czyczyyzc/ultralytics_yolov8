from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.anti_uav.prepare_fixedwing_finetune import fit_monitor, is_tiny, select_replay, write_split
from scripts.anti_uav.run_fixedwing_delivery_finetune import finite_metrics, selection_score


def row(video, index, edge=6, positive=True):
    return dict(im_file=f"/images/{video}/{index:09d}.jpg", shape=(544, 960),
                cls=np.zeros((int(positive), 1)),
                bboxes=np.array([[.5, .5, edge/960, edge/544]]) if positive else np.empty((0, 4)))


def test_replay_keeps_new_frames_separate_and_balances_tiny_exposure():
    old = [row(f"old{i%4}", i, 6 if i < 100 else 100) for i in range(300)]
    new = [row("new", i, 100) for i in range(50)]
    replay = select_replay(old, new)
    assert len(replay) == 100
    assert sum(map(is_tiny, replay)) == 53
    assert Counter(x["im_file"] for x in replay) <= Counter(x["im_file"] for x in old)
    assert {Path(x["im_file"]).parent.name for x in replay} == {f"old{i}" for i in range(4)}
    assert [x["im_file"] for x in replay] == [x["im_file"] for x in select_replay(old, new)]


def test_replay_and_monitors_do_not_filter_large_boxes():
    large = row("large", 0, 900)
    assert not is_tiny(large)
    assert select_replay([large], [row("new", 0, 6)])[0] is large
    frames = [row("video", i, 900) for i in range(100)]+[row("negative", i, positive=False) for i in range(70)]
    selected = fit_monitor(frames)
    assert len(selected) == 96
    assert len({x["im_file"] for x in selected}) == 96
    assert {x["im_file"] for x in selected} <= {x["im_file"] for x in frames}
    with pytest.raises(ValueError):
        select_replay([row("old", 0, positive=False)], [large])


def test_split_preserves_duplicate_exposures_and_cache_order(tmp_path):
    rows = [row("b", 1), row("a", 0), row("a", 0)]
    listing = tmp_path/"train.txt"
    record = write_split(listing, rows, "1.0.3", "frozen-source")
    cache = np.load(listing.with_suffix(".cache"), allow_pickle=True).item()
    assert listing.read_text().splitlines() == [x["im_file"] for x in cache["labels"]]
    assert record["entries"] == 3
    assert cache["results"][-1] == 3


def test_selection_guards_old_recall_and_labels_missing_scale_metrics():
    stats = {"fitness": .7, "native/c0.03/R": .6, "native/c0.03/long_4to8px/R": .6,
             "adaptation_fit/c0.03/F2": .8, "adaptation_fit/mAP50": .8,
             "adaptation_fit/mAP50-95": .4}
    baseline = dict(stats)
    score, retained = selection_score(stats, baseline)
    assert retained and score == pytest.approx(.71)
    broken = dict(stats, **{"native/c0.03/R": .56})
    assert selection_score(broken, baseline) == pytest.approx((score-1, False))
    assert finite_metrics({"empty": float("nan"), "valid": .6}) == {"empty": None, "valid": .6}
