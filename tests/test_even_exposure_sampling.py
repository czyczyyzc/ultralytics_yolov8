from collections import Counter

import numpy as np
import pytest

from scripts.anti_uav.label_pool_sampling import (
    EvenExposureTinyAwareSampler, TinyAwareSampler, make_exposure_sampler,
)


class Dataset:
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)


def fixture(tmp_path):
    labels = []
    for category, edge in [("tiny", 6), ("ordinary", 90), ("negative", None)]:
        for group, count in [("gray_a", 100), ("gray_b", 100), ("flat_public", 1600)]:
            for i in range(count):
                boxes = [] if edge is None else [[.5, .5, edge/960, edge/544]]
                labels.append(dict(im_file=f"/{category}/{group}/{i}.png", shape=(544, 960),
                    cls=np.zeros((len(boxes), 1), np.float32),
                    bboxes=np.asarray(boxes, np.float32).reshape(-1, 4),
                    bbox_format="xywh", normalized=True))
    pool = tmp_path/"negative.txt"
    pool.write_text("\n".join(r["im_file"] for r in labels[-1800:])+"\n")
    config = dict(negative_pool=str(pool), negative_pool_count=1800,
                  negatives_per_epoch=636, anchor_slots=3600, target_negative_fraction=.15)
    return Dataset(labels), config


@pytest.mark.parametrize("epoch", [0, 1, 29])
def test_reorder_preserves_every_index_and_negative_slice(tmp_path, epoch):
    data, label_config = fixture(tmp_path)
    old = TinyAwareSampler(data, label_config, {}, 64, 20261015)
    new = EvenExposureTinyAwareSampler(data, label_config, {}, 64, 20261015)
    old.set_epoch(epoch); new.set_epoch(epoch)
    a, b = list(old), list(new)
    assert Counter(a) == Counter(b)
    assert a != b
    assert len(a) == len(b) == 4236
    assert b == list(new)
    for start in range(0, len(a), 64):
        assert Counter(old.categories[i] for i in a[start:start+64]) == Counter(new.categories[i] for i in b[start:start+64])


def test_short_positive_sources_do_not_disappear_from_tail(tmp_path):
    data, label_config = fixture(tmp_path)
    new = EvenExposureTinyAwareSampler(data, label_config, {}, 64, 20261015)
    old = TinyAwareSampler(data, label_config, {}, 64, 20261015)
    tail = lambda sampler: [i for i in list(sampler)[-424:] if len(data.labels[i]["cls"])]
    gray = lambda indices: sum("/gray_" in data.labels[i]["im_file"] for i in indices)/len(indices)
    assert gray(tail(old)) == 0
    assert gray(tail(new)) == pytest.approx(2/18, abs=.02)


def test_policy_is_explicit_and_does_not_change_legacy_or_labels(tmp_path):
    data, label_config = fixture(tmp_path)
    before = [r["bboxes"].copy() for r in data.labels]
    assert type(make_exposure_sampler(data, label_config, {})) is not EvenExposureTinyAwareSampler
    assert type(make_exposure_sampler(data, label_config, dict(policy="even_exposure_v1"))) is EvenExposureTinyAwareSampler
    with pytest.raises(ValueError, match="Unknown"):
        make_exposure_sampler(data, label_config, dict(policy="misspelled"))
    list(EvenExposureTinyAwareSampler(data, label_config, {}))
    assert all(np.array_equal(a, row["bboxes"]) for a, row in zip(before, data.labels))
