import numpy as np

from scripts.anti_uav.audit_labeler_expansion_regression import (
    AuditDataset, ShuffledCategoryProbe, label_equal, size_category, strip_locations,
)
from scripts.anti_uav.label_pool_sampling import TinyAwareSampler


def row(path, edge=None):
    boxes = [] if edge is None else [[.5, .5, edge / 960, edge / 544]]
    return dict(im_file=path, shape=(544, 960), normalized=True, bbox_format="xywh",
                cls=np.zeros((len(boxes), 1), dtype=np.float32),
                bboxes=np.array(boxes, dtype=np.float32).reshape(-1, 4))


def test_size_categories_match_sampler_boundaries():
    labels = [row('/a/0', 4), row('/a/1', 8), row('/a/2', 16), row('/a/3', 32), row('/a/4')]
    assert [size_category(x) for x in labels] == ['4to8', '4to8', '8to16', 'gt16', 'negative']
    sampler = TinyAwareSampler(AuditDataset(labels), batch_size=4)
    assert sampler.categories.tolist() == [0, 0, 1, 1, 2]


def test_native_exposure_preserved_but_tiny_fraction_can_drop():
    old = [row(f'/tiny/{i}', 6) for i in range(10)]
    old += [row(f'/normal/{i}', 32) for i in range(10)]
    old += [row(f'/negative/{i}') for i in range(5)]
    new = old + [row(f'/new/{i}', 32) for i in range(10)]
    def sample(labels):
        sampler = TinyAwareSampler(AuditDataset(labels), batch_size=8)
        indices = list(sampler)
        assert sorted(indices) == list(range(len(labels)))
        return sum(sampler.categories[i] == 0 for i in indices), len(indices)
    a, b = sample(old), sample(new)
    assert a[0] == b[0] == 10
    assert b[0] / b[1] < a[0] / a[1]


def test_recipe_comparison_ignores_locations_not_hyperparameters():
    assert strip_locations(dict(device='0', model='/tmp/a', data='/tmp/b', lr0=.001, epochs=30)) == dict(lr0=.001, epochs=30)


def test_changed_existing_bbox_is_not_hidden_by_path_identity():
    a = row('/image.jpg', 6)
    b = row('/image.jpg', 7)
    assert label_equal(a, row('/image.jpg', 6))
    assert not label_equal(a, b)


def test_ordering_probe_preserves_the_exact_epoch_multiset():
    labels = [row(f'/rgb/{i}', 32) for i in range(150)]
    labels += [row(f'/gray/{i}', 32) for i in range(20)]
    labels += [row(f'/tiny/{i}', 6) for i in range(70)]
    labels += [row(f'/negative/{i}') for i in range(40)]
    data = AuditDataset(labels)
    original = TinyAwareSampler(data, batch_size=16, seed=7)
    probe = ShuffledCategoryProbe(data, batch_size=16, seed=7)
    original.set_epoch(3)
    probe.set_epoch(3)
    a, b = list(original), list(probe)
    assert sorted(a) == sorted(b) == list(range(len(labels)))
    for start in range(0, len(labels), 16):
        assert sorted(original.categories[a[start:start+16]]) == sorted(probe.categories[b[start:start+16]])
