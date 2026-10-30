from collections import Counter

import numpy as np
import pytest

from scripts.anti_uav.audit_labeler_expansion_regression import AuditDataset
from scripts.anti_uav.label_pool_sampling import (
    BalancedTinyAwareSampler, TinyAwareSampler, balanced_video_order, make_exposure_sampler,
    source_video_identity,
)
from scripts.anti_uav.prepare_labeler_training_expansion import restrict_task_scope


def row(path, edge=None):
    boxes = [] if edge is None else [[.5, .5, edge/960, edge/544]]
    return dict(im_file=path, shape=(544, 960), normalized=True, bbox_format="xywh",
                cls=np.zeros((len(boxes), 1), np.float32),
                bboxes=np.array(boxes, np.float32).reshape(-1, 4))


def dataset(tmp_path, tiny=200, ordinary=700, negatives=300):
    rows = [row(f'/gray/tiny/{i}.jpg', 6) for i in range(tiny)]
    rows += [row(f'/gray/large/{i}.jpg', 400) for i in range(ordinary)]
    rows += [row(f'/negative/{i}.jpg') for i in range(negatives)]
    pool = tmp_path/'negative.txt'
    pool.write_text(''.join(x['im_file']+'\n' for x in rows[-negatives:]))
    config = dict(negative_pool=str(pool), negative_pool_count=negatives,
                  negatives_per_epoch=round((tiny+ordinary)*.15/.85),
                  anchor_slots=tiny+ordinary, target_negative_fraction=.15)
    return AuditDataset(rows), config


def test_flat_rgb_sources_are_distinct():
    a = source_video_identity('/data/anti_uav300_yolo/images/rgb/test-dev_sceneA_rgb_001.jpg')
    b = source_video_identity('/data/anti_uav300_yolo/images/rgb/test-dev_sceneB_rgb_001.jpg')
    assert a != b
    assert a == source_video_identity('/data/anti_uav300_yolo/images/rgb/test-dev_sceneA_rgb_002.jpg')
    with pytest.raises(ValueError, match='Cannot identify'):
        source_video_identity('/data/anti_uav300_yolo/images/rgb/unknown.jpg')


def test_weighted_order_keeps_short_videos_in_the_tail():
    keys = ['large']*1000+['short']*100
    order = balanced_video_order(list(range(1100)), keys, np.random.default_rng(5))
    assert sorted(order) == list(range(1100))
    assert 8 <= sum(keys[i] == 'short' for i in order[-110:]) <= 12


def test_quota_preserves_all_native_positives_and_bounds_repeats(tmp_path):
    data, config = dataset(tmp_path)
    sampler = BalancedTinyAwareSampler(data, config, dict(target_tiny_fraction=.30), 64, 5)
    order = list(sampler)
    counts = Counter(order)
    assert all(counts[i] >= 1 for i in range(900))
    assert all(counts[i] == 1 for i in range(200, 900))
    assert max(counts[i] for i in range(200)) <= 2
    assert sum(sampler.categories[i] == 0 for i in order)/len(order) == pytest.approx(.30, abs=.002)
    assert sum(sampler.categories[i] == 2 for i in order)/len(order) == pytest.approx(.15, abs=.002)
    for start in range(0, len(order)-64, 64):
        tiny = sum(sampler.categories[i] == 0 for i in order[start:start+64])
        assert 18 <= tiny <= 20


def test_repeat_bound_uses_unique_images_not_duplicate_schedule_slots(tmp_path):
    data, config = dataset(tmp_path, tiny=1)
    with pytest.raises(ValueError, match='repeat safety bound'):
        BalancedTinyAwareSampler(data, config, dict(target_tiny_fraction=.30))


def test_policy_is_opt_in_and_epochs_are_deterministic(tmp_path):
    data, config = dataset(tmp_path)
    legacy = make_exposure_sampler(data, config, dict(tiny_min_px=4, tiny_max_px=8))
    assert type(legacy) is TinyAwareSampler
    fixed = make_exposure_sampler(data, config, dict(policy='balanced_v2'))
    fixed.set_epoch(3)
    a = list(fixed)
    assert a == list(fixed)
    fixed.set_epoch(4)
    assert a != list(fixed)


def test_refresh_rejects_removed_or_changed_task_and_ignores_new_tasks():
    old = dict(videos=[dict(task_id='a', source_sha256='shaA')])
    a = dict(task_id='a', source_sha256='shaA', revision=2)
    assert restrict_task_scope([a, dict(task_id='b', source_sha256='shaB')], old) == [a]
    with pytest.raises(ValueError, match='disappeared'):
        restrict_task_scope([], old)
    with pytest.raises(ValueError, match='source video changed'):
        restrict_task_scope([dict(a, source_sha256='different')], old)
