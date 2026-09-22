from copy import deepcopy

import pytest

from scripts.anti_uav.run_gray_probability_ablation import assert_probability_only, assert_split_isolation, probability_config


def config():
    return dict(train="unchanged_train", val="unchanged_val", names={0: "drone"},
                label_sampling=dict(negatives_per_epoch=12, negative_pool="neg"),
                online_replacement=dict(cache="immutable_cache", seed=18, replacement_probability=.5))


@pytest.mark.parametrize("probability", [0, .15, .5, 1])
def test_probability_is_only_change(probability):
    source = config()
    before = deepcopy(source)
    result = probability_config(source, probability)
    assert_probability_only(source, result)
    assert source == before
    assert ("online_replacement" not in result) == (probability == 0)


@pytest.mark.parametrize("probability", [-1, 1.1, float("nan")])
def test_invalid_probability_rejected(probability):
    with pytest.raises(ValueError):
        probability_config(config(), probability)


@pytest.mark.parametrize("field", ["train", "val", "label_sampling", "names"])
def test_other_changes_rejected(field):
    source = config()
    result = probability_config(source, .15)
    result[field] = "changed"
    with pytest.raises(ValueError, match="beyond replacement"):
        assert_probability_only(source, result)


def test_changed_cache_rejected():
    source = config()
    result = probability_config(source, .15)
    result["online_replacement"]["cache"] = "other"
    with pytest.raises(ValueError, match="beyond probability"):
        assert_probability_only(source, result)


def test_split_guard_does_not_misclassify_holdout_protocol_directory():
    assert_split_isolation(["/data/strict_holdout_Video00004/images/Video00005/1.jpg"],
                           ["/data/gray_val/Video00009/1.jpg"])


@pytest.mark.parametrize("train,val", [
    (["/data/Video00004/1.jpg"], ["/data/gray_val/1.jpg"]),
    (["/data/video00004/1.jpg"], ["/data/gray_val/1.jpg"]),
    (["/data/holdout_Video00004/images/1.jpg"], ["/data/gray_val/1.jpg"]),
    (["same"], ["same"]),
    ([], ["validation"]),
])
def test_split_guard_rejects_actual_holdout_and_overlap(train, val):
    with pytest.raises(ValueError):
        assert_split_isolation(train, val)
