from copy import deepcopy

import pytest

from scripts.anti_uav.run_gray_probability_ablation import assert_probability_only, probability_config


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
