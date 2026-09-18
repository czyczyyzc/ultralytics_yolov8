from scripts.anti_uav.append_approved_gray_native import expansion_sampling
import pytest


def test_expansion_without_existing_pool():
    result = expansion_sampling(dict(append_only_positive=8, negative=2), {},
                                [str(i) for i in range(10)], 8, ["a", "b", "c"], .2)
    assert result == (["a", "b", "c"], 16, 4, 2, 18)


def test_expansion_preserves_existing_pool_instead_of_freezing_it(tmp_path):
    pool = tmp_path / "negative.txt"
    pool.write_text("n1\nn2\nn3\nn4\n")
    base = ["p"]*8 + ["fixed"]*2 + ["n1", "n2", "n3", "n4"]
    config = dict(label_sampling=dict(negative_pool=str(pool), negative_pool_count=4, anchor_slots=10))
    result = expansion_sampling(dict(append_only_positive=8, negative=6), config,
                                base, 8, ["new1", "new2"], .2)
    assert result == (["n1", "n2", "n3", "n4", "new1", "new2"], 16, 4, 2, 18)
    with pytest.raises(ValueError, match="Overlapping"):
        expansion_sampling(dict(append_only_positive=8, negative=6), config, base, 8, ["n1"], .2)
