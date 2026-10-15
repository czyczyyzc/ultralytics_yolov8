import json

import yaml

from scripts.anti_uav.run_p3_independent_curriculum_training import stage_yaml


def test_stage_yaml_rebalances_appended_scale_slots(tmp_path):
    donors = tmp_path / "donors.json"
    donors.write_text(json.dumps({"donors": [{"im_file": str(index)} for index in range(7)]}))
    source = {
        "train": "train.txt",
        "val": "val.txt",
        "names": {0: "drone"},
        "label_sampling": {"anchor_slots": 100},
        "online_replacement": {"cache": "cache", "epoch_offset": 3},
        "online_scale": {"donor_manifest": str(donors), "views_per_donor": 5},
    }

    native_path = stage_yaml(source, tmp_path / "native.yaml", 10, scale_views=0)
    native = yaml.safe_load(native_path.read_text())
    assert "online_scale" not in native
    assert native["label_sampling"]["anchor_slots"] == 65
    assert native["online_replacement"]["epoch_offset"] == 13

    balanced_path = stage_yaml(
        source, tmp_path / "balanced.yaml", 20, scale_views=2, keep_replacement=False
    )
    balanced = yaml.safe_load(balanced_path.read_text())
    assert balanced["online_scale"]["views_per_donor"] == 2
    assert balanced["label_sampling"]["anchor_slots"] == 79
    assert "online_replacement" not in balanced
