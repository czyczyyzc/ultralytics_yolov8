import json
from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from scripts.anti_uav.prepare_fixedwing_finetune import read_pinned_cache, write_split
from scripts.anti_uav.prepare_labeler_training_expansion import sha256
from scripts.anti_uav.run_fixedwing_finetune_delivery import (
    ROOT, completed_training, make_data, metric_table, release_note, render_handover,
)


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_no_partial_checkpoint_delivery(tmp_path):
    write_json(tmp_path/"status.json", dict(stage="finetuning_p3"))
    assert completed_training(tmp_path) is None
    write_json(tmp_path/"status.json", dict(stage="failed"))
    with pytest.raises(RuntimeError):
        completed_training(tmp_path)
    write_json(tmp_path/"status.json", dict(stage="complete"))
    weights = tmp_path/"training/p3"
    weights.mkdir(parents=True)
    (weights/"results.csv").write_text("epoch\n1\n")
    with pytest.raises(ValueError, match="partial"):
        completed_training(tmp_path)
    (weights/"results.csv").write_text("epoch\n"+"\n".join(map(str, range(1, 9))))
    candidate = tmp_path/"p3_student_fixedwing_best_fp32.pt"
    candidate.write_bytes(b"evaluated checkpoint")
    metrics = {key: .7 for key in ("native/c0.03/R", "native/c0.03/P",
                                  "adaptation_fit/c0.03/R", "adaptation_fit/c0.03/P")}
    value = dict(candidate=str(candidate), sha256=sha256(candidate), final=metrics)
    write_json(tmp_path/"comparison.json", value)
    assert completed_training(tmp_path) == value
    candidate.write_bytes(b"replaced checkpoint")
    with pytest.raises(ValueError, match="differs"):
        completed_training(tmp_path)


def row(path, positive=True):
    return dict(im_file=path, shape=(544, 960), cls=np.zeros((int(positive), 1)),
        bboxes=np.array([[.5, .5, .1, .1]]) if positive else np.empty((0, 4)),
        segments=[], keypoints=None, normalized=True, bbox_format="xywh")


def test_calibration_excludes_fit_monitor_without_deleting_fit_from_training(tmp_path):
    run = tmp_path/"run"
    data = run/"data"
    data.mkdir(parents=True)
    new = [row(f"/images/new/{i}.jpg", i != 2) for i in range(3)]
    old = row("/images/old/0.jpg")
    val = row("/images/Video00009/0.jpg")
    train_path, val_path = data/"train.txt", data/"val.txt"
    pins = {str(train_path): write_split(train_path, new+[old], "1.0.3", "snapshot"),
            str(val_path): write_split(val_path, [new[0], val], "1.0.3", "snapshot")}
    source = dict(train=str(train_path), val=str(val_path), names={0: "drone"}, trusted_label_caches=pins)
    (data/"train.yaml").write_text(yaml.safe_dump(source))
    write_json(data/"manifest.json", dict(adaptation_roots=["/images/new"], new_positive=2,
        new_reviewed_negative=1, source_snapshot_sha256="snapshot"))
    full, monitor, native, calibration = make_data(run, tmp_path/"prepared")
    def paths(config):
        y = yaml.safe_load(config.read_text())
        return set(Path(y["val"]).read_text().splitlines())
    assert paths(full) == {x["im_file"] for x in new}
    assert paths(monitor) == {new[0]["im_file"]}
    assert paths(native) == {val["im_file"]}
    y = yaml.safe_load(calibration.read_text())
    cache = read_pinned_cache(y["train"], y)
    assert len(cache["labels"]) == 3
    assert new[0]["im_file"] not in {x["im_file"] for x in cache["labels"]}
    assert new[0]["im_file"] in train_path.read_text()


def test_release_scope_and_complete_document():
    header = (ROOT/"scripts/anti_uav/templates/RK3576_FIXEDWING_FINETUNE_HANDOVER_ZH.md").read_text()
    shared = (ROOT/"scripts/anti_uav/templates/RK3576_LABELER_HANDOVER_ZH.md").read_text().split("## 5. 模型与图像协议")[1]
    values = {key: "verified-value" for key in re.findall(r"\{\{(\w+)\}\}", header+shared)}
    document = render_handover(values)
    assert "{{" not in document
    assert "公开 `yolov8n.pt` 起步" not in document
    assert "旧 52 视频 P3 Student" in document
    assert "--workers 2" in document and "--npu-masks 0,1" in document
    assert "不是板端精度" in document and "不计算 P2" not in document  # pure-P3 scope is described in section 1
    assert "无条件替代" in release_note(dict(ready_for_quantization_review=False))
    assert "通过本轮" in release_note(dict(ready_for_quantization_review=True))
    with pytest.raises(ValueError, match="Unresolved"):
        render_handover({})


def test_metric_table_recall_first():
    m = {"native/c0.03/R": .7, "native/c0.03/P": .8, "native/c0.03/TP": 7,
         "native/c0.03/FP": 2, "native/c0.03/FN": 3, "native/mAP50": .6, "native/mAP50-95": .3}
    table = metric_table({"fixedwing": m})
    assert table.index("Recall") < table.index("Precision")
    assert "70.00% | 80.00% | 7 | 2 | 3" in table
