from __future__ import annotations

import numpy as np

from scripts.anti_uav.anti_uav_rk3588 import group_model_zoo_outputs
from scripts.anti_uav.export_detector_rkopt_onnx import validate_rkopt_shapes
from scripts.anti_uav.rknn_simulator_video_clip import postprocess
from scripts.anti_uav.train_real_gray_yolo_lovo_fold import p2_target_key


def rkopt_shapes(grids: list[tuple[int, int]]) -> list[list[int]]:
    shapes = []
    for height, width in grids:
        shapes.extend(([1, 64, height, width], [1, 1, height, width], [1, 1, height, width]))
    return shapes


def test_validate_rkopt_shapes_accepts_p2_to_p5_outputs():
    validation = validate_rkopt_shapes(rkopt_shapes([(136, 240), (68, 120), (34, 60), (17, 30)]))

    assert validation["output_count"] == 12
    assert validation["branch_count"] == 4
    assert validation["pair_per_branch"] == 3


def test_group_model_zoo_outputs_accepts_four_branches():
    outputs = [np.zeros(shape, dtype=np.float32) for shape in rkopt_shapes([(136, 240), (68, 120), (34, 60), (17, 30)])]

    branches = group_model_zoo_outputs(outputs)

    assert len(branches) == 4
    assert [branch[0].shape[2:] for branch in branches] == [(136, 240), (68, 120), (34, 60), (17, 30)]


def test_validate_rkopt_shapes_keeps_three_branch_support():
    validation = validate_rkopt_shapes(rkopt_shapes([(68, 120), (34, 60), (17, 30)]))

    assert validation["output_count"] == 9
    assert validation["branch_count"] == 3


def test_p2_target_key_reuses_standard_neck_and_detect_branches():
    assert p2_target_key("model.16.conv.weight") == "model.22.conv.weight"
    assert p2_target_key("model.21.cv2.conv.weight") == "model.27.cv2.conv.weight"
    assert p2_target_key("model.22.cv2.0.0.conv.weight") == "model.28.cv2.1.0.conv.weight"
    assert p2_target_key("model.22.cv3.2.2.bias") == "model.28.cv3.3.2.bias"
    assert p2_target_key("model.22.dfl.conv.weight") == "model.28.dfl.conv.weight"
    assert p2_target_key("model.15.cv1.conv.weight") is None


def test_simulator_postprocess_accepts_p2_outputs():
    outputs = [np.zeros(shape, dtype=np.float32) for shape in rkopt_shapes([(4, 6), (2, 3), (1, 2), (1, 1)])]

    boxes, scores = postprocess(outputs, conf=0.25, nms_iou=0.45, input_height=16, input_width=24)

    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
