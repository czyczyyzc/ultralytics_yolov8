from __future__ import annotations

import numpy as np

from scripts.anti_uav.anti_uav_rk3588 import group_model_zoo_outputs
from scripts.anti_uav.export_detector_rkopt_onnx import validate_rkopt_shapes


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
