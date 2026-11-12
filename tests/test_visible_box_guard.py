import numpy as np
import pytest

from ultralytics.data.augment import RandomPerspective
from ultralytics.utils.instance import Instances


def sample(boxes):
    boxes = np.array(boxes, np.float32).reshape(-1, 4)
    image = np.full((32, 48, 3), 80, np.uint8)
    image[8, 8] = 200
    return dict(img=image, cls=np.zeros((len(boxes), 1), np.float32),
                instances=Instances(boxes, bbox_format="xyxy", normalized=False))


def apply(boxes, matrix, guard, scale=1.0):
    transform = RandomPerspective(preserve_visible_boxes=guard)
    transform.affine_transform = lambda img, border: (np.zeros_like(img), np.array(matrix, np.float32), scale)
    return transform(sample(boxes))


@pytest.mark.parametrize("boxes,matrix,scale", [
    ([[8, 8, 14, 10]], np.diag([.8, .8, 1]), .8),
    ([[2, 8, 9, 14]], [[1, 0, -8], [0, 1, 0], [0, 0, 1]], 1),
    ([[8, 8, 9, 9]], np.eye(3), 1),
])
def test_visible_boxes_rejected_by_legacy_are_rescued_with_matching_pixels(boxes, matrix, scale):
    assert len(apply(boxes, matrix, False, scale)["cls"]) == 0
    result = apply(boxes, matrix, True, scale)
    assert result["visible_box_guard"] == (1, 1, 0, 1)
    np.testing.assert_array_equal(result["instances"].bboxes, boxes)
    np.testing.assert_array_equal(result["img"], sample(boxes)["img"])


def test_genuinely_outside_target_and_reviewed_negative_remain_empty():
    shift = [[1, 0, -22], [0, 1, 0], [0, 0, 1]]
    assert apply([[8, 8, 13, 13]], shift, True)["visible_box_guard"] == (1, 0, 1, 0)
    assert apply([], np.eye(3), True)["visible_box_guard"] == (0, 0, 0, 0)


def test_valid_clipped_box_keeps_transformed_pixels_and_exact_clipped_coordinates():
    shift = [[1, 0, -4], [0, 1, 0], [0, 0, 1]]
    result = apply([[2, 8, 12, 14]], shift, True)
    assert result["visible_box_guard"] == (1, 1, 0, 0)
    np.testing.assert_array_equal(result["instances"].bboxes, [[0, 8, 8, 14]])
    assert not result["img"].any()
