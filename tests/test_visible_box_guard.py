import cv2
import numpy as np
import pytest

from ultralytics.data.augment import RandomPerspective
from ultralytics.utils.instance import Instances


def sample(boxes):
    boxes = np.array(boxes, np.float32).reshape(-1, 4)
    image = np.full((32, 48, 3), 80, np.uint8)
    image[8, 8] = 200
    return dict(img=image, cls=np.zeros((len(boxes), 1), np.float32),
                instances=Instances(boxes, segments=np.empty((0, 0, 2), np.float32),
                                    bbox_format="xyxy", normalized=False))


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


def apply_subpixel(boxes, matrix, scale=1.0):
    transform = RandomPerspective(visible_box_guard="affine_keep_subpixel_v2")
    matrix = np.array(matrix, np.float32)
    labels = sample(boxes)
    warped = cv2.warpAffine(labels["img"], matrix[:2], labels["img"].shape[1::-1])
    transform.affine_transform = lambda img, border: (warped, matrix, scale)
    return transform(labels), warped


def test_visible_thin_target_keeps_actual_scaled_pixels_and_fractional_box():
    matrix = [[.8, 0, 1], [0, .8, 3], [0, 0, 1]]
    result, warped = apply_subpixel([[8, 8, 14, 10]], matrix, .8)
    assert result["visible_box_guard"] == (1, 1, 0, 1, 0)
    np.testing.assert_allclose(result["instances"].bboxes, [[7.4, 9.4, 12.2, 11]], atol=1e-5)
    np.testing.assert_array_equal(result["img"], warped)
    assert np.max(warped[9:11, 7:13]) > 80


def test_partial_subpixel_strip_retained_and_fully_offscreen_box_removed():
    result, warped = apply_subpixel([[3, 8, 10, 12], [1, 1, 7, 5]], [[1, 0, -9.5], [0, 1, 0], [0, 0, 1]])
    assert result["visible_box_guard"] == (2, 1, 1, 1, 0)
    np.testing.assert_allclose(result["instances"].bboxes, [[0, 8, .5, 12]])
    np.testing.assert_array_equal(result["img"], warped)


def test_degenerate_projection_does_not_create_false_background():
    boxes = [[8, 8, 9, 9]]
    result, _ = apply_subpixel(boxes, [[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    assert result["visible_box_guard"] == (1, 1, 0, 0, 1)
    np.testing.assert_array_equal(result["instances"].bboxes, boxes)
    np.testing.assert_array_equal(result["img"], sample(boxes)["img"])


def test_subpixel_guard_is_restricted_to_exact_axis_aligned_geometry():
    with pytest.raises(ValueError, match="scaling and translation"):
        RandomPerspective(degrees=1, visible_box_guard="affine_keep_subpixel_v2")
