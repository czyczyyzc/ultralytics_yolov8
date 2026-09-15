import numpy as np
import pytest

from scripts.anti_uav.large_target_augmentation import context_zoom, yolo_rows
from scripts.anti_uav.build_rebalanced_gray_split import append_unseen


def test_whole_frame_box_is_preserved():
    pytest.importorskip("torch")
    from ultralytics.solutions.anti_uav import AreaFilter, Detection
    frame = np.zeros((544, 960, 3), dtype=np.uint8)
    for box in ((0, 0, 960, 544), (1, 1, 900, 500), (20, 20, 21, 21), (0, 2, 960, 3)):
        assert AreaFilter().keep(Detection(box, .9), frame)
    for box in ((0, 0, 0, 3), (3, 4, 1, 1), (0, 0, np.nan, 3)):
        assert not AreaFilter().keep(Detection(box, .9), frame)


def test_no_maximum_area_constructor():
    pytest.importorskip("torch")
    from ultralytics.solutions.anti_uav import AreaFilter
    with pytest.raises(TypeError):
        AreaFilter(max_area_ratio=.25)


def test_tiny_donor_is_not_magnified():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert context_zoom(image, [[100, 100, 106, 104]], np.random.default_rng(1)) is None


def test_full_frame_zoom_and_labels():
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)
    result = context_zoom(image, [[300, 200, 900, 600]], np.random.default_rng(2),
                          area_range=(.99, 1.001), full_frame=True)
    assert result is not None
    output, boxes, meta = result
    assert output.shape == (544, 960, 3)
    np.testing.assert_allclose(boxes[0], [0, 0, 960, 544])
    assert meta["area_fraction"] == 1 and meta["partial"]
    assert meta["upscale"] <= 4 and meta["retained_fraction"] >= .5
    values = list(map(float, yolo_rows(boxes).split()))
    np.testing.assert_allclose(values, [0, .5, .5, 1, 1])


def test_zoom_is_deterministic_and_geometry_bounded():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    a = context_zoom(image, [[600, 300, 1100, 600]], np.random.default_rng(5), area_range=(.25, .4))
    b = context_zoom(image, [[600, 300, 1100, 600]], np.random.default_rng(5), area_range=(.25, .4))
    assert a is not None and b is not None
    np.testing.assert_array_equal(a[0], b[0])
    assert a[2] == b[2]
    assert .25 <= a[2]["area_fraction"] <= .4


def test_append_keeps_old_repeats_excludes_whole_val(tmp_path):
    for key in ("seen", "new", "validation"):
        d = tmp_path/key
        d.mkdir()
        (d/"000.jpg").touch()
    records = [dict(sha256=k, image_directory=str(tmp_path/k)) for k in ("seen", "new", "validation")]
    old = ["old.jpg", "old.jpg", "negative.jpg"]
    result = append_unseen(old, records, {"seen"}, {"validation"})
    assert result == old+[str(tmp_path/"new/000.jpg")]


def test_negative_mining_keeps_positive_and_video_counts():
    pytest.importorskip("torch")
    from collections import Counter
    from scripts.anti_uav.mine_rebalanced_negatives import replace_easy_negatives
    positives = ["/images/videoA/positive.jpg"]*5
    negatives = [f"/images/videoA/{i:06d}.jpg" for i in range(0, 200, 20)]
    scores = {p: dict(score=.9 if i == 0 else 0., sharpness=10.) for i, p in enumerate(negatives)}
    result, changes = replace_easy_negatives(positives+negatives, scores)
    assert len(changes) == 2
    assert result.count(positives[0]) == 5
    assert Counter(result)[negatives[0]] == 3
    assert len(result) == 15


def test_selection_fitness_prioritizes_fixed_gray_recall_precision():
    pytest.importorskip("torch")
    from scripts.anti_uav.gray_deployment_trainer import deployment_fitness
    assert deployment_fitness(1., 1., 1.) == 1.
    assert deployment_fitness(.8, .9, .5) > deployment_fitness(.6, .9, .5)


def test_scale_metrics_include_tiny_and_full_frame_without_double_matching():
    pytest.importorskip("torch")
    from scripts.anti_uav.gray_deployment_trainer import scale_masks, matched_ground_truth
    boxes = np.array([[0., 0., 8., 8.], [0., 0., 16., 12.], [0., 0., 1920., 1080.]])
    masks = scale_masks(boxes, (1080, 1920), (544, 960))
    np.testing.assert_array_equal(masks["long_4to8px"], [True, True, False])
    np.testing.assert_array_equal(masks["area_ge80pct"], [False, False, True])
    assert len(matched_ground_truth(np.array([[.9, .8], [.7, .1]]))) == 1
    assert len(matched_ground_truth(np.empty((0, 3)))) == 0
    assert len(matched_ground_truth(np.empty((3, 0)))) == 0
