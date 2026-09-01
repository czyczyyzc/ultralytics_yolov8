from pathlib import Path

from scripts.anti_uav.build_pseudo_labeled_gray_video_split import (
    Detection,
    bridge_short_gaps,
    replace_duplicate_positive_slots,
    select_supported_detections,
)


def test_select_supported_detections_rejects_unrelated_weak_box():
    candidates = [
        Detection(10, (0.10, 0.10, 0.12, 0.12), 0.80),
        Detection(11, (0.11, 0.10, 0.13, 0.12), 0.10),
        Detection(12, (0.70, 0.70, 0.72, 0.72), 0.10),
    ]

    selected = select_supported_detections(candidates, 0.25, 5, 0.08)

    assert [detection.frame for detection in selected] == [10, 11]


def test_bridge_short_gaps_interpolates_compatible_track():
    detections = [
        Detection(0, (0.10, 0.10, 0.20, 0.20), 0.8),
        Detection(3, (0.13, 0.10, 0.23, 0.20), 0.7),
    ]

    bridged = bridge_short_gaps(detections, max_gap=3, max_center_distance=0.08)

    assert [detection.frame for detection in bridged] == [0, 1, 2, 3]
    assert bridged[1].source == "interpolated"
    assert abs(bridged[1].box[0] - 0.11) < 1e-12


def test_replace_duplicate_positive_slots_preserves_negatives(tmp_path: Path):
    image_root = tmp_path / "images" / "train"
    label_root = tmp_path / "labels" / "train"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    paths = [image_root / f"{name}.jpg" for name in ("a", "b", "n", "p")]
    for path in paths:
        path.write_bytes(b"image")
    for path in paths:
        (label_root / f"{path.stem}.txt").write_text("" if path.stem == "n" else "0 0.5 0.5 0.1 0.1\n")
    source = [paths[0], paths[1], paths[2], paths[0], paths[1], paths[0]]

    output, manifest = replace_duplicate_positive_slots(source, [paths[3]], 0.4, seed=7)

    assert len(output) == len(source)
    assert [path for path in output if path == paths[2]] == [paths[2]]
    assert paths[0] in output and paths[1] in output
    assert paths[3] in output
    assert manifest["negative_samples_unchanged"] is True
