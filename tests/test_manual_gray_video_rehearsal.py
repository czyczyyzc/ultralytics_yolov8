from pathlib import Path

from scripts.anti_uav.build_manual_gray_video_rehearsal import replace_duplicate_class_slots


def make_sample(root: Path, video: str, frame: int, positive: bool) -> Path:
    image = root / "images" / video / f"{frame:06d}.jpg"
    label = root / "labels" / video / f"{frame:06d}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    label.write_text("0 0.5 0.5 0.1 0.1\n" if positive else "")
    return image


def test_rehearsal_preserves_class_sequence_and_unique_source_coverage(tmp_path: Path):
    old_positive = make_sample(tmp_path / "old", "positive", 0, True)
    old_negative = make_sample(tmp_path / "old", "negative", 0, False)
    source = [old_positive] * 8 + [old_negative] * 8
    new_positive = {
        "clip0": [make_sample(tmp_path / "new", "clip0", 0, True)],
        "clip1": [make_sample(tmp_path / "new", "clip1", 0, True)],
    }
    new_negative = {
        "clip0": [make_sample(tmp_path / "new", "clip0", 1, False)],
        "clip1": [make_sample(tmp_path / "new", "clip1", 1, False)],
    }

    output, manifest = replace_duplicate_class_slots(
        source,
        new_positive,
        new_negative,
        positive_fraction=0.25,
        seed=7,
    )

    assert len(output) == 20
    assert old_positive in output
    assert old_negative in output
    assert set(path for paths in new_positive.values() for path in paths).issubset(output)
    assert set(path for paths in new_negative.values() for path in paths).issubset(output)
    assert [bool(label.read_text().strip()) for label in map(_label_path, output[:16])] == [True] * 8 + [False] * 8
    assert manifest["positive_rehearsal_slots"] == 2
    assert manifest["appended_positive_slots"] == 2
    assert manifest["appended_negative_slots"] == 2
    assert manifest["output_negative_fraction"] == 0.5


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")
