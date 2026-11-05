"""Focused tests; run with --noconftest to avoid unrelated training hooks."""

import json
import hashlib
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.anti_uav.prepare_uavswarm_research import mot_box_to_yolo, prepare


def sequence(root, split, name, rows, frames=(1,), size=(100, 50), color="red"):
    path = root / split / name
    (path / "img1").mkdir(parents=True)
    (path / "gt").mkdir()
    (path / "gt" / "gt.txt").write_text(rows)
    (path / "det").mkdir()
    (path / "det" / "det.txt").write_text("1,-1,90,40,8,8,1,-1,-1,-1\n")
    for frame in frames:
        Image.new("RGB", size, color).save(path / "img1" / f"{frame:06d}.jpg")
    return path


def fixture(root, train_rows="1,1,1,1,1,1,1,1,1\n", train_frames=(1,)):
    first = sequence(root, "train", "UAVSwarm-01", train_rows, train_frames)
    second = sequence(root, "test", "UAVSwarm-02", "1,1,11,6,10,5,1,1,1\n", color="blue")
    return first, second


def test_one_based_xy_and_tiny_boxes():
    assert mot_box_to_yolo(1, 1, 1, 1, 100, 50, 1) == pytest.approx((.005, .01, .01, .02))
    assert mot_box_to_yolo(1, 1, .001, .002, 100, 50, 1) == pytest.approx((.000005, .00002, .00001, .00004))
    assert mot_box_to_yolo(100, 50, 1, 1, 100, 50, 1) == pytest.approx((.995, .99, .01, .02))


def test_default_preserves_official_raw_coordinates_for_four_pixel_boxes():
    assert mot_box_to_yolo(3, 4, 4, 4, 100, 50) == pytest.approx((.05, .12, .04, .08))
    assert mot_box_to_yolo(0, 0, 4, 4, 100, 50) == pytest.approx((.02, .04, .04, .08))


def test_clipping_keeps_positive_intersection():
    assert mot_box_to_yolo(-1, -1, 3, 3, 100, 50) == pytest.approx((.01, .02, .02, .04))
    assert mot_box_to_yolo(101, 1, 1, 1, 100, 50) is None
    assert mot_box_to_yolo(1, 1, 0, 2, 100, 50) is None


def test_contract_official_split_actual_dimensions_and_gt_only(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    train, _ = fixture(root)
    (train / "seqinfo.ini").write_text("[Sequence]\nimWidth=999\nimHeight=999\n")
    audit = prepare(root, output)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert audit["split_counts"] == {"train": 1, "test": 1}
    assert [row["group"] for row in rows] == ["UAVSwarm-01", "UAVSwarm-02"]
    assert all(row["dataset"] == "uavswarm" and row["boxes"] == 1 for row in rows)
    assert all(Path(row[key]).is_absolute() for row in rows for key in ("image", "label", "source"))
    assert all(Path(row["image"]).is_symlink() for row in rows)
    assert all(row["source"].endswith("/gt/gt.txt") for row in rows)
    assert (rows[0]["width"], rows[0]["height"]) == (100, 50)
    assert list(map(float, Path(rows[0]["label"]).read_text().split())) == pytest.approx([0, .015, .03, .01, .02])
    assert audit["xy_translation"] == 0
    assert audit["coordinate_visual_qa"].startswith("not performed")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] is True
    assert summary["manifest_sha256"] == hashlib.sha256((output / "manifest.jsonl").read_bytes()).hexdigest()
    assert summary["training_approved"] is False
    assert not (output / "val.txt").exists()
    notebook = json.loads((output / "audit.ipynb").read_text())
    assert notebook["nbformat"] == 4
    exec("".join(notebook["cells"][1]["source"]), {})


def test_any_inactive_box_excludes_whole_frame_and_unannotated_not_negative(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root, "1,1,1,1,5,5,1,1,1\n1,2,10,10,5,5,0,1,1\n"
                  "2,1,1,1,.001,.001,1,1,0\n", (1, 2, 3))
    audit = prepare(root, output)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert [row["frame"] for row in rows if row["split"] == "train"] == [2]
    assert audit["skipped"] == {"inactive_gt_frame": 1, "no_gt": 1}
    assert len(Path(rows[0]["label"]).read_text().splitlines()) == 1


def test_sparse_numeric_filenames_map_exactly_not_by_sort_index(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root, "7,1,1,1,2,2,1,1,1\n", (7,))
    prepare(root, output)
    row = json.loads((output / "manifest.jsonl").read_text().splitlines()[0])
    assert row["frame"] == 7 and Path(row["image"]).name == "000007.jpg"


@pytest.mark.parametrize("rows", ["0,1,1,1,2,2,1,1,1\n", "1.5,1,1,1,2,2,1,1,1\n",
                                  "1,-1,1,1,2,2,1,1,1\n", "1,1,nan,1,2,2,1,1,1\n",
                                  "1,1,1,1,2,2,1,1\n", "1,1,1,1,2,2,1,1,2\n",
                                  "1,1,1,1,2,2,1,1,1\n1,1,1,1,2,2,1,1,1\n"])
def test_bad_gt_fails_closed_without_output(tmp_path, rows):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root, rows)
    with pytest.raises(ValueError):
        prepare(root, output)
    assert not output.exists()


def test_gt_missing_image_cannot_infer_offset(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root, "2,1,1,1,2,2,1,1,1\n")
    with pytest.raises(ValueError, match="no matching image"):
        prepare(root, output)


def test_no_fallback_to_detector_predictions(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    train, _ = fixture(root)
    (train / "gt" / "gt.txt").unlink()
    with pytest.raises(ValueError, match="Missing human GT"):
        prepare(root, output)


def test_cross_split_group_rejected(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root)
    (root / "test" / "UAVSwarm-02").rename(root / "test" / "UAVSwarm-01")
    with pytest.raises(ValueError, match="cross-split group"):
        prepare(root, output)


def test_cross_split_exact_image_rejected(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    train, test = fixture(root)
    (test / "img1" / "000001.jpg").write_bytes((train / "img1" / "000001.jpg").read_bytes())
    with pytest.raises(ValueError, match="Cross-split duplicate image"):
        prepare(root, output)


def test_duplicate_numeric_frame_rejected(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    train, _ = fixture(root)
    Image.new("RGB", (100, 50)).save(train / "img1" / "1.png")
    with pytest.raises(ValueError, match="Duplicate image frame"):
        prepare(root, output)


def test_invalid_boxes_never_create_empty_negative_label(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root, "1,1,1,1,0,1,1,1,1\n")
    audit = prepare(root, output)
    assert audit["split_counts"] == {"test": 1}
    assert audit["skipped"] == {"no_valid_positive": 1}


def test_unsupported_class_excludes_frame_not_just_object(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root, "1,1,1,1,2,2,1,2,1\n1,2,5,5,2,2,1,1,1\n")
    assert prepare(root, output)["skipped"] == {"unsupported_class_frame": 1}


def test_existing_output_untouched(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root)
    output.mkdir()
    keep = output / "keep"
    keep.write_text("unchanged")
    with pytest.raises(FileExistsError):
        prepare(root, output)
    assert keep.read_text() == "unchanged"


def test_sequence_18_is_quarantined_not_merely_warned(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root)
    sequence(root, "test", "UAVSwarm-18", "not trusted or parsed\n", color="green")
    audit = prepare(root, output)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert all(row["group"] != "UAVSwarm-18" for row in rows)
    assert not (output / "labels" / "test" / "UAVSwarm-18").exists()
    assert audit["quarantined_groups"][0]["group"] == "UAVSwarm-18"
    assert audit["quarantined_groups"][0]["exported"] == {"images": 0, "boxes": 0}
    assert audit["skipped"]["quarantined_sequence_frames"] == 1
    assert audit["readiness"].startswith("not approved for training")


def test_completion_is_published_only_after_atomic_directory_prepare(tmp_path, monkeypatch):
    root, output = tmp_path / "raw", tmp_path / "out"
    fixture(root)
    original_rename = Path.rename
    observed = []

    def check_rename(path, destination):
        if Path(destination) == output:
            assert (path / "manifest.jsonl").is_file()
            assert not (path / "summary.json").exists()
            assert not output.exists()
            observed.append(True)
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", check_rename)
    prepare(root, output)
    assert observed == [True]
    assert json.loads((output / "summary.json").read_text())["complete"] is True
    assert not (output / ".summary.json.tmp").exists()
