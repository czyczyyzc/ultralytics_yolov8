import numpy as np
import pytest

from scripts.anti_uav.append_approved_gray_native import append_samples


def test_append_preserves_every_old_slot():
    base = ["/images/old/positive.jpg"] * 3 + ["/images/old/empty.jpg"]
    added = ["/images/new/b.jpg", "/images/new/a.jpg"]
    assert append_samples(base, added, ["/images/gray_val/a.jpg"]) == base + sorted(added)


@pytest.mark.parametrize("added", [["/images/new/a.jpg"] * 2, ["/images/old/a.jpg"],
                                  ["/images/Video00004/a.jpg"], ["/images/zoom_train/a.jpg"]])
def test_append_rejects_duplicates_and_leakage(added):
    with pytest.raises(ValueError):
        append_samples(["/images/old/a.jpg"], added, [])


def test_fixed_validation_preserves_native_box_and_shape(tmp_path):
    torch = pytest.importorskip("torch")
    import cv2
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayP3Trainer, FixedShapeGrayAddOnTrainer
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir(); labels.mkdir()
    cv2.imwrite(str(images / "a.jpg"), np.zeros((1080, 1920, 3), dtype=np.uint8))
    (labels / "a.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    for cls in (FixedShapeGrayP3Trainer, FixedShapeGrayAddOnTrainer):
        trainer = object.__new__(cls)
        trainer.args = get_cfg(overrides=dict(imgsz=[544, 960], rect=False, task="detect", workers=0))
        trainer.model, trainer.data = None, dict(names={0: "drone"}, nc=1)
        dataset = trainer.build_dataset(str(images), "val", 1)
        assert not dataset.rect
        batch = dataset.collate_fn([dataset[0]])
        assert tuple(batch["img"].shape) == (1, 3, 544, 960)
        validator = object.__new__(DetectionValidator)
        validator.device = torch.device("cpu")
        prepared = validator._prepare_batch(0, batch)
        np.testing.assert_allclose(prepared["bbox"].numpy(), [[720, 405, 1200, 675]], atol=.001)
