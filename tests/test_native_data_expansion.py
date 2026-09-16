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


def test_label_pool_covers_all_negatives_and_keeps_anchors(tmp_path):
    from scripts.anti_uav.label_pool_sampling import NativeExposureSampler
    class Dataset:
        labels = [dict(im_file=f"/images/{i}.jpg", cls=np.ones((1, 1)) if i < 4 else np.empty((0, 1)))
                  for i in range(14)]
        def __len__(self):
            return len(self.labels)
    pool = tmp_path / "negative.txt"
    pool.write_text("\n".join(f"/images/{i}.jpg" for i in range(6, 14)))
    cfg = dict(negative_pool=str(pool), negative_pool_count=8, negatives_per_epoch=3, anchor_slots=6)
    sampler = NativeExposureSampler(Dataset(), cfg)
    seen, orders = set(), []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert len(indices) == len(set(indices)) == len(sampler) == 9
        assert set(range(6)) <= set(indices)
        seen.update(set(indices) - set(range(6)))
        orders.append(indices)
    assert seen == set(range(6, 14))
    sampler.set_epoch(1)
    assert list(sampler) == orders[1]
    Dataset.labels[6]["cls"] = np.ones((1, 1))
    with pytest.raises(ValueError, match="positive label"):
        NativeExposureSampler(Dataset(), cfg)


def test_real_training_loader_cycles_negative_pool(tmp_path):
    pytest.importorskip("torch")
    import cv2
    from collections import defaultdict
    from ultralytics.cfg import get_cfg
    from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayP3Trainer
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir(); labels.mkdir()
    for i in range(5):
        cv2.imwrite(str(images / f"{i}.jpg"), np.zeros((108, 192, 3), dtype=np.uint8))
        (labels / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n" if i < 2 else "")
    pool = tmp_path / "negative.txt"
    pool.write_text("\n".join(str(images / f"{i}.jpg") for i in (3, 4)))
    trainer = object.__new__(FixedShapeGrayP3Trainer)
    trainer.args = get_cfg(overrides=dict(imgsz=[544, 960], rect=False, task="detect", workers=2,
                                         close_mosaic=0, mosaic=0., mixup=0., copy_paste=0., seed=7))
    trainer.model, trainer.callbacks = None, defaultdict(list)
    trainer.data = dict(names={0: "drone"}, nc=1, label_sampling=dict(negative_pool=str(pool),
                        negative_pool_count=2, negatives_per_epoch=1, anchor_slots=3))
    trainer.train_loader = trainer.get_dataloader(str(images), batch_size=2, rank=-1)
    seen = []
    for epoch in range(2):
        trainer.epoch = epoch
        for callback in trainer.callbacks["on_train_epoch_start"]:
            callback(trainer)
        paths = []
        for batch in trainer.train_loader:
            assert tuple(batch["img"].shape[1:]) == (3, 544, 960)
            paths.extend(batch["im_file"])
        assert len(paths) == 4 and len(set(paths)) == 4
        assert {str(images / f"{i}.jpg") for i in range(3)} <= set(paths)
        seen.extend(paths)
    assert set(seen) == {str(images / f"{i}.jpg") for i in range(5)}
