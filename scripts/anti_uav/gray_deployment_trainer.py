"""Gray validation selection at fixed deployment confidence with separate zoom metrics."""

from collections import Counter, defaultdict
from copy import copy

import numpy as np

from scripts.anti_uav.frozen_p3_addon_p2_trainer import FrozenP3AddOnP2Trainer
from scripts.anti_uav.lovo_detection_trainer import LovoDetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class, box_iou


def deployment_fitness(native_f2, native_ap50, native_ap5095):
    return .5*native_f2 + .3*native_ap50 + .2*native_ap5095


def scale_masks(boxes, original_hw, input_hw):
    height, width = original_hw
    gain = min(input_hw[0]/height, input_hw[1]/width)
    wh = boxes[:, 2:]-boxes[:, :2]
    long_edge = wh.max(axis=1)*gain
    area = wh.prod(axis=1)/(height*width)
    return {"long_lt4px": long_edge < 4, "long_4to8px": (long_edge >= 4) & (long_edge <= 8),
            "long_8to16px": (long_edge > 8) & (long_edge <= 16),
            "long_gt16px": long_edge > 16,
            "area_lt5pct": area < .05, "area_5to10pct": (area >= .05) & (area < .1),
            "area_10to25pct": (area >= .1) & (area < .25),
            "area_25to50pct": (area >= .25) & (area < .5),
            "area_50to80pct": (area >= .5) & (area < .8), "area_ge80pct": area >= .8}


def matched_ground_truth(iou):
    """Use the validator's IoU=0.5 greedy one-to-one policy, returning GT indices."""
    pairs = np.array(np.nonzero(iou >= .5)).T
    if len(pairs) > 1:
        pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs[:, 0]


class GrayDeploymentValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.fixed = defaultdict(Counter)
        self.scale_counts = defaultdict(Counter)
        self.group_stats = {g: dict(tp=[], conf=[], pred_cls=[], target_cls=[]) for g in ("native", "zoom")}

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            group = "zoom" if "/zoom_val/" in batch["im_file"][si] else "native"
            prepared = self._prepare_batch(si, batch)
            boxes, classes = prepared["bbox"], prepared["cls"]
            masks = scale_masks(boxes.cpu().numpy(), prepared["ori_shape"], prepared["imgsz"])
            native_pred = self._prepare_pred(pred, prepared)
            correct = self._process_batch(native_pred, boxes, classes)
            record = self.group_stats[group]
            record["tp"].append(correct.cpu().numpy())
            record["conf"].append(native_pred[:, 4].cpu().numpy())
            record["pred_cls"].append(native_pred[:, 5].cpu().numpy())
            record["target_cls"].append(classes.cpu().numpy())
            for conf in (.01, .03, .05):
                selected = native_pred[native_pred[:, 4] >= conf]
                tp = int(self._process_batch(selected, boxes, classes)[:, 0].sum())
                counter = self.fixed[(group, conf)]
                counter["tp"] += tp
                counter["fp"] += len(selected)-tp
                counter["fn"] += len(boxes)-tp
                counter["frames"] += 1
                iou = (box_iou(boxes, selected[:, :4]) * (classes[:, None] == selected[:, 5])).cpu().numpy()
                matched = matched_ground_truth(iou)
                for size, mask in masks.items():
                    scale = self.scale_counts[(group, conf, size)]
                    scale["gt"] += int(mask.sum())
                    scale["tp"] += int(mask[matched].sum())

    def get_stats(self):
        stats = super().get_stats()
        for group, lists in self.group_stats.items():
            arrays = {key: np.concatenate(value) if value else np.empty((0, 10) if key == "tp" else (0,))
                      for key, value in lists.items()}
            map50 = map5095 = 0.
            if len(arrays["target_cls"]) and len(arrays["conf"]):
                metric = ap_per_class(**arrays, plot=False)
                map50, map5095 = float(metric[5][:, 0].mean()), float(metric[5].mean())
            stats[f"{group}/mAP50"] = map50
            stats[f"{group}/mAP50-95"] = map5095
            for conf in (.01, .03, .05):
                c = self.fixed[(group, conf)]
                precision = c["tp"]/max(c["tp"]+c["fp"], 1)
                recall = c["tp"]/max(c["tp"]+c["fn"], 1)
                f2 = 5*precision*recall/max(4*precision+recall, 1e-12)
                key = f"{group}/c{conf:.2f}"
                stats.update({f"{key}/P": precision, f"{key}/R": recall, f"{key}/F2": f2,
                              f"{key}/FP1000": 1000*c["fp"]/max(c["frames"], 1)})
        stats["fitness"] = deployment_fitness(stats["native/c0.03/F2"], stats["native/mAP50"], stats["native/mAP50-95"])
        for (group, conf, size), counts in sorted(self.scale_counts.items()):
            key = f"{group}/c{conf:.2f}/{size}"
            stats[f"{key}/GT"] = counts["gt"]
            stats[f"{key}/R"] = counts["tp"]/counts["gt"] if counts["gt"] else float("nan")
        self.metrics.gray_selection = dict(stats)
        return stats


class GraySelectionMixin:
    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        config = self.data.get("online_scale")
        if mode == "train" and config:
            if self.args.mosaic or self.args.mixup or self.args.copy_paste:
                raise ValueError("Online context views must not be mixed with Mosaic/MixUp/CopyPaste")
            from scripts.anti_uav.online_random_scale import OnlineScaleDataset
            dataset = OnlineScaleDataset(dataset, config)
        return dataset

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        args = copy(self.args)
        args.conf, args.iou, args.max_det = .001, .45, 100
        return GrayDeploymentValidator(self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks)


class GrayP3Trainer(GraySelectionMixin, LovoDetectionTrainer):
    pass


class GrayAddOnTrainer(GraySelectionMixin, FrozenP3AddOnP2Trainer):
    pass


class FixedShapeGrayValidator(GrayDeploymentValidator):
    """New experiments only: assert the real deployment canvas for every batch."""

    def preprocess(self, batch):
        result = super().preprocess(batch)
        if tuple(result["img"].shape[2:]) != (544, 960):
            raise ValueError(f"Expected fixed 544x960 validation: {result['img'].shape}")
        return result


class FixedShapeSelectionMixin:
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank not in (-1, 0) or self.args.close_mosaic:
            raise ValueError("Native exposure sampling requires single-GPU training and close_mosaic=0")
        import torch
        from torch.utils.data import DataLoader
        from ultralytics.data.build import seed_worker
        from scripts.anti_uav.label_pool_sampling import NativeExposureSampler, set_native_sampler_epoch
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        sampler = NativeExposureSampler(dataset, self.data.get("label_sampling"), self.args.seed)
        if set_native_sampler_epoch not in self.callbacks["on_train_epoch_start"]:
            self.callbacks["on_train_epoch_start"].append(set_native_sampler_epoch)
        generator = torch.Generator().manual_seed(self.args.seed)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=self.args.workers,
                          pin_memory=True, collate_fn=dataset.collate_fn, worker_init_fn=seed_worker,
                          generator=generator, persistent_workers=False)

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "val":
            return super().build_dataset(img_path, mode, batch)
        from ultralytics.data import build_yolo_dataset
        from ultralytics.utils.torch_utils import de_parallel
        args = copy(self.args)
        args.rect = False
        stride = max(int(de_parallel(self.model).stride.max()) if self.model else 0, 32)
        return build_yolo_dataset(args, img_path, batch, self.data, mode=mode, rect=False, stride=stride)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        args = copy(self.args)
        args.rect, args.conf, args.iou, args.max_det = False, .001, .45, 100
        return FixedShapeGrayValidator(self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks)


class FixedShapeGrayP3Trainer(FixedShapeSelectionMixin, GrayP3Trainer):
    pass


class FixedShapeGrayAddOnTrainer(FixedShapeSelectionMixin, GrayAddOnTrainer):
    pass
