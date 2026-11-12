"""Explicit trainer variants: never change the running legacy experiment's defaults."""

import json

from scripts.anti_uav.balanced_regression_repair import BalancedRegressionDetectionLoss
from scripts.anti_uav.p3_student_distillation import (
    P3StudentFP32DistillationTrainer, TinyAwareAddOnTrainer, TinyAwareP3Trainer,
)
from ultralytics.utils.torch_utils import de_parallel


def repair_losses(trainer):
    criterion = de_parallel(trainer.model).criterion
    if isinstance(criterion, BalancedRegressionDetectionLoss):
        return {'detector': criterion}
    return {name: getattr(criterion, name) for name in ('primary', 'auxiliary')
            if isinstance(getattr(criterion, name, None), BalancedRegressionDetectionLoss)}


def begin_repair_epoch(trainer):
    trainer.visible_box_guard_totals = None
    for loss in repair_losses(trainer).values():
        loss.begin_epoch(trainer.epoch)


def save_repair_epoch(trainer):
    output = trainer.save_dir/'regression_repair_audit'
    output.mkdir(exist_ok=True)
    report = dict(epoch=trainer.epoch+1, training_only=True,
                  losses={name: loss.audit_summary() for name, loss in repair_losses(trainer).items()})
    if trainer.visible_box_guard_totals is not None:
        keys = ("input_boxes", "retained_boxes", "offscreen_removed", "visible_rejections_rescued", "fallback_images")
        if trainer.data.get("visible_box_guard") == "affine_keep_subpixel_v2":
            keys = ("input_boxes", "retained_boxes", "offscreen_removed", "at_most_2px_kept", "invalid_projection_fallback")
        report["visible_box_guard"] = dict(zip(keys, trainer.visible_box_guard_totals))
    (output/f'epoch_{trainer.epoch+1:04d}.json').write_text(json.dumps(report, indent=2)+'\n')


def repaired_loss(model, source, previous, options, first_level_only=False,
                  loss_type=BalancedRegressionDetectionLoss):
    keys = ('tiny_multiplier', 'tiny_min_px', 'tiny_max_px', 'localization_weight',
            'localization_min_px', 'localization_max_px')
    legacy = {key: getattr(previous, key) for key in keys if hasattr(previous, key)}
    # The Student auxiliary loss was v8DetectionLoss, not a tiny-weighted loss.
    if not hasattr(previous, 'tiny_multiplier'):
        legacy.update(tiny_multiplier=1., localization_weight=0.)
    result = loss_type(model, source, first_level_only=first_level_only, **options, **legacy)
    result.stride = previous.stride
    return result


class RepairMixin:
    repair_options = dict(balance_mix=.5, warmup_epochs=3, tiny_log_size_weight=.5)

    def install_repair_callbacks(self):
        self.add_callback('on_train_epoch_start', begin_repair_epoch)
        self.add_callback('on_train_epoch_end', save_repair_epoch)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        records = batch.get("visible_box_guard")
        guard = self.data.get("visible_box_guard")
        if guard in {"affine_fallback_v1", "affine_keep_subpixel_v2"}:
            if records is None:
                raise ValueError("Missing augmentation guard audit for a protected training batch")
            batch_counts = tuple(map(sum, zip(*records)))
            before, after, offscreen = batch_counts[:3]
            if before != after + offscreen or len(batch["cls"]) != after:
                raise ValueError("A protected label disappeared after the affine stage")
            if guard == "affine_fallback_v1":
                batch_counts += (sum(r[3] > 0 for r in records),)
            totals = getattr(self, "visible_box_guard_totals", None) or [0]*5
            self.visible_box_guard_totals = [a+b for a,b in zip(totals, batch_counts)]
        return batch


class RegressionRepairP3Trainer(RepairMixin, TinyAwareP3Trainer):
    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        model = de_parallel(self.model)
        head = model.model[-1]
        loss_type = BalancedRegressionDetectionLoss
        if hasattr(head, "p3_negative_bin"):
            from scripts.anti_uav.p3_signed_dfl import P3SignedDFLLoss

            loss_type = P3SignedDFLLoss
        elif hasattr(head, "p3_offset"):
            from scripts.anti_uav.p3_nearest_offset import P3NearestOffsetLoss

            loss_type = P3NearestOffsetLoss
        previous = model.criterion
        model.criterion = repaired_loss(model, self.data, previous, self.repair_options, loss_type=loss_type)
        if loss_type is not BalancedRegressionDetectionLoss:
            self.ema.ema.criterion = repaired_loss(self.ema.ema, self.data, previous, self.repair_options,
                                                   loss_type=loss_type)
        self.install_repair_callbacks()


class RegressionRepairAddOnTrainer(RepairMixin, TinyAwareAddOnTrainer):
    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        model = de_parallel(self.model)
        model.criterion = repaired_loss(model, self.data, model.criterion, self.repair_options,
                                        first_level_only=True)
        self.install_repair_callbacks()


class RegressionRepairStudentTrainer(RepairMixin, P3StudentFP32DistillationTrainer):
    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        model = de_parallel(self.model)
        criterion = model.criterion
        criterion.primary = repaired_loss(model, self.data, criterion.primary, self.repair_options)
        criterion.auxiliary = repaired_loss(model, self.data, criterion.auxiliary, self.repair_options)
        self.install_repair_callbacks()
