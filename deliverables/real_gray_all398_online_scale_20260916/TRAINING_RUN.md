# Online random-scale training launch: 2026-09-16

Latest operator decision: stopped by user on 2026-09-16 after requesting restoration
of native augmentation. The resumed process group 1292464 is no longer running;
the pending add-on stage will not run. All saved checkpoints and logs are retained.
See `../real_gray_native_augmentation_restored_20260916/README.md` for the prepared
replacement configuration. No replacement training job has been launched.

Host: `root@47.107.185.207`
Physical GPU: `6` (A100 80GB).
Launch PID: `3566740`. The existing approximately 0.6GB Triton service was retained.
Launch commit: `48261318cd28cf31becd339e3b275eb7ea24b051`.

## Run and Data

Run directory:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/frozen_p3_addon_p2_all398_online_scale_20260916`

Progress: `status.json`, `pipeline.log`, `training_p3/p3/results.csv`, then
`training_addon/p2/results.csv` within the run directory. This launch record is not
a live status report. The detached process survives this SSH session ending.

Dataset:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_all398_online_scale_v2_20260916/train_hardneg_gray_monitor.yaml`

Initial P3:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_strict_holdout_Video00004_newclips01_20260902/training/strict_holdout_Video00004_neg15_newclips01_v1_20260902/weights/best.pt`

## Protocol

- Train P3 for 15 epochs, then freeze P3 and train add-on P2 for 15 epochs.
- Input: width 960, height 544. Batch: 64. Nominal effective batch: 128.
- Seed: 20260915, matching the static-augmentation reference.
- P3 initial learning rate: 0.0001; add-on P2: 0.001. AdamW, cosine schedule.
- Keep all 89,748 native schedule entries and their hard-negative replacements.
- Add four freshly sampled views per eligible donor per epoch: 398 donors, 1,592 views.
- Total per epoch: 91,340 samples, 1,428 batches on this single GPU.
- Gray training sources: 14 videos, alongside the preserved Anti-UAV300 schedule.
- Native-gray validation: whole held-out stationary video00009, sampled to 1,421 frames.
- Fixed synthetic scale stress validation: 64 derived views, diagnostic only.
- Checkpoint fitness uses native-gray F2(conf=.03), mAP50 and mAP50-95, not RGB metrics.
- Video00004 is test-only; it is never used for training, mining or checkpoint selection.

At launch verification, epoch 1 was advancing with finite box / class / DFL losses
and approximately 10.4GB training GPU memory. No completion or performance gain
is claimed by this record.

## Interrupted Run and Recovery

The original process failed on 2026-09-16 during P3 epoch 15 (479/1428 batches)
due to an integer crop-feasibility bug. Fourteen completed epochs were preserved;
add-on P2 had not started. See `RESULTS_20260916.md` for the measured comparison
and validation-shape caveat. Do not interpret the original launch PID as active.

Fix commit: `84a793c`. Recovery PID: `1292464`, same GPU 6, using `--resume-p3`.
Recovery log: `pipeline_resume_20260916.log`; live state remains `status.json`.
Failed-state backups are in `failure_backup_20260916/`, and a `recovery_*.json`
record stores the checkpoint and code revisions. Optimizer and EMA are restored;
the interrupted epoch is rerun before the pending 15-epoch add-on stage.
The original protocol and log are preserved. No full-model improvement is claimed.

## Outputs and Comparison

The runner automatically proceeds to add-on training, verifies frozen legacy P3
outputs, and finally evaluates old and new models on Video00004.
Expected weights, created only after the corresponding training stages run:

- `training_p3/p3/weights/best.pt`
- `training_addon/p2/weights/best.pt`
- `training_addon/p2/weights/last.pt`

Final test outputs: `evaluation/`, `comparison.json`, `COMPARISON.md`.
Those automatic comparison files use the old manual-clips0123 models as reference.
For the specific static-versus-online augmentation comparison, also compare with
the completed sibling run:

`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/frozen_p3_addon_p2_rebalanced_grayval_fullscale_20260915`

The sibling used 480 fixed augmentation images and 90,228 total entries per epoch.
The new run uses 1,428 rather than 1,410 batches per epoch, so this is a matched-
epoch comparison, not an exactly matched-optimizer-step experiment. Models remain
PT reference results until separately exported and verified on RKNN INT8 hardware.
