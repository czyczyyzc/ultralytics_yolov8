# 40-Video Online Drone Replacement Training

## Audit

Server: `root@47.107.185.207`.
Repository: `/mnt/chenziye/codes/ultralytics_yolov8`.

The 2026-09-21 audit found 35 approved-task manifests: 22 videos already in the
28-video training set, 12 new training candidates, and one held-out validation
video. Six additional training videos come from the legacy registry, not these
35 approved-task folders. The new training split therefore contains 40 gray
videos, not 35. Existing Anti-UAV300 RGB entries and historical repetition slots
are preserved.

- Since the last trained 28-video version: 12 videos, 24,067 positive frames,
  25,343 explicit negative frames. No uncertain or unreviewed frames occur in
  these 12 manifests. Video SHA256, annotation file hashes, frame sets and
  COCO/YOLO box agreement passed verification.
- Since the prepared 37-video dataset: three videos, 8,912 positive and 8,046
  negative frames. Only these three require new frame extraction/backgrounds.
- `Video00004`: final test only, excluded by video hash from training/augmentation.
- `Video00009`: unchanged validation subset for checkpoint selection, not an
  untouched final test. Neither it nor its backgrounds enter augmentation.
- Exact video hashes do not rule out reencoded overlap or same-session
  correlations. This is not a session-independent generalization claim.

Audit file:
`/mnt/andrew/anti_uav_model_refinement/data/approved_expansion_snapshot_20260921.json`

## Data and Augmentation

Native expanded dataset:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_native_fullpool_40videos_20260921`

Online dataset and cache:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_online_replacement_40videos_20260921`

The online training YAML is `train_online_gray_monitor.yaml`. It becomes usable
only when `extension_status.json` and `summary.json` report `complete`.
`train_control_gray_monitor.yaml` contains the identical original-image schedule
without replacement, for a future augmentation-only ablation.

The 37-video cache is reused through read-only SQLite links after checking code,
asset catalog, annotations, frame geometry and database integrity. Keep the old
`real_gray_online_replacement_37videos_20260918` directory and its images intact.
New videos receive the same background preparation and rejection policy.

- Use the original 53 enabled transparent cutouts, not the 275 newly segmented
  candidates which still need actual-scale compositing validation.
- Replacement probability is 0.5 for eligible cached positives. An epoch-aware,
  seeded asset cycle varies the aircraft type reproducibly.
- Update bbox from the rendered target. Preserve pixels outside the edit mask.
- Keep negatives unchanged. Keep original image and bbox if background repair
  or cutout rendering is unsafe. Large targets outside the compositing size
  range remain training positives; the size gate is not a detector area filter.
- Preserve the existing YOLO transforms (`scale=.2`, `translate=.05`,
  `fliplr=.5`; no Mosaic/MixUp/CopyPaste or extra large-scale views).
- Reviewed new positives use stride 1. Explicit negatives enter the rotating
  label pool, with about 15% negative epoch exposure while retaining old anchors.
  All negative candidates are not necessarily visited in every epoch.
- No 53-fold image materialization. A 30-epoch two-stage run does not expose
  every original frame to all 53 cutouts.

## Training and Results

Experiment directory:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online53_20260921`

Protocol: P3 15 epochs, then frozen-P3 add-on P2 15 epochs, batch 64, input
960x544, single GPU 0. The native-exposure sampler deliberately supports one GPU;
do not pass a multi-GPU device list without adding/test-validating distributed
sampling. GPU availability is rechecked immediately before training.

Initialization uses the same pretrained P3 checkpoint as the 28-video experiment,
with a fresh optimizer. This is a fresh run, not random initialization and not a
resume of the deployed 28-video checkpoint. Both old checkpoints remain intact.

Launcher:
`scripts/anti_uav/run_online_gray_expansion_training.py`

It waits for cache completion, fails closed on preparation errors, runs the real
online YOLO data-loader/bbox smoke test, writes 20 original/replacement previews,
and then invokes `run_rebalanced_fullscale_training.py`. Final detector evaluation
compares old/new P3 and old/new add-on at fixed 960x544, NMS IoU .45, match IoU .5,
conf .01/.03/.05. This is PT FP32 evaluation, not board RKNN/INT8 validation.

Runtime files are authoritative:

- `experiment_status.json`: waiting, QA, training, evaluation, complete or failed.
- `status.json`: training stage and latest completed epoch.
- `logs/training.log`: training output.
- `training_p3/p3/results.csv` and `training_addon/p2/results.csv`: loss/metrics.
- `training_p3/p3/weights/best.pt`: new pure P3, only after training creates it.
- `training_addon/p2/weights/best.pt`: new frozen-P3/P2, only after training creates it.
- `frozen_checks.json`: P3 output preservation checks after add-on training.
- `COMPARISON.md` and `comparison/results.json`: written after final evaluation.
- `augmentation_preview/index.html`: original/replacement images with updated boxes.

Adding data and enabling replacement together tests the combined training change.
It does not isolate augmentation benefits from added-data benefits or the extra
optimizer steps caused by a larger dataset at the same epoch count.

Code synchronization: local Git commit/push followed by server fast-forward pull.
If server GitHub SSH times out, pull an incremental bundle of the pushed commit;
do not edit training code only on the server.
