# Full-scale gray training: 2026-09-15

## Policy

All valid original GT boxes and positive sampling multiplicities are retained.
The obsolete 25% prediction-area ceiling has been removed from the Python board
adapter and replay entry points. This was an inference heuristic, not a training
annotation rule. Native C++ RK-BoT-SORT had no equivalent ceiling in the inspected
code. Confidence thresholds and NMS remain necessary; keeping every raw network
prediction would retain duplicate and unreliable boxes.

Size here means either the GT long edge after 960x544 letterboxing, or GT bbox area
divided by image area. Bbox area is not the drone silhouette's pixel area.

## Verified Data

- Original schedule: 86,505 entries; original prefix preserved before hard-negative replacement.
- Original positive exposure: 73,529 entries, unchanged after negative replacement.
- Original 4-8 input-pixel GT exposure: 30,464 box instances, unchanged.
- Appended native schedule: 89,748 entries, including 76,436 positives and 13,312 negatives.
- Added zoom positives: 480 from 220 unique self-collected gray training frames.
- Final schedule: 90,228 entries; 14.7537% negatives.
- Hard negatives: 461 replacements of easy negative slots, no positive or per-video negative count changes.
- Each zoom source contributes at most four samples; resize gain is at most 4x.
- Small, insufficiently resolved boxes are not used as zoom donors, but their originals remain in training.
- RGB originals remain in training; they are not zoom donors because sampled RGB footage contained baked-in camera HUDs.

| Added bbox area fraction | Samples |
| --- | ---: |
| 5%-10% | 48 |
| 10%-25% | 96 |
| 25%-50% | 96 |
| 50%-80% | 162 |
| 80%-99% | 30 |
| 99%-100% | 48 |

Close-up augmentation includes clipped, partly out-of-frame drones. Labels are
transformed and clipped with the image crop. These images do not supply missing
fine detail or prove accuracy on real close-range footage. The newly added native
videos have no >=10% GT boxes; real large-target footage is still needed for a
representative large-target test.

## Split and Selection

Fourteen self-collected videos train alongside the preserved Anti-UAV300 schedule.
The entire longest `stationary_video00009` is reserved for validation: 1,421 native
frames plus 64 derived scale-stress frames, never used for training. Other videos
from the same date remain in training at the user's request; this is whole-video
separation, not a claim of independent acquisition sessions.

Video00004 remains test-only. It is never used for mining, augmentation or epoch
selection. New weights use native-gray selection fitness:
`0.5 * F2(conf=.03) + 0.3 * mAP50 + 0.2 * mAP50-95`.
Precision, recall and FP/1000 are logged at confidence .01, .03 and .05. Recall is
also logged by pixel size and bbox area, separately for native and augmented
validation. Bins with no GT return NaN, not measured zero recall. Native validation
contains 392 4-8px GT boxes but no >=80% boxes; synthetic validation has eight >=80%
boxes. Synthetic validation is diagnostic only and does not select checkpoints.

## Server Paths

Host: `root@47.107.185.207`.

Dataset:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_rebalanced_grayval_fullscale_v5_20260915`

Training YAML: dataset directory + `/train_hardneg_gray_monitor.yaml`.
Native validation images are reused, read-only, from the verified v3 split.
The earlier v1/v2/v4 preparation attempts are incomplete and must not be trained on;
v3 contains the superseded mixed RGB/gray zoom set and is not the training dataset.

Run:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/frozen_p3_addon_p2_rebalanced_grayval_fullscale_20260915`

Run files: `pipeline.log`, `status.json`, `protocol.json`.
Training is P3 15 epochs, then frozen P3 with trainable add-on P2 for 15 epochs,
physical GPU 6, batch 64, effective nominal batch 128, input 960x544.
The existing small Triton process is not stopped.

Expected checkpoint paths, produced only after their stages finish:
- `training_p3/p3/weights/best.pt`
- `training_addon/p2/weights/best.pt`
- `training_addon/p2/weights/last.pt`

After training, the runner verifies bit-exact frozen P3 outputs and evaluates old
and new P3/add-on weights on Video00004. Results will be in `evaluation/`,
`comparison.json` and `COMPARISON.md`. These are PT-reference detector results, not
RKNN INT8 deployment validation, tracker metrics or board FPS.

Code was committed locally, pushed to origin and fast-forward pulled on the server
through Git bundles because server-to-GitHub SSH was unavailable. Launch commit:
`b731e32`. Nine regression tests passed; gray validation smoke also passed.

## Local QA

`audit/audit.json` records exact schedule and scale counts.
`audit/scale_preview.jpg` shows transformed GT boxes on augmentation examples.
`audit/preview_sources.json` records the exact source frames and crop transforms.
`audit/hard_negative_preview.jpg` shows six high-scoring labeled-empty candidates.
Only `audit/scale_preview.jpg` is the final gray-only preview; the top-level
`scale_preview.jpg` is the superseded mixed RGB/gray QA image.
