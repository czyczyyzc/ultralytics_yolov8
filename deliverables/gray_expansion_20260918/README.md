# Approved Grayscale Expansion and Replacement Synthesis

Audit date: 2026-09-18. Server: `47.107.185.207`. Code synchronized by local Git push and server Git pull; generator revision at launch: `bf5d4e2`. No detector/tracker weights or GPU training jobs were launched by this task.

## Why the Previous Batch Had 856 Images

The earlier job capped selection at 20 source frames per video and four asset variants per frame. Its source resolver registered only 22 of the 28 training videos because six old human-adjudicated videos used a different annotation format. Four of those 22 approved registrations also pointed to historical approved-image directories that were not in the current training list: the current list used verified `manual_gray` exports instead. That left 18 videos with selected frames (360 frames total). Safe background reconstruction succeeded on 219 source frames from 17 videos, producing 856 unique accepted variants after asset/contrast/duplicate checks.

The 2026-09-17 rerun used the same seed/inputs and produced byte-identical copies of those 856 images. Its files must not be counted as new data. The seven earlier smoke-test outputs were additional variants, not seven new independent scenes.

The new resolver includes all six legacy training videos using their original human-adjudicated visible annotations and the explicit current training split. It does not alter the old `training_allowed=false` evaluation manifest or pretend these sources are newly platform-approved. Historical manual-image aliases require an exact video hash match and agreement between the historical YOLO box and the current approved COCO box.

## New Approved Data

`approved_snapshot.json` records 32 approved tasks at audit time: 22 already represented in training, nine new training candidates and one held-out validation task. All nine new videos passed video SHA256, annotation-file SHA256, frame classification and COCO/YOLO geometry checks. They contain 15,155 positive and 17,297 negative frames, with zero uncertain or unreviewed frames. A nearly empty new clip contributes only 21 positives and 6,004 negatives; it must not dictate the per-epoch class ratio.

The prepared real-data snapshot is:

```text
/mnt/andrew/anti_uav_model_refinement/data/real_gray_native_fullpool_37videos_20260918/
```

Use `train_hardneg_gray_monitor.yaml` with the existing `FixedShapeGrayP3Trainer` / `FixedShapeGrayAddOnTrainer` and `NativeExposureSampler`, not a generic loader that ignores `label_sampling`. `train_37videos.yaml` is a local reference copy, containing server paths.

| Item | Before | After |
| --- | ---: | ---: |
| Gray training video identities | 28 | 37 |
| Candidate training-list entries, including deliberate repetitions/RGB | 146,893 | 179,345 |
| Positive entries per epoch | 108,041 | 123,196 |
| Negative entries per epoch | 19,066 | 21,740 |
| Entries per epoch | 127,107 | 144,936 |
| Rotating negative-pool candidates | 25,540 | 42,837 |

The original 146,893-entry prefix is byte-for-entry preserved, as are validation-list contents. Both old and new rotating negatives remain eligible rather than becoming fixed anchors. There are 13,312 fixed negative anchors plus 8,428 sampled negatives per epoch, yielding 14.9997% negatives. Every rotating-pool label was checked to be empty. With an uninterrupted sampler schedule, six epochs cover the entire rotating pool. These counts describe real data only; synthetic candidates are not appended to this training configuration.

The nine new videos use positive_stride=1 and negative_stride=1, not temporal thinning. Video00004 and the original sunny/frontlight/stationary Video00009 remain excluded by exact video SHA256 from real training, source synthesis and temporal donors. Other video files containing the numbers 00004/00009 in their names are distinct identities, not automatically excluded. Exact hashes do not rule out reencoded/overlapping clips or same-session correlation; this remains a validation limitation.

## Expanded Synthesis Job

```text
/mnt/andrew/anti_uav_model_refinement/data/real_gray_replacement_expanded37_20260918/
```

Started with PID 3770721, four independent worker processes, CPU affinity 120-127, niceness 15, OMP/OpenBLAS limited to two threads. It uses 53 usable transparent assets from the existing catalog; opaque photos and accessory-containing IDs 24/25 remain excluded. No GPU is used.

`generation_plan.json` verifies 37/37 registered training identities, no missing adapters, and 3,604 selected source frames. Each video contributes up to 100 frames, stratified by original target size, with up to four different asset variants per frame. The nearly empty video has only four source frames inside the supported synthesis-size range. **14,416 is a maximum attempt count, not a completed or guaranteed accepted count.** Background/contrast gates are unchanged. Prior successful source frames and prior output hashes are excluded, including the earlier smoke tests. Rejected original images and large targets remain in real training.

The method still supports original-coordinate short edges >=3 px and long edges <=160 px; it does not promise replacement of every positive frame, large/full-frame objects or every opaque catalog photo. All training video identities are considered, but a video can still yield zero acceptable replacements if reconstruction is unreliable. It is frame-level augmentation, not a temporally consistent synthetic tracking video.

Job files:

- `job_status.json`: parent running/complete/failed state, PID and timestamps.
- `plan/plan.json`: per-video eligibility and selected-frame counts.
- `shard_00/` through `shard_03/`: independently auditable batches, each containing `images/`, `labels/`, `masks/`, previews and manifests.
- `shard_00.log` through `shard_03.log`: generation and full-verification logs.
- `shard_NN/verification.json`: created only after that shard completes and passes its integrity audit.
- `summary.json` and `train_synthetic_unique.txt`: created only after all four shards pass, with cross-shard output-hash deduplication.

The parent log is the job directory path with `.log` appended. This report records launch, not completion. Read current server status for the final accepted count. The generated training list is a **candidate review list**, not automatically enabled in training. Visual realism and training benefit still require review and a controlled experiment.

## Verification and Code

24 synthesis/source-isolation tests and 10 native-loader/negative-pool tests passed on the server. The legacy-format smoke batch produced eight verified samples. The subsequent 37-source four-worker smoke run completed with 16 verified unique samples across 16 videos; its full summary is `smoke_37_summary.json`. These are smoke tests, not the expanded batch result. A legacy-source preview was downloaded and visually inspected (`legacy_preview.png`).

New/updated entry points under `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/`:

- `audit_approved_gray_expansion.py`: freezes newly approved identities after source and label verification; use the main `.venv/bin/python` for YAML dependencies.
- `append_approved_gray_native.py`: builds a fresh real-data snapshot while preserving repeated baseline slots and earlier rotating negatives; uses the main `.venv/bin/python`.
- `build_gray_replacement_batch.py`: source adapters, hash-based holdout guards, prior-batch exclusion, `--plan-only`, `--include-legacy` and optional video shards.
- `run_gray_replacement_expansion.py`: bounded parallel synthesis, automatic per-shard verification and unique candidate-list publication; uses `.venv_gray_synthesis/bin/python`.
- `verify_gray_replacement_batch.py`: read-only verification of complete individual shards.

No prior dataset, source image, source label, held-out split or model deployment was replaced. A 37-video real-only training run and a separate real-plus-reviewed-synthetic comparison have **not** been started.
