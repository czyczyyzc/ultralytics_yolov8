# Online Grayscale Target Replacement

## Design and Scope

- 37 explicitly allowlisted training videos, including the nine newly approved videos.
- 72,236 reviewed positive frames, with no per-video cap and stride 1.
- 11,663 positive originals missing from the earlier stride-sampled schedule are extracted once.
- 53 native-alpha cutouts; catalog entries 24 and 25 (controllers) remain excluded.
- 111 uncertain/unreviewed frames are excluded from replacement. Negatives are not replaced.
- Video00004 and the stationary Video00009 holdouts remain excluded by video SHA256.
- Existing training lists, repeated exposure, images, labels and validation files are not overwritten.

Expensive temporal registration/foreground removal is done once per frame. The cache stores
small lossless background ROIs, not 53 complete images. At training time a selected cutout is
rendered into the ROI and the normalized class-0 bbox is recomputed from its blurred alpha.
Outside the edit mask, original pixels are unchanged. Normal YOLO transformations follow.

Default policy: approximately 50% original and 50% attempted replacement for cache-ready
positive slots, balanced over epochs. Failed cutouts fall back to the original image/box.
Assets follow a seeded per-frame permutation. For one exposure per epoch, 106 epochs cover
53 attempted assets and 53 original slots; a 15-epoch run does NOT show each frame all assets.
The add-on stage continues the epoch offset from P3 instead of restarting the asset cycle.

Background repair currently supports original-frame short edge >=3 px and long edge <=160 px.
4,224 positive frames are outside that repair range. They remain in training as ORIGINALS;
this is not a detector area filter. Other unsafe backgrounds also retain their originals.
There is no claim of perfect blending, temporal pose consistency, or proven accuracy gains.

## Locations and Preparation

Server: `root@47.107.185.207`

```bash
REPO=/mnt/chenziye/codes/ultralytics_yolov8
DATA=/mnt/andrew/anti_uav_model_refinement/data
PY=$REPO/.venv/bin/python
CACHE=$DATA/real_gray_online_replacement_37videos_20260918

# First launch. Add --resume only for the same code/input plan after interruption.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 taskset -c 120-127 nice -n 15 "$PY" \
  "$REPO/scripts/anti_uav/prepare_online_gray_replacement.py" \
  --dataset "$DATA/real_gray_native_fullpool_37videos_20260918" \
  --catalog "$DATA/drone_asset_catalog_20260916/catalog.json" \
  --output "$CACHE" --workers 4
```

Inspect `job_status.json` and per-video `*.status.json`; preparation is CPU-only.
SQLite commits one source frame at a time; resume skips only committed frames.
The job pauses with an error below 25 GiB free disk space. A changed plan/code is rejected
on resume rather than silently mixing algorithms. No model training is auto-started.

After preparation completes, the output contains:

- `index.json`, `summary.json`, `plan.json`, and one background SQLite file per video.
- `images/` and `labels/`: only the previously missing ORIGINAL positive frames.
- `train_hardneg.txt`: exact baseline prefix/repetitions plus missing positive originals.
- `train_online_gray_monitor.yaml`: online replacement treatment.
- `train_control_gray_monitor.yaml`: same expanded data without replacement.
- `val_monitor.txt`: byte-identical baseline validation list.

The negative pool is unchanged; its per-epoch quota is recalculated to remain near 15%.
Planned epoch size is 134,859 positive slots + 23,799 negative slots = 158,658, not x53.
Final audited counts must be read from `summary.json` after successful completion.

## Training Integration and Verification

Use the repository's existing `run_rebalanced_fullscale_training.py` with
`--dataset "$CACHE" --data-yaml "$CACHE/train_online_gray_monitor.yaml" --fixed-validation`.
Its usual `--initial-p3`, `--old-run`, `--run-dir`, `--device`, and `--epochs` arguments
are still required. The P3 and add-on stages use `FixedShapeGrayP3Trainer` and
`FixedShapeGrayAddOnTrainer`; use a fresh run directory, not an old checkpoint's resume.
No Mosaic/MixUp/CopyPaste/online_scale combination is enabled by this implementation.

For a fair augmentation experiment, run the control YAML with the same initialization,
epochs, batch, seed and data. Comparing only with the earlier 28-video model confounds
new data, denser positive sampling and augmentation. Keep model selection on the existing
gray validation video and report Video00004 separately as test-only.

CPU smoke cache: `$DATA/online_gray_replacement_smoke_20260918`

QA: `$DATA/online_gray_replacement_smoke_20260918/qa/summary.json`

Six real source frames x 53 cutouts: 318/318 successful replacements; updated boxes,
outside-mask pixel equality and actual multiworker YOLO collation verified.
Mean cached ROI render time: 6.47 ms. Single-process cold image-load/YOLO-transform
time: original 9.11 ms, replacement 18.82 ms. These are a small CPU smoke test, not
GPU training throughput or a measured model-accuracy improvement.

Local `preview_contact_sheet.jpg` shows original and replaced targets with their respective
boxes. `smoke_benchmark.json` contains the machine-readable test results.
The previous 9,271-image offline batch remains separate and is not concatenated automatically.
