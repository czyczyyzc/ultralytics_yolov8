# All-donor online random scale

This is an opt-in augmentation implementation and verified data configuration.
A new long training run was launched on 2026-09-16 at the user's request; see
`TRAINING_RUN.md` for the launch record. The completed 2026-09-15 experiment
and its 480 fixed augmentation images are preserved for comparison.

Status update: the original launch failed in P3 epoch 15. The crop bug was fixed
and checkpoint recovery launched; see `RESULTS_20260916.md` for the recovery and
matched-epoch evaluation. The interim P3 results do not show an overall gain.

## Recipe

- Preserve all 89,748 native training slots, including positive repetitions and hard-negative replacements.
- Replace only the previous extra 480 static views with dynamic-view slots.
- All 398 unique eligible self-collected gray training images participate.
- Eligibility is native-image bbox short edge >=48px and long edge >=96px; single-target donor selection only.
- Each donor contributes four extra views per epoch by default: 1,592 extra slots, 91,340 total.
- Every visit samples a new crop, position and scale. A seeded worker RNG makes the sequence reproducible.
- Choose uniformly among feasible bbox-area bins: 2%-10%, 10%-25%, 25%-50%, 50%-80%, 80%-100%.
- Within a selected bin, sample log-uniform area. Pixel and geometry limits can trigger a fallback to an intact view.
- Do not force all donors to 80% or full-frame: enforce <=4x resize from the native crop.
- Request partially truncated views with probability 15%; retain at least 60% of the source box area.
- Otherwise retain all of the source box. The source image itself can already contain a partially visible drone.
- Apply mild gray brightness/contrast variation, low-probability noise / blur, and horizontal flips to extra views.
- Do not apply Mosaic, MixUp, CopyPaste or a second random geometric transform to these extra views.
- Original slots retain the baseline training transforms (including scale=.2); original 4-8px targets are never used as large-target crop donors.
- Validation and test data remain unchanged and are never online augmented.

These are starting settings, not demonstrated optimal hyperparameters. The added
views are 1.74% of an epoch. If further balancing is needed, change the per-donor
view count and size sampling after examining independent validation; do not remove
old small-target exposure or duplicate near-identical frames blindly.

Original annotations are unchanged. All new boxes follow image crop / clipping /
resize / flip exactly. The 0.01px boundary tolerance only handles decimal label
serialization; it does not drop border boxes or change native label files.

## Measured QA

All 398 donors were exercised in 1,592 draws, with no skipped donors. This seeded
preview draw contained 518 boxes at 2%-10%, 456 at 10%-25%, 438 at 25%-50%, 142 at
50%-80%, and 38 at >=80% area. Actual training draws vary. Size bins are not globally
balanced because many sources cannot reach the larger sizes without excessive
upsampling or truncation.

Of 231 requested partial draws, 38 actually truncated a source box; others remained
intact or fell back to intact crops. Maximum resize was 3.9834x; minimum retained
source-box area was 61.27%. None of this guarantees fine detail or real near-field
accuracy. Real close-range footage is still required for representative evaluation.

Fifteen unit/regression tests passed. Two mixed batches of native positives,
negatives and dynamic samples passed the two-worker loader test at [4,3,544,960].
Repeated reads differ; resetting the random seed reproduces the sample and label.
Both P3 and add-on trainer dataset hooks were exercised. A mixed P3 batch also
completed CPU forward/backward with finite losses and gradients. No optimizer step
was taken, and existing weights were not modified.

## Usage

Server: `root@47.107.185.207`

Prepared dataset:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_all398_online_scale_v2_20260916`

Config: dataset directory + `/train_hardneg_gray_monitor.yaml`.
The previous same-named directory without `_v2` is an incomplete preparation attempt.

The opt-in `online_scale` YAML section is consumed by `GrayP3Trainer` and
`GrayAddOnTrainer`. A generic YOLO trainer will not create the virtual slots.
The existing `scripts/anti_uav/run_rebalanced_fullscale_training.py` runner supports
the prepared dataset; use a fresh `--run-dir` when starting an experiment.
Use the same pre-Video00009 initial P3 checkpoint and selection protocol as the
baseline, not a model previously trained on that validation video.

Rebuild with another seed or number of views, into a fresh output:

```bash
.venv/bin/python scripts/anti_uav/prepare_online_random_scale.py \
  --source /mnt/andrew/anti_uav_model_refinement/data/real_gray_rebalanced_grayval_fullscale_v5_20260915 \
  --output /path/to/new/dataset \
  --views-per-donor 4 --seed 20260916 --smoke-loader
```

Keep the same held-out Video00009 for native gray checkpoint selection and the
same fixed synthetic scale stress set for diagnostics. Video00004 remains test-only.
Comparisons should state the increased training steps caused by extra slots, and
ideally also include a fixed-step comparison. Do not infer a performance gain from
augmentation preview quality alone.

Local files: `scale_preview.jpg`, `preview_sources.json`, `online_scale_audit.json`.
Preview images are QA draws, not the on-disk training dataset.

Official reference for the existing scale parameter:
https://docs.ultralytics.com/guides/yolo-data-augmentation/#scale-scale
