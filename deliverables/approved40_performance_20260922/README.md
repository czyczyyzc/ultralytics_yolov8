# P3 / Frozen-P3 + Add-on P2 Performance Experiments

## Scope

This is an isolated optimization experiment, not a deployment replacement.
Old checkpoints, RKNN files and runtime defaults are unchanged. No maximum-area
box filter is added. Video00004 is not evaluated or mined in this tuning phase.
Video00009 remains the checkpoint-selection/calibration validation video, not an
independent test. A new untouched video is desirable for final generalization
assessment because Video00004 has already been inspected repeatedly.

## Controlled Training

Server: `root@47.107.185.207`.
Repository: `/mnt/chenziye/codes/ultralytics_yolov8`.
Runtime: repository `.venv/bin/python`, Python 3.10.12, PyTorch 2.5.1+cu121.

| Arm | GPU | Replacement probability | Directory under `runs/anti_uav/` |
| --- | ---: | ---: | --- |
| Original images | 0 | 0 | `approved40_prob00_v2_20260922` |
| Low-probability replacement | 1 | 0.15 | `approved40_prob15_v2_20260922` |
| Completed reference | Previously 1 | 0.50 | `approved40_online328_20260922` |

Both new arms were launched at 12:28 CST on 2026-09-22. They run P3 for 15 epochs,
then freeze P3 and train the add-on P2 for 15 epochs. Batch size is 64, input is
960x544, and each epoch has 2,643 batches. The initial checkpoint, seed, optimizer,
learning rates, base YOLO augmentation, gray/RGB schedule, historical repetitions,
rotating negative pool and checkpoint-selection rule match the reference.
Negative exposure remains about 15%; the 0.15 above is a different quantity:
the probability of attempting a replacement for an eligible cached positive.
Rejected replacements still use the original image and label.

The 328-entry catalog and background cache are reused without modification:

`/mnt/andrew/anti_uav_model_refinement/data/real_gray_online_replacement_40videos_assets328_20260922`

Launcher:

`scripts/anti_uav/run_gray_probability_ablation.py`

Example (use a NEW run directory and an idle GPU):

```bash
.venv/bin/python scripts/anti_uav/run_gray_probability_ablation.py \
  --reference-run runs/anti_uav/approved40_online328_20260922 \
  --run-dir runs/anti_uav/NEW_PROBABILITY_EXPERIMENT \
  --probability 0.15 --device 1 --detach
```

Each new run records hashes of the training list, validation list, negative pool
and initial checkpoint. It refuses overwrites and checks the Python/PyTorch
versions against the original training log. Separate GPU locks prevent this
launcher from accidentally sharing a GPU with another instance; current GPU
memory and utilization are also checked before starting.

Authoritative files, created as stages complete:

- `experiment_status.json`: orchestration status or error.
- `status.json`: actual training stage and last completed epoch.
- `training.log`: training progress/losses; `console.log`: parent process log.
- `training_p3/p3/results.csv` and `training_addon/p2/results.csv`: epoch metrics.
- `training_p3/p3/weights/best.pt` and `training_addon/p2/weights/best.pt`: candidates.
- `frozen_checks.json`: saved best/last P3 raw-output preservation checks.
- `validation_results.json` and `COMPARISON.md`: completed native validation comparison.
- `branch_calibration/`: automatic post-training branch diagnosis and calibration.

The two unsuffixed `approved40_prob00_20260922` / `approved40_prob15_20260922`
directories are failed PRE-FLIGHT attempts, not training results. The original
path guard incorrectly matched protocol directory names containing
`holdout_Video00004`; exact video-directory matching now replaces this substring
check. Those attempts never started training and are retained for traceability.

## Branch Calibration

Implementation: `scripts/anti_uav/calibrate_addon_branches.py`.

The tool uses the same fixed-shape validation loader and preserves the original
NMS output. It caches pre-NMS candidates with legacy/P2 provenance, scans 64
threshold pairs, and selects minimum FP subject to no aggregate or populated
scale-bucket TP loss. It then reruns real model inference with the selected
pre-NMS gates and asserts that TP/FP/FN and scale recalls equal cached replay.
Accepted box scores and coordinates are not rescaled. Matching is one-to-one,
supports multiple GT boxes and uses the repository's IoU=0.5 policy.

Current candidate for the completed 40-video/328-cutout P3+P2 checkpoint:

- Legacy P3/P4/P5 threshold: 0.075.
- Add-on P2 threshold: 0.03.
- Gates apply BEFORE common NMS, IoU 0.45, max_det 100.
- This is NOT a proposed 0.075 threshold for the standalone P3 model.
- Model SHA256: `6aec79be8e1562d5cbab4f7b1427d156edbd4fff8dba3095302b17b85aac6472`.

Completed same-runtime validation result, verified by a second real forward
pass (not just cached replay), 1,421 native frames / 797 GT:

| Metric | Uniform 0.03 baseline | Branch-gated candidate |
| --- | ---: | ---: |
| TP | 605 | 605 |
| FP | 309 | 290 |
| FN | 192 | 192 |
| Precision | 66.19% | 67.60% |
| Recall | 75.91% | 75.91% |
| 4-8px Recall | 77.81% | 77.81% |

FP drops by 19 (6.15%) on the selection data. A separate per-frame replay also
found zero originally matched GT lost and zero new GT gained. This is not an
independent-test gain, statistical significance claim, or proof of large-target
performance: this native validation subset has no GT occupying 5% or more of
the image. The 64 synthetic zoom views remain outside native metrics.

Existing-model diagnosis output on the server:

`runs/anti_uav/approved40_prob_ablation_20260922/online328_branch_calibration_verified/`

Only a `status.json` with `stage=complete` and
`selected_policy_verified_by_real_forward=true` confirms the final real-forward
check. `candidate_policy.json` includes the model hash and input/NMS contract;
it is an experimental FP32 policy, not an installed board configuration.
This check has passed. All original baseline metrics exactly match the previous
completed experiment. The final summary, raw metric JSON, policy and completion
receipt are also downloaded beside this README. The relevant 45 tests passed
both locally and in the server's matching training environment.

An earlier exploratory `online328_branch_calibration` used PyTorch 2.9.1 and
produced one extra TP. It is not the comparison basis. The `_torch251` run
restored the original 605/309 baseline, and the `_verified` run adds the live
forward-policy check. Do not mix environments when comparing tiny-object boxes.

## Next Gate

The current changes intentionally do NOT mix hard-sample reweighting or crop
augmentation into the probability ablation. After these arms finish, compare
validation precision/recall, FP per 1,000 frames, localization and scale buckets.
Then add training-only reviewed hard examples as a separate controlled arm,
followed by a separate crop-and-resize arm. Retain original tiny-object and
large-object exposures; never mine Video00004/Video00009 into training.

No new training gain is claimed until completed results exist. Final candidates
still require frozen-threshold independent evaluation, RKNN INT8 accuracy checks
and board FPS/latency measurements before replacing deployment artifacts.

Code sync follows local Git commit/push, then server fast-forward pull. When the
server cannot reach GitHub SSH, it pulls a bundle containing the pushed commits.
