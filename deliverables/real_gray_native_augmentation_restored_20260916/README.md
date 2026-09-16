# Native Augmentation Restored: 2026-09-16

At the user's request, discontinue the extra fixed-480 and online-scale recipes.
The online process group 1292464 was stopped with SIGINT; all archived data,
checkpoints and logs remain intact. Neither experiment replaces deployment weights.

Restoration means the original augmentation recipe, NOT reverting new videos or
negative mining: keep the 89,748 native schedule entries, positive repetitions,
461 hard-negative replacements, Anti-UAV300 and all 14 gray training videos.
Remove extra static/dynamic crop slots only. Keep video00009 as validation and
Video00004 as test-only. The unchanged synthetic validation views are diagnostic,
not training data or checkpoint-selection inputs.

## Restored Configuration

Server dataset:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_native_augmentation_restored_20260916`

Training YAML: `train_hardneg_gray_monitor.yaml` under that directory.
There is no `online_scale` key, no `zoom_train` entry, and no area-based removal
of original labels. `manifest.json` records schedule hashes and retained counts.

Verified original P3 and add-on training augmentation parameters:
`scale=0.2`, `translate=0.05`, `fliplr=0.5`, `hsv_h=0.015`, `hsv_s=0.4`,
`hsv_v=0.3`, `degrees=0`, `shear=0`, `perspective=0`, `flipud=0`,
`mosaic=0`, `mixup=0`, `copy_paste=0`. These are already the existing
`run_rebalanced_fullscale_training.py` runner's base transforms; turning off
the opt-in extra dataset wrapper restores them without editing archived runs.

Preparation script: `scripts/anti_uav/restore_native_gray_augmentation.py`.
This preparation does not start a new training run. Do not resume old experimental
optimizer states with changed data and label the result a clean baseline.
Before launching a new comparison, resolve the training-validation shape caveat
documented in `real_gray_all398_online_scale_20260916/RESULTS_20260916.md`.

## Direct Crop-and-Resize Alternative

Both discontinued recipes already crop original image context and resize it;
they do not paste a target onto another background. A simpler recipe would use
fixed-aspect crops around a training GT, mild center jitter, direct resize to
960x544, and transformed boxes, without forced area-bin rejection sampling or
additional noise/blur. Preserve whole-frame native samples and cap extra views.
This is a proposal, not an enabled replacement or a demonstrated improvement.

For example, a native 480x272 crop resized to 960x544 doubles an 8px target to
16px at network input; it would have been about 4px in a 1920-wide full frame
resized to 960. It magnifies available pixels, not missing optical detail.
GT-centered training crops are not the same as inference ROIs: inference needs
a predicted/previous track location and a full-frame reacquisition policy.
Very large targets require suitable native source images or explicit, correctly
labeled truncation; this simple operation cannot guarantee >=80% real-target recall.
