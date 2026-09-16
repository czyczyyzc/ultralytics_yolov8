# Native Full-Pool Data Expansion: 2026-09-16

## Scope

Use the frozen 23-approved-task snapshot, adding all 14 previously unused videos
to the 14-video native baseline. Keep Anti-UAV300, all prior training slots and
positive repetitions, and existing hard-negative substitutions unchanged.
No fixed-480, online zoom, copy-paste or new crop-and-resize recipe is enabled.

Source dataset:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_native_augmentation_restored_20260916`

New dataset:
`/mnt/andrew/anti_uav_model_refinement/data/real_gray_native_fullpool_28videos_20260916`

Snapshot: `deliverables/data_inventory_20260916/new_approved_audit.json` in the repo.
Later uploads are not silently included in this experiment.

## Label-Based Sampling

The operator rejected fixed positive/negative frame strides. Extraction is now
stride 1 for every included reviewed frame, using full original video resolution.
New pool: 31,605 positive + 25,540 reviewed negative = 57,145 eligible frames.
The 101 uncertain frames and zero unreviewed frames are excluded. An uncertain
frame is never relabeled as background. Labels and videos are checksum verified,
frame dimensions/counts decoded, and label presence checked against the manifest.

Every epoch keeps all 89,748 original slots and all 31,605 new positive frames.
Add 5,754 new negative frames per epoch, cycling without replacement within each
epoch through the full shuffled negative pool. Every new negative is covered
within five epochs. Pool order and epoch shuffle are deterministic by seed.

Expected per epoch: 108,041 positive entries + 19,066 negative entries = 127,107,
approximately 85% positive and 15% negative. This is a sampling proportion, not
a probability-valued label or detector confidence. Each new positive has inclusion
probability 1 per epoch; each new negative has marginal inclusion probability
5,754/25,540, approximately 22.53%, over shuffled pool order. An epoch still keeps
all 13,312 prior negative slots. The baseline's unchanged negative share is 14.83%.

Candidate list: 146,893 entries including original repetitions; the label-aware
sampler, not the generic YOLO loader, selects the 127,107-entry epoch schedule.
Expected distinct image paths: 119,134, of which 83,398 are gray-source paths.
The 4-8px original target exposure is not removed. More data can still change
relative exposure and optimization dynamics; no improvement is assumed.

## Controlled Experiment

Run root on server 47:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916`

Runner: `scripts/anti_uav/run_native_pool_comparison.py`.
GPU: physical 6, preserving the existing small Triton service.
Sequence: verify/extract all new data, train `expanded_28`, train `baseline_14`,
then evaluate both arms and the old deployment reference.

Each arm: P3 15 epochs, freeze its selected P3, then add-on P2 15 epochs.
Same initial P3, seed 20260915, batch 64, nominal effective batch 128, AdamW,
P3 lr0=0.0001 and add-on lr0=0.001. Frozen P3 outputs are checked bit-exact.
Base augmentation: scale=.2, translate=.05, fliplr=.5, hsv_h=.015,
hsv_s=.4, hsv_v=.3, with Mosaic/MixUp/CopyPaste disabled as before.

Both arms use a finite epoch loader to avoid prefetch crossing a negative-pool
boundary. Validation is asserted to be 544x960, rather than auto-rectangular.
Selection uses native-gray F2(conf=.03), AP50 and AP50-95. Same whole stationary
video00009 validation; Video00004 is final test only. The 64 synthetic stress
views remain diagnostic only, never training or checkpoint-selection inputs.

Outputs: `expanded_28/training_p3/p3/weights/best.pt`,
`expanded_28/training_addon/p2/weights/best.pt`, and the corresponding
`baseline_14/...` paths under the run root, only after each stage completes.
Final results: `COMPARISON.md`, `comparison.json`, `evaluation/`.
Progress: parent `status.json`, `logs/`, and each arm's `status.json`/results CSVs.

Evaluation: actual 960x544 FP32 PT detector, conf .01/.03/.05, NMS IoU .45,
match IoU .5, max_det 100. Report P/R, TP/FP/FN, mAP and 4-8px recall.
This does not include RKNN INT8 conversion, RK-BoT-SORT metrics or board FPS.
Matched epochs mean the larger dataset gets more optimizer steps; this is not
a matched-compute experiment or a multi-seed causal proof. Whole-video hashes
exclude exact duplicates, not all re-encoded or overlapping same-session clips.
