# Frozen-P3 + Add-on P2 Strict Comparison

Date: 2026-09-03

## Conclusion

Frozen-P3 + Add-on P2 is the strongest of the three tested detectors on the strictly held-out `Video00004`.
It preserves every P3/P4/P5 raw output tensor bit-for-bit while adding a separately trained stride-4 branch.

- Standard mAP50 improves from 77.14% (P3) and 79.07% (full P2) to 91.31%.
- Standard mAP50-95 improves from 43.85% (P3) and 44.13% (full P2) to 52.18%.
- At the same `conf=0.03`, recall is 100.00%, but precision is only 45.07% because this threshold is too low for the new branch.
- The recommended PT operating point from the measured sweep is `conf=0.45`: 88.79% precision, 93.75% recall and 91.21% F1.
- At `conf=0.45`, recall for 4-6 px targets is 92.06%, versus 76.19% for P3 and 88.10% for full P2 at `conf=0.03`.

Do not retain the previous deployment threshold of `0.01` for this checkpoint. The confidence distribution changed materially.

## Strict Protocol

- Input: `960x544` (W x H), `rect=False`, no batch-dependent rectangular padding.
- Test video: `Video00004`, 2,359 complete frames, including 448 positive and 1,911 target-absent frames.
- Match criterion: IoU >= 0.50; NMS IoU = 0.45.
- Training data: the same 83,118-image strict dataset used by the existing strict P3/full-P2 experiments: 70,650 positive slots and 12,468 gray negatives.
- Leakage control: `Video00004` is absent from training and model selection. The new INT8 calibration list also excludes all 64 old `Video00004` calibration images and contains 384 images from the other gray videos.
- Model selection: maximum `0.1*mAP50 + 0.9*mAP50-95` on Anti-UAV300 RGB validation only. `Video00004` was not used to choose a stage or epoch.
- Frozen regression: P3/P4/P5 raw output maximum absolute errors are `[0.0, 0.0, 0.0]` for both `best.pt` and `last.pt` after each training stage.

This is a strict same-data/same-test evaluation of the three requested current models. It is not a multi-seed statistical study: P3 and full P2 used seed `20260902`; Add-on P2 used the recorded deterministic seed `20260903`.

## Architecture and Training

The add-on consumes the frozen standard neck P3 feature, upsamples it by 2, concatenates it with the frozen backbone P2 feature, and applies a new C2f P2 adapter plus new P2 box/class towers. The add-on never feeds back into P3, P4 or P5.

Only 87,713 parameters were trainable. Backbone, standard neck, P3/P4/P5 detection towers, BatchNorm buffers and their EMA copies were frozen. The selected checkpoint is Stage 1 epoch 10; the lower-learning-rate Stage 2 did not improve the RGB validation fitness.

| Model | Parameters | Training schedule | Selected RGB mAP50-95 |
|---|---:|---|---:|
| Current P3 | 3,011,043 | Existing strict 15-epoch model | 68.53% peak in its run |
| Full retrain P2 | 2,926,692 | 15 + 15 epochs, all model paths trainable | 66.75% |
| Frozen-P3 + Add-on P2 | 3,098,756 | 15 + 15 epochs, 87,713 add-on parameters trainable | 67.80% |

## Standard Metrics

Ultralytics standard validation metrics integrate the precision-recall curve and are not tied to the fixed deployment threshold below.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Current P3 | 78.48% | 71.21% | 77.14% | 43.85% |
| Full retrain P2 | 83.40% | 76.79% | 79.07% | 44.13% |
| Frozen-P3 + Add-on P2 | **88.81%** | **93.90%** | **91.31%** | **52.18%** |

## Same Threshold

All models use `conf=0.03` here, matching the previous strict visualization.

| Model | TP | FP | FN | Precision | Recall | F1 | Absent-frame FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current P3 | 365 | 247 | 83 | 59.64% | 81.47% | 68.87% | 10.15% |
| Full retrain P2 | 381 | 117 | 67 | 76.51% | 85.04% | 80.55% | 4.13% |
| Frozen-P3 + Add-on P2 | **448** | 546 | **0** | 45.07% | **100.00%** | 62.14% | 19.57% |

The same-threshold table proves the add-on recovers all labeled positive frames, but also shows that `0.03` is not a sensible deployment threshold for this model.

## Recommended Threshold

| Model | Best tested conf | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Current P3 | 0.15 | 78.05% | 71.43% | 74.59% |
| Full retrain P2 | 0.05 | 80.58% | 80.58% | 80.58% |
| Frozen-P3 + Add-on P2 | **0.45** | **88.79%** | **93.75%** | **91.21%** |

At `conf=0.45`, Add-on P2 has 420 TP, 53 FP and 28 FN. All 28 misses are low-confidence cases; no localization-only or no-candidate misses were observed down to the `0.001` probe threshold.

| Input target max side | Frames | Add-on hits at conf 0.45 | Recall |
|---|---:|---:|---:|
| 4-6 px | 126 | 116 | 92.06% |
| 6-8 px | 279 | 261 | 93.55% |
| 8-12 px | 17 | 17 | 100.00% |
| >=12 px | 26 | 26 | 100.00% |

## Speed

Pure PyTorch forward speed was measured on the same NVIDIA A100-SXM4-80GB, batch 1, FP32, fixed `1x3x544x960` tensor, 100 warmups and five repeats of 500 synchronized iterations.

| Model | Median latency | Median FPS |
|---|---:|---:|
| Current P3 | 5.278 ms | 189.47 |
| Full retrain P2 | 6.539 ms | 152.93 |
| Frozen-P3 + Add-on P2 | 6.167 ms | 162.16 |

The add-on costs 16.8% latency versus P3, but is 5.7% lower latency than the full P2 graph in this controlled server benchmark.

The RK3588S was not reachable at `192.168.77.50` during finalization, and all local wired adapters reported inactive. Therefore no new board FPS is claimed here. The exported model is ready for the existing 12-output, four-scale C++/RKNN path; board-side FPS must be measured after the board reconnects.

RKNN Toolkit2 2.3.2 simulator validation completed on target-positive source frames 600-619 at `conf=0.45`. PT produced 10 TP / 10 FN, while INT8 produced 11 TP / 9 FN / 0 FP. This small clip confirms that the graph loads, returns 12 tensors, decodes correctly and does not show an INT8 recall loss on those frames. It is not a replacement for a full INT8 threshold sweep. Simulator median latency was 450.57 ms on x86 and must not be interpreted as RK3588S performance.

The simulator rebuild SHA differs from the delivered RKNN SHA because Toolkit2 builds are not byte-deterministic. Both artifacts were built from the same ONNX and 384-image calibration manifest; the mismatch is recorded in the simulator JSON rather than hidden.

## Artifacts

Selected PT checkpoint:

`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_frozen_p3_addon_p2_strict_holdout_Video00004_20260903/final_selected/best.pt`

PT SHA256: `56cf20d45531e47b59c44fcdce8e7fa287bb300d7d4b839607f86c927f36136b`

RKNN Toolkit2 2.3.2 INT8 model:

`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_frozen_p3_addon_p2_strict_holdout_Video00004_20260903/rknn_int8/frozen_p3_addon_p2_960x544_v232_int8.rknn`

RKNN SHA256: `3e330b4cc5488d2b0fbf67c7ac73480b0f07d310c58ffa11778297659d418abc`

RK-optimized ONNX:

`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_frozen_p3_addon_p2_strict_holdout_Video00004_20260903/rknn_int8/frozen_p3_addon_p2_960x544_rkopt.onnx`

Visualization:

`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/frozen_p3_addon_p2_strict_comparison_20260903/visualization/Video00004_P3_vs_fullP2_vs_addonP2_960x544_conf003.mp4`

The visualization is 2,880x540, 100 FPS and 2,359 frames. It uses synchronized source frames, unannotated native-image crops enlarged before drawing, and 1 px corner boxes with no crosshair.

Local deliverable root:

`/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/frozen_p3_addon_p2_strict_comparison_20260903/`

Implementation commit at result finalization: `93e5448`.

## Residual Risk

The result is strong but comes from one held-out gray video. It should be confirmed on the remaining LOVO folds before treating the gain as a general gray-domain result. The PT threshold `0.45` also needs a full INT8 threshold sweep on the board because quantization can shift confidence values outside the tested 20-frame clip.
