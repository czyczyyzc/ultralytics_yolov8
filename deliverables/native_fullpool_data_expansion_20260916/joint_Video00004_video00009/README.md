# Joint Video00004 + Stationary Video00009 Evaluation

All six detectors were rerun on one concatenated native-frame list on server 47
on 2026-09-17. These are PT FP32 detector results, not RKNN INT8, tracking or FPS.
No training was started and no deployed weights were changed.

## Evaluation Scope

- Video00004: all 2,359 previously evaluated frames, 448 GT boxes.
- Stationary video00009: the same 1,421 validation frames sampled every 10 frames,
  with 797 GT boxes. This is NOT evaluation of every raw frame of video00009.
- Total: 3,780 unique image paths, 1,245 GT boxes and 2,535 background frames.
  GT boxes are accumulated across frames, not 1,245 distinct physical drones.
- Fixed input 960x544, FP32, batch 32, NMS IoU 0.45, matching IoU 0.5,
  max detections 100, inference confidence floor 0.001.
- No synthetic zoom/sticker frames are included.
- Video00009 was used to select checkpoints. The union is a descriptive joint
  evaluation pool, NOT a newly independent or untouched test set.

Precision = sum(TP)/(sum(TP)+sum(FP)); recall = sum(TP)/(sum(TP)+sum(FN)).
F1 = 2*sum(TP)/(2*sum(TP)+sum(FP)+sum(FN)). Per-video percentages are not averaged.
mAP50 and mAP50-95 were recalculated over the combined confidence-ranked predictions,
not averaged or weighted from per-video AP. AP is independent of the displayed
operating threshold; its inference confidence floor is 0.001.

## Joint Results At Confidence 0.03

| Model | TP | FP | FN | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous deployment P3 | 650 | 653 | 595 | 49.88% | 52.21% | 51.02% | 50.09% | 28.24% |
| Baseline-14 P3 | 647 | 497 | 598 | 56.56% | 51.97% | 54.16% | 50.81% | 28.57% |
| Expanded-28 P3 | 740 | 459 | 505 | 61.72% | 59.44% | 60.56% | 60.46% | 33.66% |
| Previous deployment Frozen-P3 + Add-on P2 | 811 | 915 | 434 | 46.99% | 65.14% | 54.59% | 58.36% | 32.69% |
| Baseline-14 Frozen-P3 + Add-on P2 | 810 | 1134 | 435 | 41.67% | 65.06% | 50.80% | 55.12% | 30.87% |
| Expanded-28 Frozen-P3 + Add-on P2 | 984 | 772 | 261 | 56.04% | 79.04% | 65.58% | 73.91% | 40.29% |

For the combined detector, expanded-28 versus baseline-14 yields 174 more TP,
362 fewer FP (31.92% reduction), 174 fewer FN, precision +14.37 percentage points,
recall +13.98 points, F1 +14.78 points and mAP50 +18.78 points.

Input-scale 4-8px long-edge recall over 797 GT boxes is 71.77% for the previous
deployment combined model, 71.14% for baseline-14 and 84.69% for expanded-28.
These are GT-weighted pooled recalls, not arithmetic means of the two videos.

Pooled improvements must not hide individual-video regressions. In particular,
standalone expanded-28 P3 regressed on Video00004 mAP50; see the per-video report
in the parent directory. The combined-detector result is stronger across both videos.

## Combined Detector Confidence Sweep

| Model | Conf | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous deployment | 0.01 | 830 | 1481 | 415 | 35.92% | 66.67% | 46.68% |
| Baseline-14 | 0.01 | 824 | 1691 | 421 | 32.76% | 66.18% | 43.83% |
| Expanded-28 | 0.01 | 986 | 1161 | 259 | 45.92% | 79.20% | 58.14% |
| Previous deployment | 0.03 | 811 | 915 | 434 | 46.99% | 65.14% | 54.59% |
| Baseline-14 | 0.03 | 810 | 1134 | 435 | 41.67% | 65.06% | 50.80% |
| Expanded-28 | 0.03 | 984 | 772 | 261 | 56.04% | 79.04% | 65.58% |
| Previous deployment | 0.05 | 797 | 750 | 448 | 51.52% | 64.02% | 57.09% |
| Baseline-14 | 0.05 | 806 | 940 | 439 | 46.16% | 64.74% | 53.90% |
| Expanded-28 | 0.05 | 980 | 672 | 265 | 59.32% | 78.71% | 67.66% |

## Verification And Artifacts

All 72 checks (6 models x 3 thresholds x TP/FP/FN/frames) exactly match the sums
from the previous independent per-video evaluations. Each model/threshold has
TP+FN=1,245 and 3,780 evaluated frames. Full precision metrics, source weight
paths and per-count differences are in `results.json`; settings and the joint-list
SHA256 are in `protocol.json`.

Server run directory:

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/joint_Video00004_video00009_20260917
```

The two training arms matched initialization, seed, augmentation and epochs,
not optimizer steps. Previous deployment weights are a practical reference,
not the controlled data-only baseline. Generated sticker data was not used.
