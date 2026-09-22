# Pure P3: Video00004 + Video00009

FP32 detector only; no P2, tracking, branch calibration or RKNN quantization.
Fixed best.pt, 960x544 input, NMS IoU .45, matching IoU .5, max_det 100.
P/R/counts use conf .03; AP is ranked from conf floor .001.
Video00004: 2359 test frames / 448 GT. Video00009: 1421 selection-validation frames / 797 GT.
The 64 synthetic zoom views are excluded. The pooled set contains 3780 frames / 1245 GT.
The pooled set is NOT an independent test: Video00009 selected checkpoints.
No checkpoint or threshold is selected using this evaluation.
Pooled counts are summed and ratios recomputed; pooled AP is recomputed from all native predictions.

## Video00004_test

| Model | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | 57.51% | 75.22% | 337 | 249 | 111 | 69.35% | 43.44% | 72.59% |
| gray40_assets53_prob50_p3 | 45.14% | 75.67% | 339 | 412 | 109 | 70.20% | 41.34% | 73.09% |
| gray40_assets328_prob50_p3 | 45.27% | 75.89% | 340 | 411 | 108 | 70.93% | 41.12% | 73.33% |
| gray40_assets328_prob15_p3 | 43.88% | 76.79% | 344 | 440 | 104 | 69.98% | 41.64% | 74.32% |
| gray40_prob00_p3 | 53.59% | 76.56% | 343 | 297 | 105 | 75.07% | 45.39% | 74.07% |

## Video00009_selection_subset

| Model | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | 65.74% | 50.56% | 403 | 210 | 394 | 56.00% | 28.01% | 44.90% |
| gray40_assets53_prob50_p3 | 66.88% | 51.94% | 414 | 205 | 383 | 57.68% | 30.59% | 46.43% |
| gray40_assets328_prob50_p3 | 68.75% | 53.83% | 429 | 195 | 368 | 60.31% | 32.01% | 49.49% |
| gray40_assets328_prob15_p3 | 67.14% | 53.58% | 427 | 209 | 370 | 58.54% | 30.85% | 49.23% |
| gray40_prob00_p3 | 72.08% | 53.45% | 426 | 165 | 371 | 60.64% | 31.86% | 49.49% |

## pooled_native

| Model | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | 61.72% | 59.44% | 740 | 459 | 505 | 60.46% | 33.66% | 58.97% |
| gray40_assets53_prob50_p3 | 54.96% | 60.48% | 753 | 617 | 492 | 61.58% | 34.51% | 59.97% |
| gray40_assets328_prob50_p3 | 55.93% | 61.77% | 769 | 606 | 476 | 63.64% | 35.27% | 61.61% |
| gray40_assets328_prob15_p3 | 54.30% | 61.93% | 771 | 649 | 474 | 62.12% | 34.83% | 61.98% |
| gray40_prob00_p3 | 62.47% | 61.77% | 769 | 462 | 476 | 65.51% | 36.67% | 61.98% |
