# Frozen P3 + Add-on P2: synthetic appearance evaluation

960x544 FP32; conf .03 for counts, matching IoU .5, NMS .45, max_det 100; AP floor .001.
Five fixed best.pt files; no training, checkpoint selection, threshold tuning, tracker or RKNN.
Full sets: 3780 frames / 1245 GT. Replaced pairs: 981 positive frames only.
The 328 source assets were used in training; Video00009 selected checkpoints. Not an independent or unseen-type test.
Original-size 4-8px group membership is frozen before replacement; do not compare shifting bbox-size denominators.

## combined / all

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_addon_p2 | original | 56.04% | 79.04% | 984 | 772 | 261 | 73.91% | 40.29% |
| old28_addon_p2 | synthetic | 53.71% | 73.90% | 920 | 793 | 325 | 63.15% | 27.74% |
| gray40_assets53_prob50_addon_p2 | original | 49.26% | 83.21% | 1036 | 1067 | 209 | 76.60% | 41.52% |
| gray40_assets53_prob50_addon_p2 | synthetic | 50.64% | 85.46% | 1064 | 1037 | 181 | 78.41% | 43.37% |
| gray40_assets328_prob50_addon_p2 | original | 49.30% | 84.42% | 1051 | 1081 | 194 | 79.02% | 42.34% |
| gray40_assets328_prob50_addon_p2 | synthetic | 50.97% | 86.67% | 1079 | 1038 | 166 | 81.68% | 46.78% |
| gray40_assets328_prob15_addon_p2 | original | 45.26% | 82.89% | 1032 | 1248 | 213 | 77.14% | 41.10% |
| gray40_assets328_prob15_addon_p2 | synthetic | 45.17% | 83.05% | 1034 | 1255 | 211 | 74.46% | 38.72% |
| gray40_prob00_addon_p2 | original | 51.73% | 82.81% | 1031 | 962 | 214 | 78.51% | 42.27% |
| gray40_prob00_addon_p2 | synthetic | 49.08% | 77.03% | 959 | 995 | 286 | 66.34% | 29.09% |

## combined / replaced

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_addon_p2 | original | 78.12% | 84.81% | 832 | 233 | 149 | 84.37% | 45.89% |
| old28_addon_p2 | synthetic | 75.15% | 78.29% | 768 | 254 | 213 | 76.53% | 31.49% |
| gray40_assets53_prob50_addon_p2 | original | 73.41% | 88.38% | 867 | 314 | 114 | 88.11% | 47.78% |
| gray40_assets53_prob50_addon_p2 | synthetic | 75.91% | 91.23% | 895 | 284 | 86 | 91.82% | 51.08% |
| gray40_assets328_prob50_addon_p2 | original | 76.47% | 88.79% | 871 | 268 | 110 | 88.73% | 47.93% |
| gray40_assets328_prob50_addon_p2 | synthetic | 79.98% | 91.64% | 899 | 225 | 82 | 92.24% | 54.12% |
| gray40_assets328_prob15_addon_p2 | original | 77.39% | 88.28% | 866 | 253 | 115 | 88.21% | 47.26% |
| gray40_assets328_prob15_addon_p2 | synthetic | 76.95% | 88.48% | 868 | 260 | 113 | 86.77% | 45.09% |
| gray40_prob00_addon_p2 | original | 82.46% | 88.69% | 870 | 185 | 111 | 88.54% | 48.07% |
| gray40_prob00_addon_p2 | synthetic | 78.54% | 81.35% | 798 | 218 | 183 | 78.58% | 32.69% |

## combined / replaced_original_4to8px

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_addon_p2 | original | 79.92% | 87.90% | 625 | 157 | 86 | 86.15% | 43.71% |
| old28_addon_p2 | synthetic | 75.89% | 81.01% | 576 | 183 | 135 | 77.97% | 29.54% |
| gray40_assets53_prob50_addon_p2 | original | 71.13% | 90.44% | 643 | 261 | 68 | 89.09% | 44.24% |
| gray40_assets53_prob50_addon_p2 | synthetic | 72.98% | 92.69% | 659 | 244 | 52 | 92.32% | 45.86% |
| gray40_assets328_prob50_addon_p2 | original | 75.26% | 91.14% | 648 | 213 | 63 | 90.14% | 44.45% |
| gray40_assets328_prob50_addon_p2 | synthetic | 78.04% | 92.97% | 661 | 186 | 50 | 92.67% | 49.63% |
| gray40_assets328_prob15_addon_p2 | original | 76.90% | 90.86% | 646 | 194 | 65 | 89.71% | 43.88% |
| gray40_assets328_prob15_addon_p2 | synthetic | 75.97% | 90.72% | 645 | 204 | 66 | 87.51% | 39.44% |
| gray40_prob00_addon_p2 | original | 83.68% | 91.56% | 651 | 127 | 60 | 90.83% | 45.15% |
| gray40_prob00_addon_p2 | synthetic | 79.18% | 83.97% | 597 | 157 | 114 | 79.62% | 29.86% |

## Video00004 / all

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_addon_p2 | original | 51.67% | 100.00% | 448 | 419 | 0 | 95.81% | 60.96% |
| old28_addon_p2 | synthetic | 47.70% | 92.41% | 414 | 454 | 34 | 78.55% | 34.81% |
| gray40_assets53_prob50_addon_p2 | original | 37.61% | 99.55% | 446 | 740 | 2 | 94.67% | 56.28% |
| gray40_assets53_prob50_addon_p2 | synthetic | 36.44% | 96.88% | 434 | 757 | 14 | 86.61% | 46.88% |
| gray40_assets328_prob50_addon_p2 | original | 36.62% | 99.55% | 446 | 772 | 2 | 95.19% | 55.82% |
| gray40_assets328_prob50_addon_p2 | synthetic | 36.29% | 97.77% | 438 | 769 | 10 | 90.53% | 51.61% |
| gray40_assets328_prob15_addon_p2 | original | 32.53% | 99.55% | 446 | 925 | 2 | 93.81% | 56.22% |
| gray40_assets328_prob15_addon_p2 | synthetic | 30.92% | 95.31% | 427 | 954 | 21 | 81.93% | 40.75% |
| gray40_prob00_addon_p2 | original | 39.79% | 100.00% | 448 | 678 | 0 | 94.80% | 57.94% |
| gray40_prob00_addon_p2 | synthetic | 35.46% | 88.17% | 395 | 719 | 53 | 74.42% | 31.04% |

## Video00004 / replaced

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_addon_p2 | original | 85.25% | 100.00% | 416 | 72 | 0 | 99.49% | 62.97% |
| old28_addon_p2 | synthetic | 78.12% | 91.83% | 382 | 107 | 34 | 88.38% | 37.47% |
| gray40_assets53_prob50_addon_p2 | original | 69.23% | 99.52% | 414 | 184 | 2 | 99.39% | 58.51% |
| gray40_assets53_prob50_addon_p2 | synthetic | 66.67% | 96.63% | 402 | 201 | 14 | 95.97% | 50.56% |
| gray40_assets328_prob50_addon_p2 | original | 74.86% | 99.52% | 414 | 139 | 2 | 99.36% | 57.71% |
| gray40_assets328_prob50_addon_p2 | synthetic | 74.91% | 97.60% | 406 | 136 | 10 | 97.26% | 54.46% |
| gray40_assets328_prob15_addon_p2 | original | 77.24% | 99.52% | 414 | 122 | 2 | 99.40% | 59.22% |
| gray40_assets328_prob15_addon_p2 | synthetic | 72.34% | 94.95% | 395 | 151 | 21 | 91.75% | 43.99% |
| gray40_prob00_addon_p2 | original | 86.85% | 100.00% | 416 | 63 | 0 | 99.37% | 60.72% |
| gray40_prob00_addon_p2 | synthetic | 77.73% | 87.26% | 363 | 104 | 53 | 83.34% | 33.74% |

## Video00004 / replaced_original_4to8px

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_addon_p2 | original | 85.93% | 100.00% | 391 | 64 | 0 | 99.50% | 62.60% |
| old28_addon_p2 | synthetic | 78.73% | 91.82% | 359 | 97 | 32 | 88.48% | 36.84% |
| gray40_assets53_prob50_addon_p2 | original | 68.85% | 99.49% | 389 | 176 | 2 | 99.40% | 57.84% |
| gray40_assets53_prob50_addon_p2 | synthetic | 66.26% | 96.42% | 377 | 192 | 14 | 95.46% | 48.38% |
| gray40_assets328_prob50_addon_p2 | original | 74.66% | 99.49% | 389 | 132 | 2 | 99.34% | 57.44% |
| gray40_assets328_prob50_addon_p2 | synthetic | 74.71% | 97.44% | 381 | 129 | 10 | 96.86% | 52.41% |
| gray40_assets328_prob15_addon_p2 | original | 77.03% | 99.49% | 389 | 116 | 2 | 99.39% | 58.66% |
| gray40_assets328_prob15_addon_p2 | synthetic | 71.98% | 94.63% | 370 | 144 | 21 | 91.00% | 41.77% |
| gray40_prob00_addon_p2 | original | 87.08% | 100.00% | 391 | 58 | 0 | 99.40% | 60.19% |
| gray40_prob00_addon_p2 | synthetic | 77.68% | 87.21% | 341 | 98 | 50 | 83.13% | 32.74% |

## Video00009 / all

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_addon_p2 | original | 60.29% | 67.25% | 536 | 353 | 261 | 61.55% | 28.19% |
| old28_addon_p2 | synthetic | 59.88% | 63.49% | 506 | 339 | 291 | 56.13% | 23.75% |
| gray40_assets53_prob50_addon_p2 | original | 64.34% | 74.03% | 590 | 327 | 207 | 68.93% | 33.48% |
| gray40_assets53_prob50_addon_p2 | synthetic | 69.23% | 79.05% | 630 | 280 | 167 | 77.20% | 42.61% |
| gray40_assets328_prob50_addon_p2 | original | 66.19% | 75.91% | 605 | 309 | 192 | 71.95% | 34.99% |
| gray40_assets328_prob50_addon_p2 | synthetic | 70.44% | 80.43% | 641 | 269 | 156 | 79.08% | 45.04% |
| gray40_assets328_prob15_addon_p2 | original | 64.47% | 73.53% | 586 | 323 | 211 | 69.61% | 32.96% |
| gray40_assets328_prob15_addon_p2 | synthetic | 66.85% | 76.16% | 607 | 301 | 190 | 73.18% | 38.77% |
| gray40_prob00_addon_p2 | original | 67.24% | 73.15% | 583 | 284 | 214 | 69.90% | 33.60% |
| gray40_prob00_addon_p2 | synthetic | 67.14% | 70.77% | 564 | 276 | 233 | 64.36% | 28.99% |

## Video00009 / replaced

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_addon_p2 | original | 72.10% | 73.63% | 416 | 161 | 149 | 71.20% | 32.45% |
| old28_addon_p2 | synthetic | 72.42% | 68.32% | 386 | 147 | 179 | 66.94% | 27.05% |
| gray40_assets53_prob50_addon_p2 | original | 77.70% | 80.18% | 453 | 130 | 112 | 79.06% | 38.97% |
| gray40_assets53_prob50_addon_p2 | synthetic | 85.59% | 87.26% | 493 | 83 | 72 | 89.43% | 51.65% |
| gray40_assets328_prob50_addon_p2 | original | 77.99% | 80.88% | 457 | 129 | 108 | 80.04% | 40.03% |
| gray40_assets328_prob50_addon_p2 | synthetic | 84.71% | 87.26% | 493 | 89 | 72 | 89.12% | 53.98% |
| gray40_assets328_prob15_addon_p2 | original | 77.53% | 80.00% | 452 | 131 | 113 | 79.09% | 37.90% |
| gray40_assets328_prob15_addon_p2 | synthetic | 81.27% | 83.72% | 473 | 109 | 92 | 84.08% | 46.21% |
| gray40_prob00_addon_p2 | original | 78.82% | 80.35% | 454 | 122 | 111 | 79.31% | 38.10% |
| gray40_prob00_addon_p2 | synthetic | 79.23% | 76.99% | 435 | 114 | 130 | 74.88% | 32.55% |

## Video00009 / replaced_original_4to8px

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_addon_p2 | original | 71.56% | 73.12% | 234 | 93 | 86 | 60.99% | 15.80% |
| old28_addon_p2 | synthetic | 71.62% | 67.81% | 217 | 86 | 103 | 62.64% | 19.47% |
| gray40_assets53_prob50_addon_p2 | original | 74.93% | 79.38% | 254 | 85 | 66 | 71.62% | 23.44% |
| gray40_assets53_prob50_addon_p2 | synthetic | 84.43% | 88.12% | 282 | 52 | 38 | 88.94% | 43.03% |
| gray40_assets328_prob50_addon_p2 | original | 76.18% | 80.94% | 259 | 81 | 61 | 74.61% | 25.28% |
| gray40_assets328_prob50_addon_p2 | synthetic | 83.09% | 87.50% | 280 | 57 | 40 | 87.81% | 46.65% |
| gray40_assets328_prob15_addon_p2 | original | 76.72% | 80.31% | 257 | 78 | 63 | 74.24% | 23.22% |
| gray40_assets328_prob15_addon_p2 | synthetic | 82.09% | 85.94% | 275 | 60 | 45 | 84.58% | 37.25% |
| gray40_prob00_addon_p2 | original | 79.03% | 81.25% | 260 | 69 | 60 | 75.76% | 23.24% |
| gray40_prob00_addon_p2 | synthetic | 81.27% | 80.00% | 256 | 59 | 64 | 75.11% | 26.54% |
