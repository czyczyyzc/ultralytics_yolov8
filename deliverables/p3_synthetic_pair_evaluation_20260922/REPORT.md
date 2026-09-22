# Frozen standalone P3: synthetic appearance evaluation

960x544 FP32; conf .03 for counts, matching IoU .5, NMS .45, max_det 100; AP floor .001.
Five fixed best.pt files; no training, checkpoint selection, threshold tuning, tracker or RKNN.
Full sets: 3780 frames / 1245 GT. Replaced pairs: 981 positive frames only.
The 328 source assets were used in training; Video00009 selected checkpoints. Not an independent or unseen-type test.
Original-size 4-8px group membership is frozen before replacement; do not compare shifting bbox-size denominators.

## combined / all

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | original | 61.72% | 59.44% | 740 | 459 | 505 | 60.46% | 33.66% |
| old28_p3 | synthetic | 58.25% | 53.01% | 660 | 473 | 585 | 51.08% | 24.22% |
| gray40_assets53_prob50_p3 | original | 54.96% | 60.48% | 753 | 617 | 492 | 61.58% | 34.51% |
| gray40_assets53_prob50_p3 | synthetic | 56.00% | 58.88% | 733 | 576 | 512 | 60.95% | 34.83% |
| gray40_assets328_prob50_p3 | original | 55.93% | 61.77% | 769 | 606 | 476 | 63.64% | 35.27% |
| gray40_assets328_prob50_p3 | synthetic | 56.21% | 59.60% | 742 | 578 | 503 | 62.61% | 37.07% |
| gray40_assets328_prob15_p3 | original | 54.30% | 61.93% | 771 | 649 | 474 | 62.12% | 34.83% |
| gray40_assets328_prob15_p3 | synthetic | 53.50% | 59.52% | 741 | 644 | 504 | 58.92% | 32.64% |
| gray40_prob00_p3 | original | 62.47% | 61.77% | 769 | 462 | 476 | 65.51% | 36.67% |
| gray40_prob00_p3 | synthetic | 58.14% | 53.65% | 668 | 481 | 577 | 55.98% | 26.40% |

## combined / replaced

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | original | 81.74% | 64.32% | 631 | 141 | 350 | 74.62% | 39.99% |
| old28_p3 | synthetic | 78.05% | 56.17% | 551 | 155 | 430 | 66.51% | 28.46% |
| gray40_assets53_prob50_p3 | original | 79.70% | 65.24% | 640 | 163 | 341 | 75.69% | 41.15% |
| gray40_assets53_prob50_p3 | synthetic | 83.56% | 63.20% | 620 | 122 | 361 | 76.81% | 42.67% |
| gray40_assets328_prob50_p3 | original | 81.47% | 66.77% | 655 | 149 | 326 | 77.37% | 41.57% |
| gray40_assets328_prob50_p3 | synthetic | 83.85% | 64.02% | 628 | 121 | 353 | 77.24% | 44.91% |
| gray40_assets328_prob15_p3 | original | 81.26% | 67.18% | 659 | 152 | 322 | 76.55% | 41.56% |
| gray40_assets328_prob15_p3 | synthetic | 81.06% | 64.12% | 629 | 147 | 352 | 74.30% | 39.46% |
| gray40_prob00_p3 | original | 86.54% | 66.87% | 656 | 102 | 325 | 77.94% | 42.67% |
| gray40_prob00_p3 | synthetic | 82.10% | 56.57% | 555 | 121 | 426 | 68.86% | 29.80% |

## combined / replaced_original_4to8px

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | original | 82.04% | 62.31% | 443 | 97 | 268 | 72.65% | 34.58% |
| old28_p3 | synthetic | 77.49% | 54.71% | 389 | 113 | 322 | 64.78% | 24.40% |
| gray40_assets53_prob50_p3 | original | 77.76% | 63.43% | 451 | 129 | 260 | 73.70% | 34.73% |
| gray40_assets53_prob50_p3 | synthetic | 80.54% | 58.79% | 418 | 101 | 293 | 73.23% | 34.73% |
| gray40_assets328_prob50_p3 | original | 80.10% | 65.12% | 463 | 115 | 248 | 75.83% | 35.22% |
| gray40_assets328_prob50_p3 | synthetic | 81.61% | 59.92% | 426 | 96 | 285 | 74.17% | 37.49% |
| gray40_assets328_prob15_p3 | original | 80.38% | 65.68% | 467 | 114 | 244 | 75.20% | 35.61% |
| gray40_assets328_prob15_p3 | synthetic | 79.89% | 61.46% | 437 | 110 | 274 | 72.12% | 31.76% |
| gray40_prob00_p3 | original | 87.62% | 65.68% | 467 | 66 | 244 | 77.50% | 37.09% |
| gray40_prob00_p3 | synthetic | 82.05% | 55.27% | 393 | 86 | 318 | 67.01% | 24.86% |

## Video00004 / all

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | original | 57.51% | 75.22% | 337 | 249 | 111 | 69.35% | 43.44% |
| old28_p3 | synthetic | 51.30% | 66.29% | 297 | 282 | 151 | 56.69% | 26.59% |
| gray40_assets53_prob50_p3 | original | 45.14% | 75.67% | 339 | 412 | 109 | 70.20% | 41.34% |
| gray40_assets53_prob50_p3 | synthetic | 40.79% | 64.29% | 288 | 418 | 160 | 59.61% | 32.95% |
| gray40_assets328_prob50_p3 | original | 45.27% | 75.89% | 340 | 411 | 108 | 70.93% | 41.12% |
| gray40_assets328_prob50_p3 | synthetic | 41.80% | 66.52% | 298 | 415 | 150 | 62.97% | 36.25% |
| gray40_assets328_prob15_p3 | original | 43.88% | 76.79% | 344 | 440 | 104 | 69.98% | 41.64% |
| gray40_assets328_prob15_p3 | synthetic | 39.87% | 68.53% | 307 | 463 | 141 | 59.74% | 30.90% |
| gray40_prob00_p3 | original | 53.59% | 76.56% | 343 | 297 | 105 | 75.07% | 45.39% |
| gray40_prob00_p3 | synthetic | 46.45% | 64.29% | 288 | 332 | 160 | 59.18% | 25.53% |

## Video00004 / replaced

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | original | 89.91% | 75.00% | 312 | 35 | 104 | 87.18% | 52.53% |
| old28_p3 | synthetic | 80.00% | 65.38% | 272 | 68 | 144 | 74.32% | 31.41% |
| gray40_assets53_prob50_p3 | original | 82.85% | 75.48% | 314 | 65 | 102 | 86.62% | 48.77% |
| gray40_assets53_prob50_p3 | synthetic | 78.74% | 63.22% | 263 | 71 | 153 | 76.79% | 39.60% |
| gray40_assets328_prob50_p3 | original | 84.22% | 75.72% | 315 | 59 | 101 | 87.62% | 48.45% |
| gray40_assets328_prob50_p3 | synthetic | 81.25% | 65.62% | 273 | 63 | 143 | 79.47% | 42.83% |
| gray40_assets328_prob15_p3 | original | 85.07% | 76.68% | 319 | 56 | 97 | 88.13% | 50.14% |
| gray40_assets328_prob15_p3 | synthetic | 78.12% | 67.79% | 282 | 79 | 134 | 77.40% | 36.42% |
| gray40_prob00_p3 | original | 93.81% | 76.44% | 318 | 21 | 98 | 89.63% | 52.52% |
| gray40_prob00_p3 | synthetic | 82.45% | 63.22% | 263 | 56 | 153 | 72.55% | 28.73% |

## Video00004 / replaced_original_4to8px

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | original | 90.25% | 73.40% | 287 | 31 | 104 | 86.45% | 51.35% |
| old28_p3 | synthetic | 80.25% | 64.45% | 252 | 62 | 139 | 73.36% | 30.02% |
| gray40_assets53_prob50_p3 | original | 82.57% | 73.91% | 289 | 61 | 102 | 85.68% | 47.36% |
| gray40_assets53_prob50_p3 | synthetic | 77.78% | 60.87% | 238 | 68 | 153 | 74.97% | 36.44% |
| gray40_assets328_prob50_p3 | original | 83.82% | 74.17% | 290 | 56 | 101 | 86.70% | 47.11% |
| gray40_assets328_prob50_p3 | synthetic | 80.52% | 63.43% | 248 | 60 | 143 | 77.93% | 39.82% |
| gray40_assets328_prob15_p3 | original | 84.73% | 75.19% | 294 | 53 | 97 | 87.30% | 48.71% |
| gray40_assets328_prob15_p3 | synthetic | 77.18% | 65.73% | 257 | 76 | 134 | 75.67% | 33.31% |
| gray40_prob00_p3 | original | 94.21% | 74.94% | 293 | 18 | 98 | 88.98% | 51.08% |
| gray40_prob00_p3 | synthetic | 82.03% | 61.89% | 242 | 53 | 149 | 71.38% | 27.12% |

## Video00009 / all

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | original | 65.74% | 50.56% | 403 | 210 | 394 | 56.00% | 28.01% |
| old28_p3 | synthetic | 65.52% | 45.55% | 363 | 191 | 434 | 50.58% | 23.76% |
| gray40_assets53_prob50_p3 | original | 66.88% | 51.94% | 414 | 205 | 383 | 57.68% | 30.59% |
| gray40_assets53_prob50_p3 | synthetic | 73.80% | 55.83% | 445 | 158 | 352 | 63.59% | 36.51% |
| gray40_assets328_prob50_p3 | original | 68.75% | 53.83% | 429 | 195 | 368 | 60.31% | 32.01% |
| gray40_assets328_prob50_p3 | synthetic | 73.15% | 55.71% | 444 | 163 | 353 | 63.81% | 38.14% |
| gray40_assets328_prob15_p3 | original | 67.14% | 53.58% | 427 | 209 | 370 | 58.54% | 30.85% |
| gray40_assets328_prob15_p3 | synthetic | 70.57% | 54.45% | 434 | 181 | 363 | 60.69% | 34.60% |
| gray40_prob00_p3 | original | 72.08% | 53.45% | 426 | 165 | 371 | 60.64% | 31.86% |
| gray40_prob00_p3 | synthetic | 71.83% | 47.68% | 380 | 149 | 417 | 56.58% | 27.93% |

## Video00009 / replaced

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | original | 75.06% | 56.46% | 319 | 106 | 246 | 66.42% | 32.26% |
| old28_p3 | synthetic | 76.23% | 49.38% | 279 | 87 | 286 | 61.36% | 26.89% |
| gray40_assets53_prob50_p3 | original | 76.89% | 57.70% | 326 | 98 | 239 | 68.04% | 35.93% |
| gray40_assets53_prob50_p3 | synthetic | 87.50% | 63.19% | 357 | 51 | 208 | 76.92% | 44.94% |
| gray40_assets328_prob50_p3 | original | 79.07% | 60.18% | 340 | 90 | 225 | 70.44% | 37.22% |
| gray40_assets328_prob50_p3 | synthetic | 85.96% | 62.83% | 355 | 58 | 210 | 75.94% | 46.61% |
| gray40_assets328_prob15_p3 | original | 77.98% | 60.18% | 340 | 96 | 225 | 68.76% | 36.00% |
| gray40_assets328_prob15_p3 | synthetic | 83.61% | 61.42% | 347 | 68 | 218 | 72.43% | 41.85% |
| gray40_prob00_p3 | original | 80.67% | 59.82% | 338 | 81 | 227 | 69.90% | 36.22% |
| gray40_prob00_p3 | synthetic | 81.79% | 51.68% | 292 | 65 | 273 | 66.52% | 31.28% |

## Video00009 / replaced_original_4to8px

| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old28_p3 | original | 70.27% | 48.75% | 156 | 66 | 164 | 54.36% | 14.36% |
| old28_p3 | synthetic | 72.87% | 42.81% | 137 | 51 | 183 | 54.29% | 17.03% |
| gray40_assets53_prob50_p3 | original | 70.43% | 50.62% | 162 | 68 | 158 | 56.96% | 18.51% |
| gray40_assets53_prob50_p3 | synthetic | 84.51% | 56.25% | 180 | 33 | 140 | 71.31% | 32.78% |
| gray40_assets328_prob50_p3 | original | 74.57% | 54.06% | 173 | 59 | 147 | 61.37% | 20.42% |
| gray40_assets328_prob50_p3 | synthetic | 83.18% | 55.62% | 178 | 36 | 142 | 69.89% | 34.96% |
| gray40_assets328_prob15_p3 | original | 73.93% | 54.06% | 173 | 61 | 147 | 58.87% | 19.20% |
| gray40_assets328_prob15_p3 | synthetic | 84.11% | 56.25% | 180 | 34 | 140 | 68.61% | 30.49% |
| gray40_prob00_p3 | original | 78.38% | 54.37% | 174 | 48 | 146 | 61.75% | 19.53% |
| gray40_prob00_p3 | synthetic | 82.07% | 47.19% | 151 | 33 | 169 | 62.09% | 22.34% |
