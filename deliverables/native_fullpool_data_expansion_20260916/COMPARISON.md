# Native Data Expansion: Controlled Detector Comparison

P3 15 epochs, then frozen-P3 add-on P2 15 epochs per arm. Same initialization,
seed, base augmentation and fixed 960x544 validation/selection policy.
More data means more optimizer steps at matched epochs; compute is not matched.
PT FP32 detector results, not RKNN INT8, tracker metrics or board FPS.
Old deployment weights are a practical reference, not the controlled data-only baseline.
Video00004 is test-only; no checkpoint selection or mining uses this test.

## gray_validation

| Model | Conf | P | R | FP | mAP50 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_14_p3 | 0.01 | 37.98% | 42.03% | 547 | 39.62% | 35.20% |
| baseline_14_addon | 0.01 | 31.49% | 47.18% | 818 | 39.24% | 44.90% |
| expanded_28_p3 | 0.01 | 57.39% | 53.58% | 317 | 56.00% | 49.74% |
| expanded_28_addon | 0.01 | 52.85% | 67.50% | 480 | 61.55% | 69.39% |
| deployment_old_p3 | 0.01 | 37.51% | 40.15% | 533 | 38.75% | 32.65% |
| deployment_old_addon | 0.01 | 36.28% | 47.93% | 671 | 41.74% | 46.94% |
| baseline_14_p3 | 0.03 | 46.66% | 39.40% | 359 | 39.62% | 30.36% |
| baseline_14_addon | 0.03 | 38.31% | 46.05% | 591 | 39.24% | 42.60% |
| expanded_28_p3 | 0.03 | 65.74% | 50.56% | 210 | 56.00% | 44.90% |
| expanded_28_addon | 0.03 | 60.29% | 67.25% | 353 | 61.55% | 68.88% |
| deployment_old_p3 | 0.03 | 45.85% | 38.14% | 359 | 38.75% | 28.83% |
| deployment_old_addon | 0.03 | 43.42% | 45.55% | 473 | 41.74% | 42.60% |
| baseline_14_p3 | 0.05 | 50.41% | 38.27% | 300 | 39.62% | 28.57% |
| baseline_14_addon | 0.05 | 41.44% | 45.55% | 513 | 39.24% | 41.58% |
| expanded_28_p3 | 0.05 | 68.92% | 48.68% | 175 | 56.00% | 42.35% |
| expanded_28_addon | 0.05 | 62.51% | 66.75% | 319 | 61.55% | 68.11% |
| deployment_old_p3 | 0.05 | 48.91% | 36.64% | 305 | 38.75% | 26.28% |
| deployment_old_addon | 0.05 | 45.05% | 43.91% | 427 | 41.74% | 39.54% |

## Video00004

| Model | Conf | P | R | FP | mAP50 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_14_p3 | 0.01 | 56.09% | 80.13% | 281 | 74.52% | 78.02% |
| baseline_14_addon | 0.01 | 33.91% | 100.00% | 873 | 87.76% | 100.00% |
| expanded_28_p3 | 0.01 | 50.28% | 80.58% | 357 | 69.35% | 78.52% |
| expanded_28_addon | 0.01 | 39.68% | 100.00% | 681 | 95.81% | 100.00% |
| deployment_old_p3 | 0.01 | 38.34% | 81.47% | 587 | 72.78% | 79.51% |
| deployment_old_addon | 0.01 | 35.61% | 100.00% | 810 | 90.21% | 100.00% |
| baseline_14_p3 | 0.03 | 70.70% | 74.33% | 138 | 74.52% | 71.60% |
| baseline_14_addon | 0.03 | 44.93% | 98.88% | 543 | 87.76% | 98.77% |
| expanded_28_p3 | 0.03 | 57.51% | 75.22% | 249 | 69.35% | 72.59% |
| expanded_28_addon | 0.03 | 51.67% | 100.00% | 419 | 95.81% | 100.00% |
| deployment_old_p3 | 0.03 | 54.06% | 77.23% | 294 | 72.78% | 74.81% |
| deployment_old_addon | 0.03 | 50.34% | 100.00% | 442 | 90.21% | 100.00% |
| baseline_14_p3 | 0.05 | 75.71% | 70.98% | 102 | 74.52% | 67.90% |
| baseline_14_addon | 0.05 | 50.92% | 98.88% | 427 | 87.76% | 98.77% |
| expanded_28_p3 | 0.05 | 59.28% | 73.44% | 226 | 69.35% | 70.62% |
| expanded_28_addon | 0.05 | 55.93% | 100.00% | 353 | 95.81% | 100.00% |
| deployment_old_p3 | 0.05 | 62.29% | 74.11% | 201 | 72.78% | 71.36% |
| deployment_old_addon | 0.05 | 58.05% | 99.78% | 323 | 90.21% | 99.75% |

