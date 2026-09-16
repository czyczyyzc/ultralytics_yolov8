# Interim Online Random-Scale Comparison

P3 only, matched epoch 14; actual input 960x544. Online add-on P2 has not completed.

## Independent Native Gray Validation

| Model | Conf | P | R | FP/1000 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_p3 | 0.01 | 33.54% | 41.53% | 461.65 | 39.23% | 22.14% |
| online_p3 | 0.01 | 27.85% | 40.40% | 586.91 | 37.33% | 21.39% |
| static_p3 | 0.03 | 43.18% | 38.52% | 284.31 | 39.23% | 22.14% |
| online_p3 | 0.03 | 39.51% | 38.27% | 328.64 | 37.33% | 21.39% |
| static_p3 | 0.05 | 48.05% | 37.01% | 224.49 | 39.23% | 22.14% |
| online_p3 | 0.05 | 44.54% | 37.39% | 261.08 | 37.33% | 21.39% |

## Completed Add-On Models On Video00004

These are previous completed models, not the unfinished online model. Conf=0.03.

| Model | P | R | FP | mAP50 |
| --- | ---: | ---: | ---: | ---: |
| old_addon | 50.34% | 100.00% | 442 | 90.13% |
| new_addon | 29.73% | 100.00% | 1059 | 86.26% |

Old add-on: original manual-clips0123 model. New add-on: completed fixed-480-augmentation model.
Do not use these test results to select epochs or add Video00004 frames to training.
