# 44-video detector: FP32 vs INT8

Target: rk3576; RKNN Toolkit2 2.3.2 host simulator.
Not a board accuracy or FPS measurement. Tracking is excluded.
Identical validation-loader RGB pixels for all backends: 960x544, padding 114.
Common NMS IoU .45, max_det 100; fixed metrics conf .01/.03/.05; AP floor .001.
Video00004 is held-out test; Video00009 selected checkpoints and is NOT independent test.
Native frames only: zoom validation views excluded. Pooled AP is globally recomputed.
Rebuilt RKNN SHA256 equals delivered artifact: False.
If hashes differ, this measures a rebuild with identical inputs/config, not the delivered binary.

## Video00004_test

| Backend | Conf | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pt_fp32 | 0.01 | 25.87% | 82.59% | 370 | 1060 | 78 | 57.21% | 34.17% | 80.74% |
| onnx_fp32 | 0.01 | 25.87% | 82.59% | 370 | 1060 | 78 | 57.21% | 34.17% | 80.74% |
| rknn_int8_simulator | 0.01 | 25.93% | 82.14% | 368 | 1051 | 80 | 56.23% | 20.44% | 80.74% |
| pt_fp32 | 0.03 | 32.17% | 78.12% | 350 | 738 | 98 | 57.21% | 34.17% | 75.80% |
| onnx_fp32 | 0.03 | 32.17% | 78.12% | 350 | 738 | 98 | 57.21% | 34.17% | 75.80% |
| rknn_int8_simulator | 0.03 | 31.12% | 74.33% | 333 | 737 | 115 | 56.23% | 20.44% | 72.35% |
| pt_fp32 | 0.05 | 34.86% | 74.55% | 334 | 624 | 114 | 57.21% | 34.17% | 71.85% |
| onnx_fp32 | 0.05 | 34.86% | 74.55% | 334 | 624 | 114 | 57.21% | 34.17% | 71.85% |
| rknn_int8_simulator | 0.05 | 33.96% | 72.54% | 325 | 632 | 123 | 56.23% | 20.44% | 70.37% |

## Video00009_selection_subset

| Backend | Conf | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pt_fp32 | 0.01 | 58.61% | 55.08% | 439 | 310 | 358 | 58.10% | 30.17% | 50.26% |
| onnx_fp32 | 0.01 | 58.61% | 55.08% | 439 | 310 | 358 | 58.10% | 30.17% | 50.26% |
| rknn_int8_simulator | 0.01 | 55.50% | 52.57% | 419 | 336 | 378 | 57.34% | 25.95% | 43.88% |
| pt_fp32 | 0.03 | 68.82% | 53.45% | 426 | 193 | 371 | 58.10% | 30.17% | 47.70% |
| onnx_fp32 | 0.03 | 68.82% | 53.45% | 426 | 193 | 371 | 58.10% | 30.17% | 47.70% |
| rknn_int8_simulator | 0.03 | 63.16% | 51.19% | 408 | 238 | 389 | 57.34% | 25.95% | 41.58% |
| pt_fp32 | 0.05 | 72.50% | 50.94% | 406 | 154 | 391 | 58.10% | 30.17% | 44.13% |
| onnx_fp32 | 0.05 | 72.50% | 50.94% | 406 | 154 | 391 | 58.10% | 30.17% | 44.13% |
| rknn_int8_simulator | 0.05 | 66.06% | 49.81% | 397 | 204 | 400 | 57.34% | 25.95% | 39.80% |

## pooled_native

| Backend | Conf | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pt_fp32 | 0.01 | 37.13% | 64.98% | 809 | 1370 | 436 | 55.40% | 30.94% | 65.75% |
| onnx_fp32 | 0.01 | 37.13% | 64.98% | 809 | 1370 | 436 | 55.40% | 30.94% | 65.75% |
| rknn_int8_simulator | 0.01 | 36.20% | 63.21% | 787 | 1387 | 458 | 55.11% | 23.15% | 62.61% |
| pt_fp32 | 0.03 | 45.46% | 62.33% | 776 | 931 | 469 | 55.40% | 30.94% | 61.98% |
| onnx_fp32 | 0.03 | 45.46% | 62.33% | 776 | 931 | 469 | 55.40% | 30.94% | 61.98% |
| rknn_int8_simulator | 0.03 | 43.18% | 59.52% | 741 | 975 | 504 | 55.11% | 23.15% | 57.21% |
| pt_fp32 | 0.05 | 48.75% | 59.44% | 740 | 778 | 505 | 55.40% | 30.94% | 58.22% |
| onnx_fp32 | 0.05 | 48.75% | 59.44% | 740 | 778 | 505 | 55.40% | 30.94% | 58.22% |
| rknn_int8_simulator | 0.05 | 46.34% | 57.99% | 722 | 836 | 523 | 55.11% | 23.15% | 55.33% |

