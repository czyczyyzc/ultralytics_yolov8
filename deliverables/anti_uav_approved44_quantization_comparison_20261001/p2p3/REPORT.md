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
| pt_fp32 | 0.01 | 10.87% | 99.78% | 447 | 3664 | 1 | 85.64% | 50.86% | 99.75% |
| onnx_fp32 | 0.01 | 10.87% | 99.78% | 447 | 3664 | 1 | 85.64% | 50.86% | 99.75% |
| rknn_int8_simulator | 0.01 | 11.82% | 97.77% | 438 | 3268 | 10 | 75.59% | 33.98% | 98.02% |
| pt_fp32 | 0.03 | 16.35% | 99.78% | 447 | 2287 | 1 | 85.64% | 50.86% | 99.75% |
| onnx_fp32 | 0.03 | 16.35% | 99.78% | 447 | 2287 | 1 | 85.64% | 50.86% | 99.75% |
| rknn_int8_simulator | 0.03 | 17.52% | 97.32% | 436 | 2053 | 12 | 75.59% | 33.98% | 97.53% |
| pt_fp32 | 0.05 | 19.50% | 99.78% | 447 | 1845 | 1 | 85.64% | 50.86% | 99.75% |
| onnx_fp32 | 0.05 | 19.50% | 99.78% | 447 | 1845 | 1 | 85.64% | 50.86% | 99.75% |
| rknn_int8_simulator | 0.05 | 20.25% | 96.43% | 432 | 1701 | 16 | 75.59% | 33.98% | 96.54% |

## Video00009_selection_subset

| Backend | Conf | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pt_fp32 | 0.01 | 55.78% | 73.27% | 584 | 463 | 213 | 67.94% | 32.11% | 75.26% |
| onnx_fp32 | 0.01 | 55.78% | 73.27% | 584 | 463 | 213 | 67.94% | 32.11% | 75.26% |
| rknn_int8_simulator | 0.01 | 54.04% | 68.88% | 549 | 467 | 248 | 64.80% | 27.15% | 69.39% |
| pt_fp32 | 0.03 | 64.66% | 72.77% | 580 | 317 | 217 | 67.94% | 32.11% | 74.74% |
| onnx_fp32 | 0.03 | 64.66% | 72.77% | 580 | 317 | 217 | 67.94% | 32.11% | 74.74% |
| rknn_int8_simulator | 0.03 | 60.34% | 67.38% | 537 | 353 | 260 | 64.80% | 27.15% | 67.60% |
| pt_fp32 | 0.05 | 67.17% | 71.89% | 573 | 280 | 224 | 67.94% | 32.11% | 73.47% |
| onnx_fp32 | 0.05 | 67.17% | 71.89% | 573 | 280 | 224 | 67.94% | 32.11% | 73.47% |
| rknn_int8_simulator | 0.05 | 62.74% | 66.75% | 532 | 316 | 265 | 64.80% | 27.15% | 66.58% |

## pooled_native

| Backend | Conf | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pt_fp32 | 0.01 | 19.99% | 82.81% | 1031 | 4127 | 214 | 69.19% | 36.92% | 87.70% |
| onnx_fp32 | 0.01 | 19.99% | 82.81% | 1031 | 4127 | 214 | 69.19% | 36.92% | 87.70% |
| rknn_int8_simulator | 0.01 | 20.90% | 79.28% | 987 | 3735 | 258 | 64.44% | 28.02% | 83.94% |
| pt_fp32 | 0.03 | 28.28% | 82.49% | 1027 | 2604 | 218 | 69.19% | 36.92% | 87.45% |
| onnx_fp32 | 0.03 | 28.28% | 82.49% | 1027 | 2604 | 218 | 69.19% | 36.92% | 87.45% |
| rknn_int8_simulator | 0.03 | 28.80% | 78.15% | 973 | 2406 | 272 | 64.44% | 28.02% | 82.81% |
| pt_fp32 | 0.05 | 32.43% | 81.93% | 1020 | 2125 | 225 | 69.19% | 36.92% | 86.83% |
| onnx_fp32 | 0.05 | 32.43% | 81.93% | 1020 | 2125 | 225 | 69.19% | 36.92% | 86.83% |
| rknn_int8_simulator | 0.05 | 32.34% | 77.43% | 964 | 2017 | 281 | 64.44% | 28.02% | 81.81% |
