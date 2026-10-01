# 44 视频版 YOLO：FP32 与 RKNN INT8 精度对照

评测日期：2026-10-01，47 服务器。仅评测 detector，不包含 Dist、GMC 或其他 tracker。

**结果来源是 RK3576 目标的 Toolkit2 2.3.2 主机量化模拟器，不是板端实测。** 模拟器需要由 ONNX 重新建立量化图，本次沿用交付模型的源 ONNX、384 图校准清单、归一化及编译配置。重编文件 SHA256 与交付 RKNN 不同，所以这些数值不能作为交付二进制的板端验收结果；交付文件没有被覆盖。

## 评测口径

- 输入宽 960、高 544。三个后端共享同一验证数据加载器产生的 RGB 像素、114 padding 和 `/255` 归一化；不是分别调用三套预处理。
- Precision、Recall、计数和小目标召回默认取 `conf=0.03`；NMS IoU=0.45、匹配 IoU=0.5、最多 100 框。AP 取 `conf=0.001` 起的完整排序曲线。
- Video00004：2,359 帧、448 GT，独立留出测试。Video00009：1,421 原尺寸验证帧、797 GT，曾用于挑选 checkpoint，不能称为独立测试。
- 合并集：3,780 帧、1,245 GT，排除 64 张 zoom 验证视图。计数求和后重新计算 P/R；mAP 从所有预测重算，不平均两个视频或各分片的 mAP。
- 两种模型各分成 16 个互不重叠的评测分片；合并验证完整覆盖，并恢复原帧顺序后进行 AP 排序。
- 4–8 px 指标按 GT 在模型输入上的长边分桶。未使用评测视频校准、训练或调阈值。

## 合并结果

| 模型 | 后端 | Precision | Recall | TP | FP | FN | mAP50 | mAP50–95 | 4–8 px Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P3 | PT FP32 | 45.46% | 62.33% | 776 | 931 | 469 | 55.40% | 30.94% | 61.98% |
| P3 | RKNN INT8 模拟 | 43.18% | 59.52% | 741 | 975 | 504 | 55.11% | 23.15% | 57.21% |
| Frozen-P3 + Add-on P2 | PT FP32 | 28.28% | 82.49% | 1027 | 2604 | 218 | 69.19% | 36.92% | 87.45% |
| Frozen-P3 + Add-on P2 | RKNN INT8 模拟 | 28.80% | 78.15% | 973 | 2406 | 272 | 64.44% | 28.02% | 82.81% |

| 模型 | Precision 变化 | Recall 变化 | FP 变化 | mAP50 变化 | mAP50–95 变化 | 4–8 px Recall 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P3 | -2.28 pp | -2.81 pp | +44 | -0.29 pp | -7.79 pp | -4.77 pp |
| Frozen-P3 + Add-on P2 | +0.51 pp | -4.34 pp | -198 | -4.75 pp | -8.90 pp | -4.64 pp |

`pp` 为百分点，变化均为 INT8 减 FP32。ONNX FP32 与 PT FP32 的 TP/FP/FN 一致，本表四舍五入精度下的指标也一致；P2 分视频 mAP50–95 的浮点差异不足 0.0003 个百分点，未观察到有实质影响的导出损失。

## 分视频结果

| 视频 | 模型 | 后端 | Precision | Recall | FP | mAP50 | mAP50–95 | 4–8 px Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Video00004 | P3 | FP32 | 32.17% | 78.12% | 738 | 57.21% | 34.17% | 75.80% |
| Video00004 | P3 | INT8 模拟 | 31.12% | 74.33% | 737 | 56.23% | 20.44% | 72.35% |
| Video00004 | P2+P3 | FP32 | 16.35% | 99.78% | 2287 | 85.64% | 50.86% | 99.75% |
| Video00004 | P2+P3 | INT8 模拟 | 17.52% | 97.32% | 2053 | 75.59% | 33.98% | 97.53% |
| Video00009 | P3 | FP32 | 68.82% | 53.45% | 193 | 58.10% | 30.17% | 47.70% |
| Video00009 | P3 | INT8 模拟 | 63.16% | 51.19% | 238 | 57.34% | 25.95% | 41.58% |
| Video00009 | P2+P3 | FP32 | 64.66% | 72.77% | 317 | 67.94% | 32.11% | 74.74% |
| Video00009 | P2+P3 | INT8 模拟 | 60.34% | 67.38% | 353 | 64.80% | 27.15% | 67.60% |

## 如何解读

P3 的 mAP50 几乎不变，不代表量化基本无损：Recall、小目标召回均下降，且 mAP50–95 的损失明显，说明更严格 IoU 下的定位表现受到影响。P2+P3 的 Precision 略增来自 FP 减少，同时少了 54 个 TP，不能称为整体性能提升。Video00004 上 P2+P3 的 mAP50 单独下降 10.05 pp，合并表不能替代独立测试结果。

FP32 的高误检问题也不是量化造成的：P2+P3 在合并集量化前已有 2,604 个 FP，量化后仍有 2,406 个。这一训练版本仍不宜直接替换生产模型。若继续优化量化，应优先检查匹配新数据分布的校准集、检测头输出量化及混合精度方案；需要实际试验，不能承诺无损。

历史 FP32 报告采用 batch=32，本次使用同帧 batch=1、关闭 TF32 的配对链路重新评测，个别计数与历史报告有小差异；量化损失按本次配对 FP32 基线计算。

## 文件与复现

本地 `p3/REPORT.md`、`p2p3/REPORT.md` 包含三个后端、两个视频及合并集在 `conf=0.01/0.03/0.05` 下的完整指标。对应 `results.json`、`protocol.json` 保存原始数值、模型/清单 SHA、分片来源和构建信息。

47 服务器完整评测目录：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_quantization_eval_rk3576_20261001/p3_merged/
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_quantization_eval_rk3576_20261001/p2p3_merged/
```

源权重仍为 `approved44_direct_online328_20260929/training_p3/p3/weights/best.pt` 和 `training_addon/p2/weights/best.pt`，完整路径及 SHA 见各模型 `protocol.json`。交付 RKNN 仍在服务器 `runs/anti_uav/approved_rknn_exports_rk3576_20260930/44_p3/` 与 `44_p2p3/`，没有替换。

复现脚本：`/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/compare_gray_rknn_quantization.py`。传入 `--weights --onnx --rknn --calibration --output --target rk3576 --device cuda:0`；可用 `--shards 16 --shard-index 0..15` 独立运行，然后 `--output <新目录> --merge-shards <16个分片目录>` 汇总。不要用 `--limit-per-split` 的 smoke 结果代替全量评测。

当前原生部署的 padding 默认值可能为 0，而这里统一采用训练/验证的 114。板端验收需对齐实际预处理，并验证交付二进制；不得把此报告推广为 RK3588 模型、Dist/GMC 跟踪精度或 FPS 报告。
