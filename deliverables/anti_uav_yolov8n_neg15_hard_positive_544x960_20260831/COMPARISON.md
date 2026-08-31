# YOLOv8n neg15 困难正样本增量训练与板端验证

日期：2026-08-31

输入：`960x544 (W x H)`

部署：RK3588S、RKNN-Toolkit2/Runtime `2.3.2`、INT8、原生 C++、3 NPU context

## 1. 结论

推荐将 `neg15 hard-positive epoch 1` 作为新的默认部署模型，detector decode 保持 `conf=0.01`。在同一 Video00004、同一 C++ 链路、同一 IoU 0.50 下，新模型相对原 `neg15 e10`：

| RKNN 模型，`conf=0.01` | TP | FP | Precision | Recall | F1 | 空帧 FP 率 |
|---|---:|---:|---:|---:|---:|---:|
| 原 neg15 e10 | 325 | 76 | 81.05% | 72.54% | 76.56% | 1.88% |
| **新 neg15 hard-positive e1** | **336** | **78** | **81.16%** | **75.00%** | **77.96%** | **2.20%** |

Recall 提升 `2.46 pp`，F1 提升 `1.40 pp`，Precision 提升 `0.11 pp`。代价是空帧 FP 率增加 `0.32 pp`，因此不能描述为所有误报指标都改善。

## 2. 数据与训练

困难正样本仅从 7 段灰度训练正样本挖掘，Anti-UAV300 RGB 样本仍完整保留作 rehearsal：

| 项目 | 数量 |
|---|---:|
| 唯一灰度正样本 | 8,424 |
| 完全漏检 `missed` | 2,425 |
| 低置信匹配 `weak` | 553 |
| 定位失败 `localization` | 34 |
| 困难正样本合计 | 3,012 |
| 困难正样本 rehearsal 槽位 | 7,930 |

增量数据集仍为 83,118 个训练槽位，其中正样本 70,650、负样本 12,468，负样本比例保持 `15.00036%`。负样本身份和顺序完全不变，43,749 个唯一正样本仍全部至少出现一次。困难正样本只替换原训练集中的重复正样本槽位。

训练从原 `neg15 e10.pt` 开始，使用 4 张 A100、global batch 64、`lr0=2e-5`、无 warmup，共训练 10 epochs。每个 epoch 都在 Video00004 上按固定阈值评测，第 1 个 epoch 的 Recall 和 F1 最优，因此选择训练文件 `weights/epoch0.pt`，不选择 `last.pt` 或按 RGB 指标产生的 `best.pt`。

## 3. PT 对比

| 模型与阈值 | TP | FP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| 原 neg15 e10，`conf=0.03` | 320 | 39 | 89.14% | 71.43% | 79.31% |
| 新 hard-positive e1，`conf=0.03` | 325 | 38 | 89.53% | 72.54% | 80.15% |
| 原 neg15 e10，`conf=0.05` | 308 | 33 | 90.32% | 68.75% | 78.07% |
| 新 hard-positive e1，`conf=0.05` | 307 | 32 | 90.56% | 68.53% | 78.02% |

PT 的提升主要发生在低分区间；最终采用板端 INT8 `conf=0.01` 的真实结果作为部署判断依据。

## 4. 板端速度

| 链路 | 完整视频 FPS | 平均 NPU 推理 | 说明 |
|---|---:|---:|---|
| detector-only | 114.75 | 19.83 ms | 3 contexts，`conf=0.01` |
| detector + RK-BoT-SORT | 108.51 | 20.77 ms | high/low/new=`0.03/0.01/0.05` |

模型结构仍为标准 YOLOv8n P3/P4/P5，8.2 GFLOPs。速度差异主要来自板端温度，不是困难正样本训练增加了计算量。

## 5. 部署配置

板端默认模型：

```text
/data/anti_uav/models/yolov8n_recall_safe_neg15_hardpos_e1_544x960_v232_int8.rknn
```

固定配置：

```text
DETECTOR_CONF=0.01
TRACK_HIGH_THRESH=0.03
TRACK_LOW_THRESH=0.01
NEW_TRACK_THRESH=0.05
```

启动：

```bash
/data/anti_uav/bin/run_recall_safe_deployment.sh VIDEO_OR_STREAM OUTPUT_DIR
```

## 6. 模型路径与哈希

47 服务器目录：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/
```

| 文件 | SHA256 |
|---|---|
| `yolov8n_recall_safe_neg15_hardpos_e1.pt` | `428807cb6214a3af8f6f87775291152058728f5042a45cfe2ffdc2b0f541a0ea` |
| `yolov8n_recall_safe_neg15_hardpos_e1_544x960_rkopt.onnx` | `00867478b302bffbb221f3faab2e27f49c25706702449bf118a8d5bad89fdba5` |
| `yolov8n_recall_safe_neg15_hardpos_e1_544x960_v232_int8.rknn` | `3d5a47132cd7b087e03170ba398db7a57af730d834ebb0c8d0c77d81e776f98e` |

服务器、本地交付目录和 RK3588S 板端 RKNN 哈希已逐一核对一致。

## 7. 适用范围

Video00004 属于 final-all-gray 训练覆盖范围，因此本结果是同视频部署回归，不能作为独立泛化证明。正式发布到新场景前，应按相同困难正样本策略重训 7 个 LOVO fold，并报告 macro Recall、Precision、FP/frame、空帧 FP 率和事件级首次捕获率。
