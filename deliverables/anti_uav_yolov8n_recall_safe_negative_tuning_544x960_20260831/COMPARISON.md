# YOLOv8n 负样本 Recall-Safe 调优报告

日期：2026-08-31

输入：`960x544 (W x H)`

部署：RK3588S、RKNN-Toolkit2/Runtime `2.3.2`、INT8、原生 C++、3 NPU context

## 1. 结论

负样本可以降低 FP，同时不必牺牲当前部署工作点的 Recall。最终保留两个模型：

1. **严格保 Recall：classification-head-only epoch 3**
   - 只更新 `model.22.cv3.*` 分类分支，backbone、neck、box/DFL 分支完全冻结。
   - 权重审计显示非 `cv3` 参数最大绝对差异为 `0.0`。
   - RKNN `conf=0.01` 与无负样本基线同为 `342 TP / 76.34% Recall`，FP 从 `368` 降到 `247`，下降 `32.9%`。
   - 这是“Recall 不能降低”要求下的推荐模型。

2. **最佳综合性能：15% recall-safe full fine-tune epoch 10**
   - RKNN `conf=0.03` 为 `294 TP / 60 FP`，Precision `83.05%`、Recall `65.62%`、F1 `73.32%`。
   - 相比旧无负样本部署 `conf=0.25`，Recall 提升 `7.81 pp`，FP 下降 `73.2%`。
   - 适合允许重新标定阈值、优先追求 F1 和低 FP 的部署。

不推荐正样本偏置 calibration。`224 positive + 112 negative` 的 RKNN 在 `conf=0.05` 反而比平衡 calibration 少 5 个 TP，最终统一采用 `224 positive + 224 negative`。

## 2. 负样本策略

旧 neg20 数据集包含 `70,650` 个正样本和 `17,662` 个负样本，困难负样本优先加入，其余随机抽样。新策略使用：

- 负样本占比候选：`10%`、`15%`，最终选择 `15%`，即 `12,468` 个负样本。
- 目标出现前后 `0.5 s` 的空帧不作为负样本，排除 `1,374` 个过渡帧。
- 连续灰度空帧按时间抽稀，去除 `11,906` 个高度重复帧。
- 困难负样本最多占负样本配额 `15%`；实际选择 `954` 个，占 `7.65%`。
- RGB 与 7 段灰度视频按组 round-robin，避免长视频支配负样本池。
- 正样本 `70,650` 个全部保留，不通过删除正样本做“平衡”。

数据构建脚本：`scripts/anti_uav/build_recall_safe_negative_mix.py`

分类头训练脚本：`scripts/anti_uav/train_detector_class_head_only.py`

## 3. PT 筛选结果

Video00004，共 `2,359` 帧，其中目标存在 `448` 帧、目标缺失 `1,911` 帧；匹配阈值为 IoU `0.50`。

| 模型与阈值 | TP | FP | Precision | Recall | F1 | 空帧 FP 率 |
|---|---:|---:|---:|---:|---:|---:|
| 无负样本 baseline, `conf=0.05` | 305 | 219 | 58.21% | 68.08% | 62.76% | 10.31% |
| neg15 full FT e10, `conf=0.05` | 308 | 33 | 90.32% | 68.75% | 78.07% | 1.52% |
| class-head e3, `conf=0.05` | 305 | 122 | 71.43% | 68.08% | 69.71% | 6.12% |
| 无负样本 baseline, `conf=0.25` | 271 | 185 | 59.43% | 60.49% | 59.96% | 9.52% |
| class-head e3, `conf=0.20` | 275 | 42 | 86.75% | 61.38% | 71.90% | 1.99% |

`class-head e3` 在相同 `conf=0.05` 下 TP 与 baseline 完全一致；`full FT e10` 则在 Recall 略升的同时大幅压低 FP。

## 4. RKNN 板端结果

### 4.1 相同高召回阈值 `conf=0.01`

| RKNN 模型 | TP | FP | Precision | Recall | F1 | 空帧 FP 率 |
|---|---:|---:|---:|---:|---:|---:|
| 无负样本 baseline | 342 | 368 | 48.17% | 76.34% | 59.07% | 11.62% |
| 旧 neg20 full FT | 309 | 84 | 78.63% | 68.97% | 73.48% | 1.57% |
| 新 neg15 recall-safe full FT | 325 | 76 | 81.05% | 72.54% | 76.56% | 1.88% |
| **新 neg15 class-head e3** | **342** | **247** | **58.06%** | **76.34%** | **65.96%** | **10.52%** |

旧 neg20 的主要问题是压低 FP 的同时丢失 33 个 TP。新 full FT 找回其中 16 个 TP；class-head e3 找回全部 33 个 TP，并仍减少 121 个 FP。

### 4.2 推荐部署工作点

| 工作点 | TP | FP | Precision | Recall | F1 | 空帧 FP 率 |
|---|---:|---:|---:|---:|---:|---:|
| 旧无负样本部署, `conf=0.25` | 259 | 224 | 53.62% | 57.81% | 55.64% | 9.94% |
| strict Recall: class-head e3, `conf=0.15` | 271 | 90 | 75.07% | 60.49% | 67.00% | 3.98% |
| best F1: full FT e10, `conf=0.03` | 294 | 60 | 83.05% | 65.62% | 73.32% | 1.47% |
| high Recall: full FT e10, `conf=0.004` | 330 | 97 | 77.28% | 73.66% | 75.43% | 2.25% |

`conf < 0.004` 不可用。INT8 分数在此处存在量化台阶；`conf=0.001` 会从 427 个左右的检测突然增加到 `301,952` 个检测，后处理耗时升到 `11.02 ms/frame`。

## 5. 板端速度

| 模型 | 条件 | 三核 FPS | 单帧等效 FPS | 平均推理 |
|---|---|---:|---:|---:|
| class-head e3 | 完整视频，`conf=0.01`，54.5 C 起跑 | **116.97** | 42.56 | 19.90 ms |
| full FT e10 | 完整视频，`conf=0.05`，冷态 | **116.99** | 42.50 | 19.93 ms |
| full FT e10 | 59 C 起跑并升到 82 C | 110.30 | 41.66 | 20.25 ms |

模型拓扑和计算量没有变化。110--117 FPS 的波动来自板端温度和热限频，不是负样本训练引起。

## 6. 推荐配置

严格要求 Recall 不低于无负样本高召回基线时：

```bash
MODEL=/data/anti_uav/models/yolov8n_recall_safe_classhead_e3_544x960_v232_int8.rknn

/data/anti_uav/bin/native_yolov8_video "$MODEL" VIDEO.mp4 \
  --workers 3 --core-mask 0_1_2 --worker-cpu-base 4 --queue-size 3 \
  --tracker none --conf 0.01 --nms-iou 0.45
```

追求综合 F1 和低 FP 时：

```bash
MODEL=/data/anti_uav/models/yolov8n_recall_safe_neg15_e10_544x960_v232_int8_balanced.rknn

/data/anti_uav/bin/native_yolov8_video "$MODEL" VIDEO.mp4 \
  --workers 3 --core-mask 0_1_2 --worker-cpu-base 4 --queue-size 3 \
  --tracker none --conf 0.03 --nms-iou 0.45
```

接 RK-BoT-SORT 时建议 detector decode/second-stage 使用 `0.01`，第一阶段匹配从 `0.03` 起，新轨迹阈值从 `0.05` 起，再在独立 LOVO fold 上标定。

## 7. 模型路径与哈希

### 47 服务器

严格保 Recall RKNN：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_recall_safe_classhead_e3_544x960_20260831_v232_int8/yolov8n_recall_safe_classhead_e3_544x960_v232_int8.rknn
SHA256 a83c6a11a45d2cd6aff17655dc0279f2f276115af60a82a57099fa4c462a309a
```

最佳 F1 RKNN：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_recall_safe_neg15_e10_544x960_20260831_v232_int8/yolov8n_recall_safe_neg15_e10_544x960_v232_int8_balanced.rknn
SHA256 a61304fe4df8789a589e7eef175aa3208e0004ac12583b9b33d2c72cf5aa772e
```

### RK3588S 板端

```text
/data/anti_uav/models/yolov8n_recall_safe_classhead_e3_544x960_v232_int8.rknn
/data/anti_uav/models/yolov8n_recall_safe_neg15_e10_544x960_v232_int8_balanced.rknn
```

### 本地交付目录

```text
/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/anti_uav_yolov8n_recall_safe_negative_tuning_544x960_20260831/
```

完整阈值结果：`board/threshold_sweep.json`、`board/threshold_sweep.csv`。

## 8. 适用范围

本报告是 Video00004 同视频、同 IoU、同 C++ 链路的严格部署回归，但 `final_all_gray` 训练集合覆盖该视频，因此不能代替真正独立泛化结论。发布前应使用相同策略重训并汇总 7 个 LOVO fold；验收规则应设为：每个 fold 的 Recall 不低于原模型允许容差，同时汇总 FP/frame、空帧 FP 率和 F1。
