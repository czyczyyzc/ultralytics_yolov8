# RK3576 固定翼场景 P3 Student 蒸馏微调部署交接

生成时间：{{CREATED_AT}}

## 1. 模型版本与验收范围

本交付接续旧 52 视频版的完整训练流程，在完整旧样本清单及重复次数上追加 21 段新增标注视频。先从旧 P2+P3 Teacher 继续训练 P2 增量分支，再从旧 Student 的四尺度训练 checkpoint 继续 FP32 蒸馏。不是从先前退化的纯监督微调模型初始化。

- Teacher：6 epochs，学习率 `1e-4`，仅更新 P2 adapter 和 P2 检测分支，原 P3 图及 BN 统计冻结；原 P3 原始输出需逐值一致。
- Student：8 epochs，学习率 `5e-5`，保留训练期辅助 P2、分类/DFL/跨尺度/特征蒸馏，恢复 `0.12` 的额外定位损失。
- 旧 Student 已训练的特征投影权重继续使用，不随机重建；部署导出时物理移除辅助 P2 和投影层。
- 板端仍为纯 P3/P4/P5、`960×544`、RK3576 专用 W8A8 PTQ INT8，没有 QAT，没有增加 P2 推理计算。

{{RELEASE_NOTE}}

已完成 FP32 新旧对比、九输出 ONNX 检查、RK3576 编译、Toolkit 加载和 300 帧宿主机同帧量化检查。板端 FPS、温升和曝光到结果延迟尚未实测，不能沿用旧模型或 RK3588 的测速数字。

## 2. 文件与位置

服务器交付目录：

```text
{{DELIVERY_DIR}}
```

服务器训练目录：

```text
{{TRAINING_DIR}}
```

训练配置、监控清单及签名：

```text
{{DATA_DIR}}
```

| 包内文件 | 用途 | SHA256 |
| --- | --- | --- |
| `models/detector_960x544_rk3576_w8a8.rknn` | RK3576 板端模型 | `{{MODEL_SHA256}}` |
| `models/p3_student_best_fp32.pt` | 去掉辅助 P2 的 P3 权重 | `{{PT_SHA256}}` |
| `models/detector_960x544_rkopt.onnx` | 九输出 ONNX | `{{ONNX_SHA256}}` |
| `metadata/` | 训练配置、冻结一致性、选权重、评测和量化记录 | 见 `SHA256SUMS` |
| `source/code.tar.gz` | C++ 检测、因果运动关联、GMC 和构建代码 | 见 `SHA256SUMS` |

仓库 `git@github.com:czyczyyzc/ultralytics_yolov8.git`，交付固定提交 `{{COMMIT}}`，源码遵循仓库 AGPL-3.0 许可。初始化权重的路径和 SHA256 记录于 `metadata/training_protocol.json`。

## 3. 数据、损失与 checkpoint 选择

保留旧训练体系（52 段自采灰度视频及 Anti-UAV300 RGB）的完整候选清单、所有原正样本曝光及重复次数；旧正曝光 180,264，不是仅回放 48,614 个。追加新增正样本 24,307 张、可用负样本 3,338 张。负样本继续混池轮换，不把全部候选负样本在每轮重复使用。

沿用修复后的 `balanced_v2` 排序及有上限的 tiny 重采样，避免恢复已发现的 RGB 来源合并及轮末堆积问题。每轮正曝光 218,259、负曝光 38,516，共 256,775；4–8 px 帧约 30%、负样本约 15%。原图、标签和旧模型均未覆盖。

Student 损失系数：辅助 P2 `0.25`，P3 分类蒸馏 `0.10`、DFL 蒸馏 `0.05`、P2→P3 分类 `0.25`、框 `0.10`、特征 `0.10`；tiny 倍率 `1.75`，额外定位损失 `0.12`、适用长边 4–960 px。所有数值与旧 Student 方法对应；续训使用低学习率和缩短轮数，不是重复完整的从头训练。

输入 `960×544`、batch 64、nbs 128、seed 20261015、AdamW、cosine decay、warmup 关闭。Mosaic/MixUp/Copy-Paste/旋转/透视/垂直翻转关闭；translate 0.05、scale 0.2、水平翻转 0.5、HSV 0.015/0.4/0.3。沿用 328 张候选贴图、安全缓存及 0.5 在线替换概率；不适合替换时保留原图原框，无训练框面积过滤。

新增视频的 confirmed 正样本可包含待审核或标注中快照；uncertain、未标注和未审核 absent 排除。不能把待审核标签称为已验收。`Video00004` 不参与训练及选权重，`Video00009` 用于验证；新增监控帧本身属于训练。

按原 Video00009 fitness 选择 checkpoint，不混入新增训练拟合成绩作为奖励。候选需同时满足旧场景 Recall 下降不超过 3 个百分点、4–8 px Recall 下降不超过 5 个百分点。没有合格候选则不自动交付替代模型。最终合并保留集及 Video00004 还需通过独立的发布检查；此检查不用于反复挑选训练 checkpoint。

## 4. 检测精度与量化检查

FP32 detector-only，`conf=0.03`、NMS IoU 0.45、匹配 IoU 0.5；AP 按 conf 0.001 的完整置信度排序计算。Tracker 沿用 C++ 因果运动关联及质量门控 GMC，不将检测指标冒充身份关联指标。

### 4.1 Video00004 + Video00009 合并保留集

3,780 帧、1,245 个 GT，AP 全局重算。Video00009 用于选择，因此此合并集不是完全独立测试集。

{{RETAINED_METRICS}}

### 4.2 新增 21 段视频的全量训练拟合

27,645 帧，包含 24,307 张正样本及 3,338 张负样本。不是独立固定翼泛化成绩。

{{FIT_METRICS}}

### 4.3 同帧量化检查

宿主机 RKNN Toolkit2 2.3.2，非板端精度。Video00004、Video00009 各 100 帧，合并 200 帧：

{{QUANT_NATIVE_METRICS}}

新增来源训练监控 100 帧，排除出校准清单，单独统计：

{{QUANT_FIT_METRICS}}

300 帧检查不代替全量量化评测。交付 `.rknn` 的 SHA256 等于检查记录的 `rebuilt_sha256`；报告中 `rebuilt_matches_delivered` 比较的是前置 reference 文件，并非本交付包。`pooled_native` 只合并 Video00004 和 Video00009，不包含新增训练拟合帧。
