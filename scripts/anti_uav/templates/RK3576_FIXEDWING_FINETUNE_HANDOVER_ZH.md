# RK3576 固定翼场景适配 P3 检测与跟踪部署交接

生成时间：{{CREATED_AT}}

## 1. 版本与验收范围

本交付从已交付的 **旧 52 视频 P3 Student** 初始化，使用全部 21 段新增标注视频及旧样本回放做 8 epochs 监督微调。部署为纯 P3/P4/P5 三尺度模型，输入 `960×544`，RK3576 专用 W8A8 INT8。没有重新训练 Teacher，没有增加 P2 推理，没有 QAT。

{{RELEASE_NOTE}}

已完成微调前后 FP32 检测对比、九输出 ONNX 检查、RK3576 编译、Toolkit 加载和 300 帧宿主机同帧量化检查；交付二进制就是实际完成检查的 `.rknn`。宿主机检查不等于板端验收，未实测本模型在 RK3576 的 FPS、温升或曝光到结果延迟。

Tracker 沿用原生 C++ 因果运动关联及质量门控 GMC，权重更新不代表身份关联准确率已提高。下列精度均为 detector-only，不含跟踪筛选。

## 2. 文件与位置

服务器交付目录：

```text
{{DELIVERY_DIR}}
```

服务器训练目录：

```text
{{TRAINING_DIR}}
```

冻结训练清单及标签缓存：

```text
{{DATA_DIR}}
```

| 包内文件 | 用途 | SHA256 |
| --- | --- | --- |
| `models/detector_960x544_rk3576_w8a8.rknn` | RK3576 专用板端模型 | `{{MODEL_SHA256}}` |
| `models/p3_student_best_fp32.pt` | 微调后的 P3 检测权重 | `{{PT_SHA256}}` |
| `models/detector_960x544_rkopt.onnx` | 九输出 RKNN 优化 ONNX | `{{ONNX_SHA256}}` |
| `metadata/` | 训练来源、指标、量化检查与编译日志 | 见 `SHA256SUMS` |
| `source/code.tar.gz` | 原生 C++ 检测、关联、GMC 与构建脚本 | 见 `SHA256SUMS` |

源码仓库：`git@github.com:czyczyyzc/ultralytics_yolov8.git`，固定提交 `{{COMMIT}}`。源码遵循仓库 AGPL-3.0 许可。

## 3. 训练与数据配置

旧权重来自 `approved52_public_repro_cross4x_realigned_20261022/p3_student_distilled_best_fp32.pt`，其 SHA256 和完整路径记录在 `metadata/training_protocol.json`。

- 全部新增正样本 24,307 张保留；旧正样本回放 48,614 个曝光，按来源和目标尺寸抽取，不等于完整重跑旧训练清单。
- 新增可用负样本 3,338 张进入审核负样本池，连同旧池轮换；每轮 72,921 个正曝光、12,868 个负曝光，共 85,789 个曝光。候选池图数不等于每轮训练图数。
- `balanced_v2` 采样下 4–8 px 帧约 30%、负样本约 15%。小目标尺寸按增强前 `960×544` 画布上的框长边判断。
- batch 64、nbs 128、AdamW、初始学习率 `5e-5`、cosine decay、warmup 关闭、seed 20261030；8 epochs，每轮保存 checkpoint。
- 使用 FP32 监督损失及训练 AMP，不启用 QAT 或新增蒸馏。部署继续使用 PTQ INT8。
- Mosaic/MixUp/Copy-Paste/旋转/透视/垂直翻转关闭；translate 0.05、scale 0.2、水平翻转 0.5，HSV 0.015/0.4/0.3。
- 沿用 328 张候选透明贴图；仅对具备安全背景缓存的帧按 0.5 概率替换并更新框。无安全缓存、失败或不适合替换的样本保留原图、原框；不按面积删除训练框。
- 冻结快照中的确定正样本可来自 approved/submitted/in_progress 视频；未标注、uncertain 和未审核 absent 排除。不能把待审核正样本称为已通过验收。

`Video00004` 不参与训练或选择；`Video00009` 用于旧场景保留监控。新增视频全部用于训练，其监控帧也属于训练，不是独立固定翼验证集。Checkpoint 按旧场景验证和新增场景拟合的组合指标选择，并惩罚旧场景 Recall 下降超过 3 个百分点或 4–8 px Recall 下降超过 5 个百分点的候选。

## 4. 检测精度对比

共同协议：输入 `960×544`、FP32 detector-only；固定指标 `conf=0.03`、NMS IoU 0.45、匹配 IoU 0.5；AP 使用 conf 0.001 的完整排序计算。Recall 排在 Precision 前。

### 4.1 保留集：Video00004 + Video00009

合并 3,780 张原图、1,245 个 GT，AP 从全体预测重新计算，不平均两段 AP。Video00009 参与选择，因此合并集不是完全独立测试集。

{{RETAINED_METRICS}}

### 4.2 全部新增 21 段视频：训练拟合

27,645 张确定标注的帧，包括 24,307 张正样本及 3,338 张负样本。此表衡量对新增训练来源的适配，不证明未见固定翼泛化性能。完整快照和所有原框保持不变。

{{FIT_METRICS}}

### 4.3 同帧 FP32 / ONNX / INT8 检查

以下均由 RKNN Toolkit2 2.3.2 宿主机运行，不是板端精度。每段均匀抽取 100 帧；固定翼监控帧排除出校准清单。小样本结果不代替全量量化精度评测。

Video00004 + Video00009 共 200 帧：

{{QUANT_NATIVE_METRICS}}

新增训练来源监控 100 帧，单独统计，不并入上述保留集：

{{QUANT_FIT_METRICS}}

详细指标、分尺度 Recall、conf 0.01/0.03/0.05 结果和量化原始记录见 `metadata/`。量化报告中的 `pooled_native` 只合并 Video00004 和 Video00009，不包含新增训练拟合帧。

原始量化报告的 `rebuilt_matches_delivered` 比较的是检查前的 reference 编译文件，而非本交付包。包内模型取自完成检查的 `rebuilt_for_simulator.rknn`，SHA256 等于报告的 `rebuilt_sha256`，并记录在 `FINAL_ARTIFACT.json`，没有把未测试的 reference 二进制交付。
