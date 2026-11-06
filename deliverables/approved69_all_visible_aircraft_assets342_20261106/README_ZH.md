# 全可见光飞机数据与扩展贴图库重训实验

## 实验范围

- 本实验用于内部研究，公共数据及贴图尚未完成统一商用授权核查，不自动替代已有交付模型。
- 审核数据快照时间：2026-11-06 11:13:23，Asia/Shanghai。平台共有 64 段已通过审核的视频，其中 Video00009 只作验证；63 段审核训练视频加 6 段历史标注来源，共 69 段自采训练视频。
- 原有 Anti-UAV300 RGB 混合数据、训练清单中旧样本的重复次数保持不变。69 是自采来源数量，不包含公共数据的视频数量。
- Video00004 不参与训练或权重选择；Video00009 不参与训练，但参与权重选择，因此两者合并指标不能称为完全独立测试指标。
- 只使用可见光 RGB/可见光灰度图像。不加入热红外、音频、未审核自采数据、uncertain/unlabeled 帧或伪标签。
- 补齐当前审核快照中原负样本池遗漏的 16,140 张 absent 帧，加入轮换池，而非全部强制塞进每轮。原正样本曝光不削减，每轮负样本比例保持约 15%。

## 公共数据

下面是导出计数；跨公共集合及与自采验证/测试集的精确解码像素去重后，最终训练计数以新数据目录的 `summary.json`、`public_accepted.json` 为准。

| 数据 | 使用范围 | 训练正帧，去重前 | 隔离规则 |
| --- | --- | ---: | --- |
| DUT Anti-UAV | 官方检测训练集可用正帧 | 5,197 | 官方 val/test 不训练；空标注不推断为负样本 |
| Halmstad DRONE | 可验证时间戳的无人机片段 | 27,038 | 按整段视频 SHA 隔离；时间对齐不可靠片段排除 |
| Halmstad AIRPLANE/HELICOPTER | 新增飞机及直升机正帧 | 19,450 | 同上，另保留 5,296 张验证正帧 |
| UAVSwarm | 人工 GT，非检测器伪标签 | 6,844 | 官方 train/test 隔离；不可靠序列排除 |
| Purdue UAV | 官方 refined v2 人工标注 | 44,182 | 50 段逐帧解码核验；另留 7,652 张整视频验证正帧；插值/无标注帧排除 |
| AOD4 | 只用官方 train 中可核验飞机正帧 | 8,659 | 不导出、不训练、不评测官方 valid/test；歧义类别、鸟类及无法确定来源的 V_* 帧排除 |

无人机、固定翼/飞机和直升机统一作为模型的目标 class 0；不按文件名猜类别，不把鸟或未知类别当作飞机/背景。所有有效尺寸的框保留，边界相交框裁到图像范围，不增加面积上限过滤。

AOD4 官方划分存在同源视频跨集合情况。本轮显式启用 `--aod-train-only`，不重分配官方 valid/test，不报告 AOD4 独立泛化结果。其未使用的 holdout 文件 SHA 元数据单独保存。这不是完成了近重复或跨数据集场景去重的声明。

Det-Fly 官方下载不可匿名访问；Drone-vs-Bird 需签署协议；LRDDv3 暂不可获取。这些数据没有加入，不代签协议或绕过访问控制。“全部”指目前可获取且完成格式、类别及标注对齐核验的可见光数据。

## 在线贴图

- 活跃目录：`/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_catalog_20261106/catalog.json`。
- 共 342 个兼容贴图条目，按透明 PNG 字节去重为 331 张；其中新增固定翼可用贴图 14 张，覆盖 8 个机型名称。
- 新背景索引：`/mnt/andrew/anti_uav_model_refinement/data/real_gray_background_approved69_assets342_20261106`。
- 复用原来经过安全检查的 57,461 个背景条目，只扩展贴图库，不改原背景、原图或原标注；旧目录及旧实验保留。
- 各训练阶段在线替换概率均为 0.50，仅对有安全缓存的自采正样本触发；不可靠替换保留原图。公共 RGB 图像没有对应安全背景缓存，不强行抹除后换图。
- 所有 342 个条目可参与随机选择，并不意味着每张训练图预生成 342 个版本。替换后 bbox 按实际合成目标更新。

## 训练流程

```text
公开 yolov8n.pt
  -> P3 initial，30 epochs
  -> 冻结本轮 P3，只训练 Add-on P2 Teacher，12 epochs
  -> 从本轮 P3 initial 初始化 Student，FP32 蒸馏 15 epochs
  -> 移除训练期辅助 P2/投影层，导出纯三尺度 P3 Student FP32
```

不依赖旧任务 checkpoint，不进行 QAT，不增加 localization 续训阶段。与 69 视频原始对照保持相同网络、损失及权重选择方法。

- 输入：960x544，batch 64，nbs 128，workers 16，seed 20261015。
- AdamW，P3/Teacher 初始学习率 0.001，Student 0.0001，余弦学习率，lrf 0.1，weight_decay 0.0005。
- Mosaic/MixUp/Copy-Paste/旋转/垂直翻转/shear/perspective/额外尺度视图关闭。
- 平移 0.05，随机缩放 0.2，水平翻转 0.5，HSV 0.015/0.4/0.3。
- 仅以 Video00009 的 `0.5*F2@0.03 + 0.3*AP50 + 0.2*AP50-95` 选取各阶段 best；公共 holdout 和 Video00004 不参与选权重。
- 仅允许使用 GPU 2 或 6，可共享，不停止其他人的任务。实际 GPU 及启动前占用记录在 `launch.json`。

## 路径与复核

服务器代码：`/mnt/chenziye/codes/ultralytics_yolov8`。代码先在本地提交并 push，再在服务器 git pull；GitHub SSH 不可达时用该提交的增量 git bundle 执行 fast-forward pull，不用 SCP 覆盖源码。

- 编排脚本：`scripts/anti_uav/run_public_uav_research.py`。
- 训练脚本：`scripts/anti_uav/run_p3_independent_curriculum_training.py`。
- 公共数据转换：`scripts/anti_uav/prepare_public_uav_research.py`、`prepare_uavswarm_research.py`、`prepare_additional_public_aircraft.py`。
- 遗漏负样本提取：`scripts/anti_uav/prepare_missing_approved_negatives.py`。
- 审核快照及核验：`/mnt/andrew/anti_uav_model_refinement/all_aircraft_20261106_audit/`。
- 新公共数据：`/mnt/chenziye/datasets/anti_uav/public_research_20261106/`，复用的 DUT/DRONE/UAVSwarm 数据在同级 `public_research_20261105/`。
- 新训练数据：`/mnt/andrew/anti_uav_model_refinement/data/real_gray_approved69_all_visible_aircraft_assets342_20261106/`。
- 新实验：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved69_all_visible_aircraft_assets342_20261106/`。

`research_status.json` 为准备/训练/比较的总状态；`status.json` 为训练阶段和 epoch；`training.log` 记录原始训练日志；`protocol.json` 固定训练参数及代码版本；`launch.json` 固定完整命令。运行状态以这些实时文件为准，不以本说明推断已训练完成。

最终文件：

- P3 initial：`foundation/p3/weights/best.pt`。
- P2+P3 Teacher：`teacher/p2p3/weights/best.pt`。
- 部署用纯 P3 Student：`p3_student_distilled_best_fp32.pt`。
- 自动比较结果：`comparison/results.json`，训练全部结束并核验完成后才生成最终比较。

比较包含 69 视频原始对照、前一轮公共无人机+328贴图实验、本轮全可见光飞机+342贴图实验，各自的 P3 initial/Teacher/Student。主要合并结果用同一批 Video00004 + Video00009（3,780 帧、1,245 GT），conf=0.03、IoU=0.5、NMS=0.45、FP32、无 tracker；AP 从合并预测重新计算，不平均两个视频的 AP。公共飞机验证正帧另作固定翼诊断，不混入该合并结果，也不代表完整背景误检评估。
