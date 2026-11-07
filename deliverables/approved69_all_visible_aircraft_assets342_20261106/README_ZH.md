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

## 启动验收

新实验已启动，实际使用 GPU 6，batch 64。初始化 checkpoint SHA256 为 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`；任务专用旧权重初始化为 false。运行协议的代码版本为 `aa06e277a3e82f347dc9fc458fbfc03adff570f7`，后续说明/QA 图片提交不改变这份运行代码。

最终公共正样本 111,349 张：DUT 5,193、Halmstad 46,481、UAVSwarm 6,834、Purdue 44,182、AOD4 8,659。111,370 个候选正样本中排除了 21 个精确像素重复项，另有 3 个空标注 DUT 条目排除，不转作负样本。

候选清单共 421,721 条，包含保留的旧重复次数和完整负样本轮换池；不是每轮全部读取。每轮 353,513 槽位，其中正样本 300,486、负样本 53,027，负样本占比约 15%。数据加载核验为 0 corrupt。

公共固定翼额外诊断保留 3,451 张正样本/3,451 个 GT，只用于训练结束后的比较，不参与 checkpoint 选择。数据转换、划分、负样本补齐及混合清单共 151 项测试在本地和服务器通过。

标注可视化抽查：

- `purdue_aircraft_v2_20261106.jpg`：Purdue Clip 1/10/50 的首、中、末正样本，画出实际导出标签。
- `aod4_official_train_aircraft_20261106.jpg`：飞机、直升机、无人机各 3 个官方 train 正样本，画出实际导出标签。两张 Anti-UAV 来源的灰度样本还与对应 visible/RGB 原始帧比对，确认不是 IR 帧。

这只是启动验收，不代表训练已完成或性能已提升；最终指标以 `comparison/results.json` 为准。

## 2026-11-07 退化排查与修复

原全公共数据实验已于 2026-11-07 08:23 完成。以下是只读审计、独立诊断和新配置准备，不是覆盖旧实验，也不是声称修复版精度已经恢复。

已核验旧清单没有删减、旧标签改变数为 0、验证清单和初始化权重相同，三阶段实际训练参数差异为 0；冻结 P3 和纯 P3 导出回归检查均为 bit-exact。此次退化不是 RKNN 量化或 Tracker 导致。

实际发现的变化：

- 小目标来源被稀释：4-8 px 正样本槽位占比由 28.14% 变为 20.86%，尽管旧正样本绝对次数保留。
- 旧轮询采样会耗尽短来源，导致阶段性偏移。新实验灰度来源整轮占 58.38%，最后 10% 为 19.61%，最后 1% 为 14.56%；均匀顺序重排反事实将各十分位灰度比例保持在约 58%。这证明了顺序缺陷，但尚未通过仅改顺序的重训证明其精度贡献。
- 灰度来源内部的负样本比例从 17.63% 增至 25.48%。全局约 15% 不等于每个域都平衡，新增公共数据主要为正样本。
- 当前 Student 的 CPU FP32 逐 GT 诊断：Video00009 有 139 个旧版可匹配、新版不能匹配的 GT，同时新增 9 个匹配；139 个中有 90 个 4-8 px、47 个小于 4 px。108 个属于预测框 IoU 在 0.3-0.5，21 个属于低置信度，10 个在 conf>=0.03 下没有 IoU>=0.3 的框。失败的 IoU 匹配不等于画面上完全没有预测框。
- 在同一批 3,451 张公共固定翼正帧统一转灰度后，旧 Student 的 R/P/AP50 为 5.82%/9.20%/4.95%，原全公共数据 Student 为 50.94%/80.83%/69.33%。新版彩色输入时 Recall 为 74.47%，说明公共数据训练和真实灰度输入仍有差异。该正帧诊断不能代替完整视频背景误检评估。

### 固定配额与循环

新策略 `fixed_domain_v1` 按 batch 64 固定以下数量；所有完整 batch 一致，而不是只保证整轮平均值。

| 分组 | 每 batch | 每 epoch |
| --- | ---: | ---: |
| 自采 4-8 px 正样本 | 19 | 104,956 |
| 自采小于 4 px 正样本 | 3 | 16,572 |
| 其他自采正样本 | 16 | 88,384 |
| Anti-UAV300 正样本 | 7 | 38,668 |
| DUT 正样本 | 1 | 5,524 |
| Halmstad 正样本 | 3 | 16,572 |
| UAVSwarm 正样本 | 1 | 5,524 |
| Purdue 正样本 | 3 | 16,572 |
| AOD4 正样本 | 1 | 5,524 |
| 已审核负样本 | 10 | 55,240 |

每轮固定 5,524 个完整 batch / 353,536 槽位。名义尺寸按随机几何增强前的 960x544 投影长边分组，scale=0.2 后单个目标尺寸可能变化。负样本实际占比为 15.625%，已明确写入配置，不再用旧的 15% 声明掩盖配额差异。

自采短池用完后重新洗牌循环，不再退出。旧 189,137 个正样本清单槽位及原重复次数每轮全部保留，额外回放每个旧索引最多一次。公共子数据集分别循环轮换，保持每个数据集比例不变，111,349 张公共正样本在 3 轮内全部覆盖；DUT 较短，每轮最多取同一公共索引 2 次。不是删除其余公共数据，也不是声称每轮都遍历全部公共帧。原始数据和标签文件不改变。

### 灰度与亮度

`input_color: gray3` 在线执行 BGR 到亮度再复制到三个相等通道，不覆盖源图片，不改 bbox，不把模型改为单通道网络。训练、验证和测试必须使用一致的颜色开关，旧模型比较时也重新以同样的灰度输入评测，不将旧彩色指标与新灰度指标混比。

新灰度训练模式为 `hsv_h=0, hsv_s=0, hsv_v=0.3`，保留随机亮度增益 0.7-1.3；训练期其他几何增强与 50% 安全贴图替换保持原样。评测不做随机亮度、翻转、贴图或循环抽样。实际训练取样已检查公共图、负样本和换贴图样本，输入均为 uint8、3x544x960、三通道完全相等。

### 配置与证据

服务器配置目录：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/public_sampling_fixedgray_ready_20261107/`。主配置 `quota_gray3.yaml` 同时启用固定比例、公共数据集子配额、全灰度和亮度增强。`order_source.yaml` 仅修顺序，`order_gray3.yaml` 增加灰度，`quota_source.yaml` 仅固定配额，便于隔离变量。

- `preparation.json`：3 轮真实清单重放、各子数据集次数、旧正样本覆盖和公共全覆盖断言。
- `manifest.json`：冻结的 69 视频范围、来源清单 hash 和分策略曝光统计。
- 顺序审计：`runs/anti_uav/public_aircraft_regression_audit_20261107/audit.json`。
- 逐 GT 诊断：`runs/anti_uav/public_gray_miss_diagnostic_20261107/diagnosis.json`。
- 公共灰度诊断：`runs/anti_uav/public_fixedwing_gray3_diagnostic_20261107/results.json`。

代码新增独立策略，旧配置默认行为不变，已按本地 git push / 服务器 git pull 同步。主入口仍是 `scripts/anti_uav/run_p3_independent_curriculum_training.py`；无需旧任务权重，仍可从公开 yolov8n.pt 运行 30/12/15 轮流程。

这些数字证明采样及输入契约正确，不保证新的完整训练一定同时达到旧灰度小目标和新固定翼精度。必须完成训练后以同色、同帧、同阈值比较，再决定是否交付；不得把 Video00004/Video00009 的失败帧挖入训练。
