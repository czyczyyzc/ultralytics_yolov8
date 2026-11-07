# 74 视频：回归监督根因修复对照实验

## 启动与对照

47 服务器，已于 2026-11-08 03:56:40（Asia/Shanghai）在 **GPU 3** 启动正式长训练，总编排 PID 2572409、训练 PID 2575466。确认 P3 initial 第 1/30 轮已执行至少 148/5,584 batch，loss 有限、没有训练异常。该信息是启动时快照，不是已训练完，也不是验证指标已有提升。

04:06:56 更新：只替换评测监督进程为 PID 2600602，训练 PID 2575466 未退出或重启，连续执行至首轮至少 2,302/5,584 batch（约 41%）。更新解决冻结贴图库视图没有直接包含公共固定翼 holdout YAML 的路径问题；评测改从已固定的公共数据来源解析，不改训练或对照文件。`supervisor_adoption.json` 固定接管记录。

显卡由空闲检测自动选择：显存使用不超过 512 MiB、可用至少 60,000 MiB、利用率不超过 5%；不使用有其他任务占用但暂时利用率低的卡。原 GPU 2 上的对照轮次未被停止或改动，启动新轮次时正在 Teacher 第 5 轮。

新实验：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved74_regression_repair_v1_20261108/`。

直接对照：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved74_gray_fixedquota_assets361_feishu7_20261107/`。另外保留旧 69 视频原生基线用于参考。只有与 74 视频直接对照的比较，才是数据和原训练配方相同的监督修复实验；不要把与旧 69 视频的比较说成单变量实验。

## 保持不变

- 相同 74 段自采训练视频、Anti-UAV300 和 111,349 张公共可见光飞机正样本，保留所有已纳入的飞机类型和已审核负样本。
- 相同 361 条贴图库、58,466 个安全背景和换贴图尝试概率 0.50，不另加 crop、固定增强图或新贴图。
- 相同 960x544、gray3、亮度 0.7-1.3、平移 0.05、scale 0.2、水平翻转 0.5；原有关闭项不打开。
- 相同 batch 64、固定配额、357,376 槽位/5,584 batch、源数据循环方式；同样 30/12/15 epochs，学习率、优化器、seed 和 checkpoint 选择规则不变。
- 从公开 `/mnt/chenziye/codes/ultralytics_yolov8/yolov8n.pt` 开始：P3 initial -> Frozen-P3 + Add-on P2 Teacher -> 从本轮 initial 初始化并蒸馏 P3 Student FP32。不继承旧私有模型，不用预检更新后的权重，不做 QAT。
- 最终 P3 仍是三尺度原网络；没有改输入分辨率、推理框大小、conf、NMS、评测 IoU 或 Tracker，没有新增部署计算节点。实际导出 FPS 仍须模型完成后验证，不预先保证数字。

数据 YAML 直接使用对照组的同一文件，`experiment_contract.json` 固定 YAML、图库、背景 index、公共来源清单及公开权重 SHA。没有就地编辑数据或对照配置。

## 具体修复

新增损失为显式 opt-in；不启用 `--regression-repair` 的旧行为保持原样。

1. **分摊多框／多匹配点监督。** 一个正样本图像获得一份 credit；同图多个 GT 分摊 credit，同一 GT 的多个正匹配点再分摊。再按自采小于 4 px、4-8 px、其他自采、Anti-UAV、公共数据分桶；尺度使用增强后的实际 GT 尺寸。桶预算来自本批图像 credit，不手造新的来源比例。
2. **质量加权的来源／尺度归一化。** 组内 GT 保留匹配质量差异，但不让“框多／匹配点多”天然取得成倍预算。缺少有效匹配的组不凭空得到回归框，也不把预算转送到其他来源。
3. **保守渐进混合。** 第 1 轮为原损失；第 2、3 轮逐步提高修复比例，第 4 轮达到原回归项 50% + 均衡回归项 50%。分类 BCE、Task-Aligned 匹配器本身、原 tiny 分类倍率保持不变。
4. **真实 GT 的小框贴合约束。** 将额外边缘监督下限由 4 px 扩到 2 px，2-16 px 增加对称的 log 宽高相对误差。过大和过小的框都会受罚，不是盲目缩框；普通和大目标仍保留原 IoU/DFL 监督，所有 GT 保留，不加面积过滤。
5. **实际覆盖监控。** 每轮按桶记录 GT、已分配 GT、没有网格内部点的 GT、正匹配点、assignment-score 和归一化回归量。P3 initial、Teacher P2 分支、Student 主分支与训练期辅助 P2 分支分别记录。

2-4 px GT 可能没有任何 P3/P4/P5 网格点在框内，这是保持原部署结构时仍然存在的限制。本轮不使用伪造匹配框或更改 GT 的办法强行消除它；监控会明确暴露这一部分，现有 P2 辅助和蒸馏保留。因此修复了监督权重机制不等于保证每个微小目标都能回归，精度收益须以训练后同口径评测为准。

阶段参数纠正：对照 P3 initial 实际 tiny=1.75、额外定位范围 4-64 px；Teacher tiny=1.75、原额外边缘项为 0；Student 主分支 tiny=1.75、额外定位范围 4-960 px。新代码逐项读取各阶段原参数，再施加同一修复，而不是误用类默认值。上一份审计报告中将这些阶段混淆的描述及相应倍率换算已更正；原始复测和探针数据不变。

## 验证与剩余风险

40 项测试通过，涵盖多 GT／多 anchor 预算不变性、缺失匹配不转移预算、空背景批次、严格网格覆盖、首轮 loss 与旧实现完全一致、有限反向梯度、冻结图库视图的固定翼评测路径，以及 Student 去掉训练分支后与纯 P3 输出逐项一致。

另在 GPU 3 运行 4 个真实增强批次，实际输入 544x960、batch 64、AMP，并直接开启完整修复强度；loss 和反向传播正常。`preflight/acceptance.json` 固定实际形状、强度和分桶统计。预检权重全部丢弃；正式训练重新加载公开权重。

监控的 CUDA 加权 bincount 可能提示浮点归约并非逐 bit 确定；这些监控值不参与 loss 或优化。协议固定 seed、数据和配方，不承诺不同 GPU 训练权重逐 bit 相同。

本轮是有对照的新实验，不是已证明涨点的交付模型。若小目标匹配率／宽高偏差不改善，或公共固定翼明显退化，不会因为“新模型”就覆盖旧交付。

## 查看与复现

同一实验目录中：`repair_launch_status.json` 是编排状态，`status.json` 是实际阶段/epoch，`training.log` 是训练日志，`protocol.json` 固定损失开关和全部阶段，`repair_launch.json` 固定命令和选卡前状态。

每阶段的 `regression_repair_audit/epoch_XXXX.json` 保存监督审计，例如 `foundation/p3/regression_repair_audit/epoch_0001.json`。这里记录的是匹配权重和归一化回归预算，不是实测梯度范数。

源码位于 `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/`：`balanced_regression_repair.py` 是损失与监控，`regression_repair_trainers.py` 适配三个阶段，`preflight_regression_repair.py` 做真实批次预检，`run_regression_repair_experiment.py` 负责选空卡、冻结对照数据、启动与最后评测。

训练完成且对照组完成后，会自动复测旧 69 视频、74 视频原损失和修复版各自的 P3 initial／Teacher／Student。统一 gray3、FP32、960x544、conf=0.03、NMS IoU=0.45、匹配 IoU=0.5；输出 Video00004、Video00009、二者合并 AP/计数，以及 3,451 张公共固定翼留出正帧。Video00009 用于选择 best，合并集不宣称完全独立测试；固定翼正帧也不替代完整视频的背景误检测试。

结果将写入本实验的 `comparison/results.json`；目前尚不存在最终对比指标。后续 RKNN 导出和板端延迟/FPS 不在当前训练阶段冒报。
