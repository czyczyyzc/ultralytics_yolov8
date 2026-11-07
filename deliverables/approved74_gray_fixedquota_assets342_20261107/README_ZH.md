# 74 视频全灰度固定配额训练记录

## 数据范围

本实验在此前 69 段自采训练视频的冻结清单上，追加用户指定的 5 段已审核视频。全部公共可见光飞机数据与原有 Anti-UAV300 数据保留，Video00004 测试与 Video00009 验证不加入训练。不是使用旧任务权重微调，仍从公开 yolov8n.pt 开始。

| 原始名称 | SHA256 前缀 | 正样本帧 | 负样本帧 | 排除 uncertain |
| --- | --- | ---: | ---: | ---: |
| 10.28 am11 晴天 尾随1/00001.ts | aea427c9af60 | 293 | 1,505 | 2 |
| 10.28 am11 晴天 尾随1/00006.ts | 68e1e0232412 | 939 | 861 | 0 |
| 新导入 00006.ts | af056ab6ef0a | 596 | 0 | 0 |
| 2026-11-3 PM1510 阴天 尾随/00005.ts | 45c11751e1d8 | 689 | 121 | 0 |
| 2026-11-3 PM1530 阴天 尾随/00005.ts | 7aa6b9430f2e | 745 | 65 | 0 |

5 段共 5,816 帧；逐帧解码导出 5,814 张有效训练图，2 张 uncertain 排除。当前 approved revision 与冻结版本必须一致，旧训练来源若重新审核或改版本则停止，不静默复用旧标签。原始图片、标签及旧实验不覆盖。

## 贴图增强

复用已有 342 个贴图条目，不重复抠图。需要补建的是新视频的安全背景缓存：只有正样本进入背景修复流程，已审核负样本可作为背景邻帧，但不会被贴成新的正样本；uncertain/unlabelled 不作邻帧。MPEG-TS 的邻帧从顺序导出的冻结图像读取，不随机跳转视频猜帧号。

安全替换概率保持 0.50。背景配准、局部一致性、轮廓、目标尺寸或边界条件不可靠时，原图和原框仍完整参加训练，不强行替换、不删除大目标。旧背景 pack 以只读链接复用，新背景按视频 SHA 和 frame 追加。完成后先检查合成图、更新后的 bbox 及干净裁剪放大图，再放行训练。

本轮背景准备已完成：3,262 张新增正样本中，1,005 张有可用背景，2,257 张仅使用原图；合并旧缓存后共有 58,466 个可用背景。342 个贴图条目对应 331 个不同 PNG，不代表 342 种互不重复的外观。已逐张查看五段原始 GT 的首/中/末正帧预览，以及五张最终替换局部图；放大后个别样本仍有轻微背景修补痕迹，不保证完美融合。验收记录为 `audit/visual_review.json`，合成检查为 `audit/composition_acceptance.json`。

## 训练配置

每个 batch 64，来源及名义尺寸配额始终固定：自采 4-8 px 19 张、小于 4 px 3 张、其他自采正样本 16 张、Anti-UAV300 正样本 7 张、公共正样本 9 张、已审核负样本 10 张。公共 9 张进一步固定为 DUT 1、Halmstad 3、UAVSwarm 1、Purdue 3、AOD4 1。

短池重新洗牌循环，保留旧清单的正样本曝光下限，额外回放有上限；公共数据分别轮换，完整覆盖需要多个 epoch，而非削减或删除公共数据。训练前重放整个公共覆盖周期，逐 batch 检查配额和原正样本覆盖，记录在 `audit/sampler_acceptance.json`。

本轮已通过三个 epoch 的采样重放验收：每轮 357,376 个槽位、5,584 个完整 batch，固定配额不随 epoch 尾部漂移；111,349 张公共正样本在三个 epoch 内全部覆盖。负样本槽位占 15.625%，并非精确 15%。这些是采样曝光次数，不等于去重后的图片数或逐轮访问全部候选图片。

输入固定 960x544，所有输入在线转为三通道相等的灰度图。仅亮度增强：hsv_h=0、hsv_s=0、hsv_v=0.3，即亮度增益 0.7-1.3。平移 0.05、随机缩放 0.2、水平翻转 0.5 保留；Mosaic/MixUp/Copy-Paste/旋转关闭。验证、最终比较同样灰度化，但不做随机增强。

流程为：公开 yolov8n.pt -> P3 initial 30 epochs -> Frozen-P3 + Add-on P2 Teacher 12 epochs -> 从本轮 P3 initial 初始化 P3 Student，FP32 蒸馏 15 epochs -> 导出纯三尺度 P3。batch 64，nbs 128，workers 16，seed 20261015。不启用 QAT 或额外 localization 续训。

## 服务器位置

- 代码：`/mnt/chenziye/codes/ultralytics_yolov8`。
- 编排：`/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/run_approved_gray_fixedquota_refresh.py`。
- 实验：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved74_gray_fixedquota_assets342_20261107`。
- 数据：`/mnt/andrew/anti_uav_model_refinement/data/real_gray_approved74_all_visible_fixedquota_20261107`。
- 训练入口 YAML：数据目录的 `train_fixed_gray.yaml`，仅在准备验收完成后生成。
- 背景：`/mnt/andrew/anti_uav_model_refinement/data/real_gray_background_approved74_assets342_20261107`。
- 贴图库：`/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_catalog_20261106/catalog.json`。

`refresh_status.json` 为冻结/解码/缓存/等待显卡/训练/比较的总状态；`status.json` 为训练阶段和已完成 epoch；`training.log` 记录训练输出；`launch.json` 记录实际 GPU、启动前占用、完整命令。准备阶段不能当作已启动 GPU 训练。

GPU 仅允许 2 或 6，不停止其他任务。显式允许共享时，启动前要求已有占用不超过 32 GiB 且剩余不少于 40,000 MiB，以给蒸馏阶段留出余量；否则等待。此共享上限与此前 16 GiB 的旧启动器不同，实际选择会记录在 launch.json。

## 产物与比较

训练全部完成后才会生成最终候选，不应提前把预期路径当作已交付文件：

- P3 initial：实验目录 `foundation/p3/weights/best.pt`。
- Teacher：实验目录 `teacher/p2p3/weights/best.pt`。
- 纯 P3 Student：实验目录 `p3_student_distilled_best_fp32.pt`。
- 三方统一比较：实验目录 `comparison/results.json`。

比较包含旧 69 视频基线、之前 69 视频全公共数据版、本轮 74 视频固定配额全灰度版，各自三个阶段。所有模型均使用同一灰度输入，conf=0.03、NMS IoU=0.45、匹配 IoU=0.5、FP32、无 Tracker。Video00004 + Video00009 的 3,780 帧/1,245 GT 合并计算 AP；该集合含用于选择权重的 Video00009，不宣称完全独立测试。公共固定翼留出正帧单独报告，不能代替完整视频背景误检测试。

公共数据与贴图库仍属于研究实验范围，商业交付前需要核查授权。修复后的采样契约正确不等于精度已恢复，必须等最终同口径比较，不自动替换已有交付模型。
