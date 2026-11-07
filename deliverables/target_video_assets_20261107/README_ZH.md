# 同事靶机视频贴图补充与重新训练

本轮 346 条贴图库训练已按用户后续要求停止，改由合并 8 张新照片的 `approved74_gray_fixedquota_assets354_beluga8_20261107` 从公开权重重训。下文运行信息保留作历史记录；本批 4 张视频素材仍保留在新 354 条库中。

## 筛选结果

输入为同事提供的 `0.mp4`、`1.mp4`、`2.mp4`，均为 1280x720、30 FPS 的显示屏翻拍。先查看完整时间线缩略图，再逐帧导出 128 张初筛 ROI 和 47 张近距/偏离十字的 ROI，共 175 张候选画面；10 张进入分割，最终选择 4 张。原始视频和裁剪 PNG 保留，不重新生成飞机 RGB。

| 素材 ID | 视频 | 0-based 帧号 | 时间 | 画面描述 |
| --- | --- | ---: | ---: | --- |
| 384 | 1.mp4 | 348 | 11.60 s | 近距离斜向机腹 |
| 406 | 2.mp4 | 504 | 16.80 s | 较小的倾斜视角 |
| 409 | 2.mp4 | 510 | 17.00 s | 展翼轮廓更宽的相邻视角 |
| 416 | 2.mp4 | 838 | 27.93 s | 近距离侧向掠过 |

这是同一靶机的 4 个视角素材，不是 4 种新机型。`0.mp4` 的完整机体样本受十字、跟踪框或图传花屏影响，没有强行取图。406/409 的源图分辨率较低且有轻微亮边，不代表高清、无伪影或经确认的迎面/左侧/右侧姿态标签。

BiRefNet 沿用固定版本 `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`，只预测 alpha。416 另外用明确复核的矩形清除机体上方独立蓝色 HUD 的 alpha，机体保护区域的 alpha 不变，所有 RGB 像素不变；没有修补、重绘或补造被遮挡的机翼。

合并后为 **346 个兼容旧 ID 的条目，335 个完全不同的 PNG**，旧 342 条目全部保留。训练仍只对有安全背景的正样本以 0.50 概率尝试换机型；新旧贴图均匀选择，合成时更新 bbox。不增加负样本，不删除大目标，不把屏幕画面或未标注整段视频作为训练图片。

## 服务器路径

服务器：`root@47.107.185.207`。数据根目录：`/mnt/andrew/anti_uav_model_refinement/data`。

- 原视频：数据根目录 `target_video_assets_sources_20261107/videos/`。
- 原像素候选与帧号来源：`target_video_assets_sources_20261107/closepass/inputs/`、`provenance.json`。
- 首轮 10 张分割候选（含排除项）：`target_video_asset_candidates_20261107/`，不要整库用于训练。
- 最终 4 张候选及透明 PNG：`target_video_asset_candidates_refined_20261107/`。
- 合并全库：`fixedwing_asset_catalog_target4_20261107/catalog.json`。
- 去重全库：`fixedwing_asset_catalog_target4_20261107/catalog_unique.json`。
- 新增 4 张清单：`fixedwing_asset_catalog_target4_20261107/new_assets.json`。
- 背景缓存视图：`real_gray_background_approved74_assets346_20261107/`，只绑定新增贴图，复用原 58,466 个安全背景及其只读 pack。
- 新训练 YAML：`real_gray_approved74_fixedquota_assets346_20261107/train_fixed_gray.yaml`。
- 新训练目录：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved74_gray_fixedquota_assets346_target4_20261107/`。

候选照片、视频 SHA、0-based 帧号、裁剪坐标及导出 PNG SHA 均记录在 provenance 与 catalog 中，原视频没有覆盖。素材照片来自 `1.mp4` 和 `2.mp4`；这里的裁剪尺寸、缩略图显示放大倍数不等于合成到训练图中的目标尺度。

## 训练边界

用户明确选择停止旧轮次，用新增贴图库重新训练。旧轮次 `approved74_gray_fixedquota_assets342_20261107` 的独立进程组 140714 已停止，确认没有存活成员；日志保留并标记 `stopped_by_user`，不作为继续训练的初始化。

新轮次依旧 74 段自采训练视频，公共可见光飞机与 Anti-UAV300 保留；这三段视频仅作为贴图来源，不改变训练视频计数。Video00004 测试和 Video00009 验证不进训练；二者合并评测包含用于选权重的 Video00009，不宣称完全独立测试。

流程保持：公开 `yolov8n.pt` -> P3 initial 30 epochs -> Frozen-P3 + Add-on P2 Teacher 12 epochs -> 从本轮 P3 initial 初始化 P3 Student、FP32 蒸馏 15 epochs。不启用 QAT，不使用停止轮次或旧交付模型的任务权重。

输入仍为 960x544、全灰度三通道；亮度增益 0.7-1.3，平移 0.05、缩放 0.2、水平翻转 0.5，其余几何/拼接策略不变。每批 64 张固定为自采 4-8 px 19 张、自采小于 4 px 3 张、其他自采正样本 16 张、Anti-UAV300 7 张、公共正样本 9 张、负样本 10 张，保持循环采样。数据清单、重复曝光、采样器、学习率和蒸馏流程均不因补贴图而更改。

训练启动前检查四张新增贴图在小/中/大三种原图目标尺度的 12 次实际合成，更新 bbox 且不改变编辑 mask 外像素；这不是检测精度或无融合痕迹的保证。全部检查通过并保存与新 catalog SHA 匹配的可视验收后，才允许启动。

本轮 12 次检查已完成并查看全部预览；新增透明图没有独立的蓝色 HUD 残留，合成图未见明显矩形补块，但仍不能保证所有在线组合无痕。已于 2026-11-07 13:31:46 (Asia/Shanghai) 在 GPU 6 启动，总编排 PID 311149、训练 PID 311257；P3 initial 第 1/30 轮已执行到 42/5,584 batch，loss 有限且暂无报错。实际采样器仍为固定配额、每轮 357,376 槽位，111,349 张公共正样本三个 epoch 轮换覆盖。相关 28 项测试通过，五个相关脚本通过语法编译检查。

本地本目录提供 `target_video_assets_new4_20261107.tar.gz`（只含 4 张新增 PNG 和来源/审核清单，不是全库）、`refined/cutouts/`、`composition_QA/new_asset_contact_sheet.jpg`、12 张 `composition_QA/asset_*_tiny|medium|large.jpg` 和 `composition_review.json`。服务器相同新增包在数据根目录；新训练目录的 `audit/composition_QA.json` 与 `audit/composition_review.json` 固定绑定合并 catalog SHA。`protocol.json`、`launch.json` 和 `preparation.json` 已下载到本地，供核查具体配置。

这批靶机外观之后属于训练已见外观，后续现场测试可以考察适配效果，但不能称为未见机型泛化测试。原素材使用授权仍需由提供方确认。

## 复现代码

代码位于 `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/`：

- `prepare_video_asset_candidates.py`：顺序解码、按冻结的帧号/ROI 导出原像素和 provenance。
- `import_fixedwing_assets.py segment`：固定分割模型的 alpha-only 抠图；`publish` 只合并明确复核的候选。
- `configs/target_video_assets_20261107_*.json`：候选窗口、原图坐标、分割 ROI、透明度排除区域和可视审核。
- `run_reviewed_asset_refresh.py --prepare-only`：只改贴图库绑定的新数据/背景视图。
- `verify_asset_refresh_composites.py`：逐张新增贴图的小/中/大合成验证与预览。
- `run_reviewed_asset_refresh.py --train-prepared`：验收一致性检查、允许的 GPU 2/6 选择、从公开权重完整重启、完成后同口径比较。

新目录的 `refresh_status.json` 是总状态，`status.json` 是训练阶段，`training.log` 是实际 batch 日志，`launch.json` 保留具体命令和显卡占用。准备完成不能当作训练已启动。
