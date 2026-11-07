# 靶机照片贴图补充

## 可见表面与视角

13 张原图全部冻结保存。最后四张照片虽然由相机从上方拍摄，但飞机本身已经翻过来，露出的是机腹；可以作为真实机腹素材，不需要做“俯拍变仰拍”的生成式处理。第 1 张产品图也是真实可见的机腹方向。

普通二维旋转/镜像不能改变可见表面；单应矩阵是平面到平面的投影映射，也不能恢复未拍到的机身底面或大幅改变遮挡关系。参见 [OpenCV 单应变换说明](https://docs.opencv.org/4.11.0/d9/dab/tutorial_homography.html)。温和的平面透视只能近似扰动已有视角；真实的大角度侧仰视应靠实拍或可靠 3D 模型渲染，不把虚构细节当真实结构。本次没有改动全局旋转、透视或其他训练几何参数。

## 筛选与抠图

| 用户图片顺序 | 素材 ID | 可见表面 | 处理 |
| --- | --- | --- | --- |
| 1 | 417 | 机腹 | 保留真实轮子、螺旋桨及机腹细节 |
| 2 | 418 | 上方斜视 | 按真实机背视角使用 |
| 3 | 419 | 上方斜视 | 按真实机背视角使用 |
| 4 | 420 | 侧面 | 保留真实起落架 |
| 5 | 421 | 上方斜视 | 按真实机背视角使用 |
| 6 | 422 | 上方 | 去掉地面/阴影的 alpha |
| 7 | 423 | 上方斜视 | 去掉地面的 alpha |
| 8-12 | 不加入完整机型库 | 翼尖/尾尖碰边、超界或疑似截断 | 保留原图，不补造缺失机体 |
| 13 | 429 | 机腹，倒放实拍 | 四个明确复核的多边形去掉外露支架 alpha |

8 张进入最终候选：2 张机腹、1 张侧面、5 张机背/上方斜视，是同一机型系列的视角素材，不是 8 种新机型。型号标签来自图片可见的 ATOMRC/Beluga 文字，不代表不同涂装/版本已由厂家验证。

沿用固定版本 BiRefNet `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`，只估计 alpha，不重新生成 RGB。429 初次结果包含支架，修正只对原图坐标中明确标出的支架区域置零 alpha；机身和机翼核心保护区的 alpha 不变，原图 RGB 不变。没有补洞、重绘、删除真实舵机/轮子或制造不可见机腹。

PNG 的透明区域仍保留原 RGB，用于验证未改原像素；部分不处理透明通道的查看器可能显示地面或支架，最终效果应看棋盘格/深色背景合成预览，训练读取真实 alpha。细螺旋桨、柔和白边不保证像素级完美。

## 合并与训练

复用已有 346 条库（含上一批 4 张视频靶机素材），已发布为 354 条、343 张不同 PNG，新增 8 张没有发现文件级重复。保留旧素材、数据清单、重复曝光、全部公共可见光飞机、Anti-UAV300 及 74 段自采训练视频。此次提供的是贴图照片，不增加训练视频计数。

仍仅对有安全背景的正样本以 0.50 概率尝试替换，机型条目均匀选择，bbox 根据最终 alpha 同步更新；不增加负样本，不把地面、支架或白色底图整体贴入训练。沿用 58,466 个可用背景，不重新计算或修改旧背景 pack。

用户已选择整批验收后停止当前 `approved74_gray_fixedquota_assets346_target4_20261107`，用新库从公开 `yolov8n.pt` 完整重训；不是接着旧模型微调。流程仍为 P3 initial 30 epochs -> Frozen-P3 + Add-on P2 Teacher 12 epochs -> 从本轮 P3 initial 初始化并蒸馏 P3 Student 15 epochs；不启用 QAT。

输入 960x544、全灰度三通道、亮度增益 0.7-1.3、平移 0.05、缩放 0.2、水平翻转 0.5；Mosaic/MixUp/Copy-Paste/旋转/透视维持关闭。每批 64 张固定配额仍为自采 4-8 px 19、自采小于 4 px 3、其他自采正样本 16、Anti-UAV300 7、公共正样本 9、负样本 10。Video00004/Video00009 不进入训练，后者仍用于选择权重。

启动前须通过本批 8 张贴图的小/中/大三档共 24 次合成检查，并保存与 catalog SHA 匹配的可视验收。检查是合成/bbox 验证，不是检测精度或完美融合保证。商业使用前仍需核查源照片使用权。

24 次合成检查已全部通过；已查看全部 8 张最终贴图预览、8 张中尺度合成预览，以及 417/429 两张机腹素材的大尺度合成。429 在复核画面中没有明显支架轮廓；亮色飞机在原目标对比度很低的背景上仍然偏淡，不人为提高对比度来美化预览。原 346 条轮次的独立进程组 311149 已退出且无存活成员，日志和数据保留。相关 30 项回归测试通过。

已于 2026-11-07 14:48:49 (Asia/Shanghai) 在 GPU 6 启动新轮次，总编排 PID 514327、训练 PID 514415。P3 initial 第 1/30 轮已实际执行到 35/5,584 batch，loss 有限、暂无报错；不是第一轮已完成。实际采样器仍为固定配额，每轮 357,376 个槽位，111,349 张公共正样本在三个 epoch 内轮换覆盖。Foundation 固定为公开 `yolov8n.pt`，没有使用停止轮次的任务权重。

## 服务器位置

服务器：`root@47.107.185.207`。数据根目录：`/mnt/andrew/anti_uav_model_refinement/data`。

- 原图与来源：`beluga_photo_assets_sources_20261107/inputs/`、`provenance.json`。
- 首轮抠图：`beluga_photo_asset_candidates_20261107/`，包含支架残留，不要用于训练。
- 最终候选：`beluga_photo_asset_candidates_refined_20261107/`。
- 合并图库：`fixedwing_asset_catalog_target4_beluga8_20261107/catalog.json`。
- 完全去重库：同目录 `catalog_unique.json`；本批 8 张：同目录 `new_assets.json`。
- 背景视图：`real_gray_background_approved74_assets354_20261107/`。
- 训练 YAML：`real_gray_approved74_fixedquota_assets354_20261107/train_fixed_gray.yaml`。
- 新实验：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved74_gray_fixedquota_assets354_beluga8_20261107/`。

本地本目录已提供 `beluga_photo_assets_new8_20261107.tar.gz`（仅本批 8 张透明 PNG 和来源/审核清单，不是完整 354 条库）、`refined/cutouts/`、`composition_QA/new_asset_contact_sheet.jpg`、`composition_QA/composition_contact_sheet.jpg` 和 24 张单独合成预览。`asset_429_large.jpg` 是倒放实拍机腹的灰度替换示例，`asset_417_large.jpg` 是机腹产品图示例。服务器新增 PNG 包在数据根目录的同名压缩包。`protocol.json`、`launch.json`、`preparation.json`、`new_assets.json`、`summary.json` 与合成验收也已下载到本地，方便核对。

代码位于 `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/`：`stage_photo_asset_inputs.py` 冻结原图，`import_fixedwing_assets.py segment/publish` 抠图与发布，`configs/beluga_photos_20261107_*.json` 固定源图/ROI/多边形/审核，`run_reviewed_asset_refresh.py` 创建只改变贴图库的新实验并启动，`verify_asset_refresh_composites.py` 检查逐张合成。

新目录 `refresh_status.json` 是总状态，`status.json` 是实际阶段，`training.log` 记录 batch，`launch.json` 固定启动命令/显卡占用，`audit/composition_review.json` 固定合成验收。准备完成不代表 GPU 训练已开始。
