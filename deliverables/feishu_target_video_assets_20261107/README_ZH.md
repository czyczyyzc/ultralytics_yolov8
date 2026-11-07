# 四段实拍靶机视频：去吊线贴图与重训

## 处理结果

2026-11-07 从四段 1280x720、约 30 FPS 的实拍视频顺序解码，抽出 70 张候选。初选 16 张做分割，再复核 10 张去线结果，最终只接受 7 张。它们是同一 Beluga 型靶机的不同视角，不是 7 种新机型，也不是经过标定的精确朝向类别。

| 原视频（本地均在 `/Users/czyczyyzc/Downloads/`） | 素材 ID | 原视频帧号（从 0 开始） | 视角 |
| --- | --- | --- | --- |
| 飞书20261107-170212.mp4 | 432、439 | 75、285 | 斜下方、近侧下方 |
| 飞书20261107-170223.mp4 | 442、449 | 60、375 | 另一侧斜下方、迎面下方 |
| 飞书20261107-170227.mp4 | 464、473 | 165、435 | 两侧近侧下方 |
| 飞书20261107-170233.mp4 | 475 | 15 | 斜下方 |

原视频 SHA256、解码时间戳、精确帧号、裁剪坐标和裁剪 PNG 校验值均保存在来源清单。没有跳帧 seek 后猜帧号，没有缩放源图像素。

## 吊线如何去除

沿用服务器分割模型，只估计透明度 alpha；飞机的原始 RGB 不重画。针对本批明确复核的刚性固定翼，用 7 px 开运算得到机体主体，再外扩 6 px 作为保留区域，与原 alpha 相交，去掉机体外的细吊线。使用主体形状而不是凸包，避免把机翼与尾翼之间的吊线保留下来。原 alpha 的机体内部空洞不会被填满。

清理仅作用于本批指定素材，不全局应用到原有旋翼/螺旋桨图库。去除 alpha 超过 8% 会报错要求复核，这只是素材清理安全检查，不是目标 bbox 面积过滤。没有添加检测框面积限制，也没有用生成式补画恢复被线遮挡的机体。

最终候选 483 有绿光边缘，486 有尾部附近的吊顶配件，495 有机头附近的吊顶配件，均未发布。其余早期候选中的灯、管道、断翼和重复视角也没有进入新库。不能保证所有线与机体接触点都像素级完美，但接受的 7 张逐张棋盘格/深色背景预览中没有明显机体外吊线或吊顶配件。

原始四段视频不变。透明 PNG 的透明区仍可保留原背景 RGB，用于验证没有改原像素；不支持 alpha 的看图软件可能显示那些隐藏像素，应看棋盘格/深色预览，训练读取真实 alpha。

## 合并、合成验证与训练

已发布：旧 354 条 + 新 7 条 = **361 条记录、350 张不同 PNG**。旧图库及旧 PNG 校验值保持不变。四段新视频仅作贴图来源，没有完整逐帧标注，因此自采训练视频数仍是 **74**，不是 78。

7 张素材各做小、中、大三档，共 **21** 次真实缓存背景合成，bbox 跟随最终机型轮廓更新；检查原图未被改写、编辑区域外像素不变、bbox 合法、无测试视频泄漏。逐张查看了全部 7 张去线预览、全部 7 张中尺度合成总览，并单独查看 475 大尺度与 449 小尺度案例。检查不是精度评测，也不能保证任意背景完美融合。33 项回归测试通过。

经用户明确批准，整批验收后停止旧轮次的独立进程组 514327，确认无存活成员，保留日志及权重。新轮次已于 **2026-11-07 17:33:44（Asia/Shanghai）** 在获准共享的 **GPU 2** 启动；总编排 PID 985709，训练 PID 985787。已实际进入 P3 initial 第 1/30 轮并观察到至少 30/5,584 个 batch，loss 有限、无报错。这不是训练已完成，也不代表精度已提升。

流程不变：公开 `/mnt/chenziye/codes/ultralytics_yolov8/yolov8n.pt` -> P3 initial 30 epochs -> Frozen-P3 + Add-on P2 Teacher 12 epochs -> 从本轮 P3 initial 初始化并蒸馏 P3 Student FP32 15 epochs。不恢复停止轮次权重，不启用 QAT。

保留全部 74 段原自采训练视频、Anti-UAV300、111,349 张公共可见光飞机正样本，以及原训练清单/重复曝光。Video00004 不参与训练；Video00009 仍用于选择权重，因此它不是从未用于模型选择的独立最终测试集。

输入 960x544，全灰度三通道，亮度增益 0.7-1.3，平移 0.05、缩放 0.2、水平翻转 0.5；Mosaic、MixUp、Copy-Paste、旋转、透视继续关闭。对有安全背景的正样本以 0.50 概率尝试换机型，按条目均匀抽取，失败保留原图；不修改 58,466 个安全背景缓存。

每批 64 张固定配额：自采 4-8 px 19、自采小于 4 px 3、其他自采正样本 16、Anti-UAV300 7、公共正样本 9、负样本 10。整轮 357,376 槽位、5,584 批，来源循环采样；公共来源每批 DUT 1、Halmstad 3、UAVSwarm 1、Purdue 3、AOD4 1，三个 epoch 内覆盖公共池。只换贴图库，不改训练采样和其他增强。

## 47 服务器位置

服务器：`root@47.107.185.207`。

- 原视频：`/mnt/andrew/anti_uav_model_refinement/data/feishu_target_video_sources_20261107/videos/`。
- 70 张候选及来源：`/mnt/andrew/anti_uav_model_refinement/data/feishu_target_video_sources_20261107/probe/`。
- 最终去线候选（含 3 张拒绝项，不直接训练）：`/mnt/andrew/anti_uav_model_refinement/data/feishu_target_video_candidates_threadfree_20261107/`。
- 合并图库：`/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_catalog_feishu7_20261107/catalog.json`。
- 同目录 `catalog_unique.json` 是文件级去重库；`new_assets.json` 是新增 7 张；`visual_review.json` 固定接受/拒绝项及理由。
- 新增素材包：`/mnt/andrew/anti_uav_model_refinement/data/feishu_target_assets_new7_20261107.tar.gz`，只含新增 7 张 PNG 及来源/审核，不是完整 361 条库。
- 训练 YAML：`/mnt/andrew/anti_uav_model_refinement/data/real_gray_approved74_fixedquota_assets361_20261107/train_fixed_gray.yaml`。
- 新缓存视图：`/mnt/andrew/anti_uav_model_refinement/data/real_gray_background_approved74_assets361_20261107/`。
- 新实验：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved74_gray_fixedquota_assets361_feishu7_20261107/`。

新实验目录下 `refresh_status.json` 是总状态，`status.json` 是阶段/epoch，`training.log` 是实际训练日志，`launch.json` 是精确命令，`foundation/p3/sampler_audit.json` 是固定配额；`audit/composition_QA.json` 和 `audit/composition_review.json` 是本批验收与校验值。

## 脚本与本地文件

代码根目录：`/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/`。

- `prepare_video_asset_candidates.py`：顺序解码、冻结 ROI 和来源。
- `import_fixedwing_assets.py segment/publish`：分割、指定 alpha 去线、仅发布审核通过项。
- `configs/feishu_target_videos_20261107_threadfree.json`：本批去线参数；`configs/feishu_target_videos_20261107_review.json`：最终 7 张接受项。
- `verify_asset_refresh_composites.py`：每张三档合成和 bbox 检查。
- `run_reviewed_asset_refresh.py`：冻结标签版本、绑定新图库，验收后完整重训。

本地目录：`/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/feishu_target_video_assets_20261107/`。`threadfree/previews/` 可看原图/棋盘格/深色对比，但其中 483、486、495 已拒绝。只使用发布后的素材包。`composition_QA/` 是 21 张最终合成结果和总览；`preparation.json`、`protocol.json`、`launch.json`、`sampler_audit.json` 固定真实配置。

这批靶机外观已用于训练适配，不能再称为未见机型测试。交付前需确认源视频和机型图片的使用授权。
