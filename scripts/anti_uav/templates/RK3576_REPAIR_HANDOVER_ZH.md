# RK3576 修复版 P3 Student 检测与跟踪部署交接

生成时间：{{CREATED_AT}}

## 1. 交付版本与边界

本包为 **74 视频回归损失修复版 P3 Student**，最终 best 为第 **{{BEST_EPOCH}}** 轮。部署输入为宽 `960`、高 `544`，RK3576 专用 W8A8 PTQ INT8；辅助 P2 和投影层已移除，仅保留 stride 8/16/32 的 P3/P4/P5。没有 QAT，也没有新增部署网络计算。

已完成完整训练和 FP32 对比、九输出 ONNX 检查、RKNN Toolkit2 2.3.2 编译及加载、同帧 200 张量化对照。交付 `.rknn` 为实际运行过宿主机验证的二进制。**尚未进行本版本 RK3576 真机精度、FPS、热稳态或曝光到结果延迟验收**；不可引用其他模型或 RK3588 的速度作为本包指标。

本轮包含研究用途公共数据，尚未逐项确认商业授权。该包用于同事集成和验证；商业交付前须核对公共数据、贴图及源码的许可。源码沿用仓库 AGPL-3.0，授权问题不能用模型格式转换解决。

## 2. 路径与校验

服务器：`root@47.107.185.207`，源码 `/mnt/chenziye/codes/ultralytics_yolov8`。

```text
交付目录：{{DELIVERY_DIR}}
训练目录：{{TRAINING_DIR}}
源 FP32：{{SOURCE_PT}}
训练配置：{{DATA}}
```

交付压缩包为交付目录同名的 `.tar.gz`，外部校验文件为 `.tar.gz.sha256`。

| 包内文件 | 用途 | SHA256 |
| --- | --- | --- |
| `models/detector_960x544_rk3576_w8a8.rknn` | RK3576 INT8 模型 | `{{MODEL_SHA256}}` |
| `models/p3_student_best_fp32.pt` | 源纯 P3 权重 | `{{PT_SHA256}}` |
| `models/detector_960x544_rkopt.onnx` | 九输出 RK 优化 ONNX | `{{ONNX_SHA256}}` |
| `source/code.tar.gz` | C++ 推理、跟踪、构建源码 | 见 `SHA256SUMS` |
| `metadata/` | 训练配置、选择记录、精度及量化记录 | 见 `SHA256SUMS` |

源码仓库 `git@github.com:czyczyyzc/ultralytics_yolov8.git`，交付源码固定提交 `{{COMMIT}}`。无关的未提交工作区改动不包含在源码包中。

## 3. 训练来源与修复

从公开 `yolov8n.pt` 开始：P3 initial 30 轮 → 冻结 P3 训练 Add-on P2 Teacher 12 轮 → 从本轮 P3 initial 初始化 Student 并蒸馏 15 轮。无需旧项目 Student 权重。best 选择依据为 Video00009 的 `0.5×F2@0.03 + 0.3×mAP50 + 0.2×mAP50-95`，不是按某个最高 Recall 临时挑选。

训练使用冻结的 74 段自采视频来源、Anti-UAV300 和已接受的可见光公共飞机数据；361 条候选贴图记录对应 350 张不同透明 PNG。所有输入在线转为三通道相同的灰度图，未覆盖原始源文件。安全背景允许时以 0.5 概率尝试换机型；不适合安全替换的正样本保留原图、原框。校准同样为灰度三通道。

固定 batch 64 配额：`{{QUOTAS}}`。各来源循环取样、每批配额固定，不在 epoch 尾部排空自采数据后仅训练公共数据。尺度按 `960×544` 输入下的目标像素计算。实际配置随包保留，不把历史准备清单中的采样审计误作本轮配置。

```json
{{AUGMENTATION}}
```

主要修复为按 GT、来源及尺度平衡回归贡献，并渐进引入微小框边缘/尺寸监督；分类与分配器不变，无伪造 GT 匹配，不改变部署阈值。

```json
{{REPAIR}}
```

`Video00004` 为保留测试；`Video00009` 为选权重验证。两者不用于训练、贴图背景训练缓存或 INT8 校准。合并指标含选择验证集，不是完全独立测试。公共固定翼 holdout 只有正样本，不可用其估计完整背景误检率。

## 4. 精度记录

**全量 FP32 detector-only**：Video00004 + Video00009 合并 3,780 帧、1,245 GT；`conf=0.03`、NMS IoU 0.45、匹配 IoU 0.5。AP 以 conf 0.001 排序全局重算，不是两视频 AP 的平均；不包含 Tracker。

{{FULL_METRICS}}

详情见 `metadata/training/comparison/results.json`，包含单视频、分尺度与固定翼评测。不能将增加数据或损失修复理解为所有指标必然提升，请按真实表格验收。

**量化同帧对照**：每段跨全片均匀抽取 100 帧，共 200 帧，输入像素相同，灰度 `960×544`，阈值同上。这是量化冒烟检查，**不是上表 3,780 帧的完整 INT8 指标，也不是真机测评**。

{{QUANT_METRICS}}

详情见 `metadata/quantization_smoke/REPORT.md`、`results.json` 及 `delivered_binary.json`。最终交付 SHA256 必须与 `evaluated_sha256` 一致；工具重建的模型可能与初始参考模型哈希不同，包内交付的是实际被评测的模型，不是未验证参考文件。

## 5. 输入输出协议

```text
平台：rk3576；Toolkit2 2.3.2，W8A8、normal/channel PTQ
校准：384 张训练来源分层图，gray3，不含 Video00004/Video00009
输入：UINT8 NHWC，宽960×高544×3；三通道相同的灰度像素
归一化：RKNN 内部 mean=0，std=255；外部不要再次除255
缩放：等比例 letterbox，双线性插值，padding=114
类别：0，drone/航空靶机（飞机类型不再分子类）
部署 conf：0.03；NMS IoU：0.45；单帧最多100框
面积过滤：无；不可因目标很大或很小而额外删除检测框
```

不能拉伸图像、重复交换通道或用“隔像素丢点”代替既定缩放。彩色来源先按 BGR 灰度转换再复制三通道；原生灰度直接复制，或使用配套灰度融合预处理。网络输出为 branch-major NCHW，每层 bbox64 / class1 / score-sum1，共九路：

| 层级 | bbox 形状 | class 与 score-sum 形状 |
| --- | --- | --- |
| stride 8 | `1×64×68×120` | `1×1×68×120` |
| stride 16 | `1×64×34×60` | `1×1×34×60` |
| stride 32 | `1×64×17×30` | `1×1×17×30` |

板端后处理须使用每个输出的实际类型、scale、zero-point 反量化，进行 DFL 解码及坐标还原；不能把输出一律当 FP32 或交换九路顺序。RK3588 编译模型不能改文件名用于 RK3576，平台不匹配可能导致加载返回 -6。

## 6. 板端安装与运行

板端运行无需 Python、PyTorch、ONNX Runtime 或 Toolkit。使用板厂适配的内核/NPU 驱动及兼容 Runtime，记录实际版本。不要为本包盲目刷 kernel/dtb。

在交付包根目录执行，板端为 aarch64：

```bash
sha256sum -c SHA256SUMS
PKG="$PWD"
DEPLOY=/home/firefly/deployments/anti_uav_p3_repair
mkdir -p "$DEPLOY/src"
tar -xzf "$PKG/source/code.tar.gz" -C "$DEPLOY/src"
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libopencv-dev libssl-dev \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libdrm-dev
cd "$DEPLOY/src"
bash scripts/anti_uav/dist_native/build.sh "$DEPLOY"
RKNN_INCLUDE=/absolute/path/to/rknpu2/include \
  bash scripts/anti_uav/build_dist_detector_on_board.sh "$DEPLOY/libanti_uav_detector.so"
cp "$PKG/models/detector_960x544_rk3576_w8a8.rknn" "$DEPLOY/"
```

下面使用 **30 FPS 恒帧率视频**。其他 FPS 须改 `MOTION_PARAMS` 第 5 项，文件模式按源 FPS 计时；不得用处理吞吐 FPS 代替源帧率。

```bash
MOTION_PARAMS=0.03,0.01,0.10,1,30,1.5,180,1500,3000,240,16,3,4,0.03,0.03,1
"$DEPLOY/anti_uav_dist_native" \
  --model "$DEPLOY/detector_960x544_rk3576_w8a8.rknn" \
  --detector-library "$DEPLOY/libanti_uav_detector.so" \
  --tracker-library "$DEPLOY/libmotion_tracker.so" \
  --gmc-library "$DEPLOY/libdist_gmc.so" \
  --tracking motion --gmc-mode estimate --motion-params "$MOTION_PARAMS" \
  --video /absolute/path/input.mp4 --decoder ffmpeg \
  --decode-threads 4 --decode-threading frame --preprocess fused \
  --workers 2 --inflight 2 --npu-masks 0,1 --cpus 4,5,6,7 \
  --conf 0.03 --iou 0.45 --warmup 100 \
  --output "$DEPLOY/runs/video_$(date +%Y%m%d_%H%M%S)" --save-observations
```

彩色视频接入前应按第 5 节转灰；上述命令适用于灰度/gray3 源。RK3576 两个 context 绑定两个 NPU 核对不同帧数据并行，不使用 RK3588 的第三核设置。采集/解码、推理与 GMC 可重叠，结果与关联仍按源时间顺序输出。双核主要提高吞吐，不会把单帧延迟除二；队列越深延迟可能越高。

## 7. 跟踪输出与集成

`--tracking motion` 使用基于 Dist 距离矩阵/LAP 基元的优化因果 C++ 关联，不是公开 Dist 原封移植，也不是 RK-BoT-SORT。使用 Kalman 预测、运动/尺寸/IoU 门控、全局一对一匹配及歧义判断。没有未来帧短窗等待；GMC 采用质量门控，失败时增加相机运动不确定性，不把失败的单位变换当作可靠静止证据。

业务要求不吞 detector 框时，应读取每帧 `observations`，**不要只读 confirmed-only 的 `displayed_tracks`**。每个 observation 保留当前 detector 原 bbox/score，不用 Kalman 预测替换坐标；状态可能为 confirmed、pending、ambiguous、below_low 或 unassigned。candidate ID 只表示暂定关联，不证明身份一定正确；歧义框可能 ID=null。没有检测的帧不会伪造预测框，也不能承诺遮挡、多目标交叉时绝无 ID 变化。

默认采用质量门控 GMC；需要无 GMC 对照时，使用 `--gmc-mode disabled`（图像坐标运动）或 `unavailable`（保留未知相机不确定性），须连同实际参数保存并重新验收，不可默默混用。`source/code.tar.gz` 中包含所需跟踪和 GMC 实现，板端全链路为 C++。

## 8. 摄像头、延迟与验收

摄像头先确认设备节点、RAW8/灰度格式、曝光参数与真实 FPS。支持的 RAW8 V4L2 设备可把视频源替换为摄像头，并使用 `--decoder v4l2 --camera-format raw8-gray --camera-policy latest --camera-dispatch independent --camera-buffers 4`，另设置正确的 `--camera-fps` 和 `--frames 12000`。独立采集不等待空闲 NPU；latest 会丢弃过时帧以避免排队，并非不丢帧模式。

先 warmup，再分别对 detector-only（加 `--detector-only`）和 detector+motion/GMC 各测至少 12,000 帧，记录吞吐、单帧 P50/P95/P99、采集掉帧、温度及 NPU/CPU/DDR 频率。热降频不能靠短测 FPS 掩盖。

驱动帧时间戳到结果不必然等于曝光开始到结果；须先确认驱动时间戳语义，获取曝光开始或通过硬件同步/外部测量，才能验收“曝光开始到结果 ≤30 ms”。本交付不将软件时间戳冒充曝光实测。

交付验收要求：全部哈希一致、Runtime 加载成功、输入/输出协议正确、同帧 INT8 精度可接受、观测框不被跟踪器静默丢弃、多目标关联及长测通过，再确认可用于正式业务。
