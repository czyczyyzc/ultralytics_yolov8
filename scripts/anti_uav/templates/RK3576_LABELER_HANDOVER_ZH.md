# RK3576 无人机检测与跟踪部署交接

生成时间：{{CREATED_AT}}

## 1. 交付内容与验证边界

本交付为 **{{TRAINING_VIDEOS}} 视频重新训练的纯 P3 Student**，输入 `960×544`，RK3576 专用 `W8A8 INT8`，配套原生 C++ 因果 Dist 关联及质量门控 GMC。

- 从公开 `yolov8n.pt` 起步，不依赖旧项目 Student 初始化。
- P3 foundation 30 epochs → 冻结 P3、训练 Add-on P2 Teacher 12 epochs → 从本轮 P3 foundation 初始化并蒸馏 Student 15 epochs。
- 训练期辅助 P2 和投影模块在导出时物理移除。部署仅 P3/P4/P5，stride 为 8/16/32，不计算 P2。
- QAT 关闭，从 FP32 Student 做 RKNN PTQ INT8。
- 已完成旧/新模型 FP32 对比、九输出 ONNX 检查、RK3576 编译、Toolkit 加载，以及 200 帧宿主机量化仿真。
- 交付的 `.rknn` 就是实际完成仿真的二进制。宿主机验证不等于真机验收。
- 本版本尚未实测 RK3576 FPS、温升、P95 延迟或曝光到结果的延迟，不沿用旧版或 RK3588 测速数字。

## 2. 文件位置

服务器交付目录：

```text
{{DELIVERY_DIR}}
```

服务器训练目录：

```text
{{TRAINING_DIR}}
```

服务器数据目录：

```text
{{DATA_DIR}}
```

| 包内文件 | 用途 | SHA256 |
| --- | --- | --- |
| `models/detector_960x544_rk3576_w8a8.rknn` | RK3576 板端模型 | `{{MODEL_SHA256}}` |
| `models/p3_student_best_fp32.pt` | FP32 评测和重新导出 | `{{PT_SHA256}}` |
| `models/detector_960x544_rkopt.onnx` | RKNN 优化九输出 ONNX | `{{ONNX_SHA256}}` |
| `metadata/` | 训练、数据、校准、旧新对比与量化检查记录 | 见 `SHA256SUMS` |
| `source/code.tar.gz` | 与本交付匹配的 C++ 源码和构建脚本 | 见 `SHA256SUMS` |

源码仓库：`git@github.com:czyczyyzc/ultralytics_yolov8.git`，固定提交 `{{COMMIT}}`。源码包遵循仓库 AGPL-3.0 许可，商业集成应核对适用授权要求。

## 3. 数据与训练配置

旧 52 视频的图像清单和重复次数保留，再追加本次数据库快照中有有效标注的视频。`Video00004` 仍只用于最终测试；`Video00009` 仅用于验证和 checkpoint 选择，均不进入训练或 INT8 校准。

已审核视频可使用 confirmed/absent；待审核及标注中视频只使用 confirmed 正样本。未标注、uncertain、无效框及未审核 absent 帧不当作负样本。此规则是为隔离已发现的 absent 漏标风险，**不表示待审核正样本已经通过人工验收**。

```json
{{DATA_POLICY}}
```

输入 `960×544`，batch 64、nbs 128、seed 20261015。Mosaic/MixUp/Copy-Paste/旋转/透视/垂直翻转关闭；translate 0.05、scale 0.2、水平翻转 0.5；HSV 为 0.015/0.4/0.3。保留旧负样本池和约 15% 的每 epoch 负样本比例，不削减原有正样本曝光。

旧 328 张贴图的在线缓存与替换概率 0.5 沿用。没有背景缓存的新视频使用原图，**不宣称新增视频已做全量换贴图**。无额外大尺度视图、无新增面积过滤，小框和占满画面的大框均保留。

## 4. 旧新性能对比

以下为 `Video00004 + Video00009` 合并 3,780 张原始图、1,245 个 GT 的 **FP32 detector-only** 结果。`conf=0.03`、NMS IoU 0.45、匹配 IoU 0.5；AP 从 conf 0.001 排序重新计算，不取两段 AP 的平均。Video00009 参与 checkpoint 选择，因此合并集不是完全独立测试集。

{{METRICS}}

详细分尺度、分阈值及单视频结果见 `metadata/fp32_comparison.json`。新增固定翼等视频已加入训练，不能将其训练来源拟合表现称为独立测试精度。

量化仿真结果见 `metadata/quantization_smoke/results.json` 和 `REPORT.md`：同帧对比 FP32、ONNX 与 RKNN INT8，每段抽取 100 帧，共 200 帧。这是量化冒烟检查，不是全量精度或板端性能；少量样本的指标变化不能替代真机完整评测。

## 5. 模型与图像协议

```text
目标平台：rk3576（不是 rk3588）
网络输入：宽 960 × 高 544，UINT8 NHWC RGB
灰度输入：复制为三通道 RGB，或使用现有 fused 灰度预处理
归一化：RKNN 内部 mean=[0,0,0]，std=[255,255,255]
letterbox：等比例缩放，padding=114
confidence：0.03
NMS IoU：0.45
单帧 max detections：100
面积过滤：无
```

禁止拉伸图像、重复除以 255，或使用“隔一个像素直接丢一个像素”替代既定缩放。C API 负责 BGR/灰度到 RGB；不能重复交换通道。

模型为九路 branch-major NCHW 输出，每个尺度依次为 bbox64/class1/score-sum1：

| 层级 | bbox | class / score-sum |
| --- | --- | --- |
| P3 stride 8 | `1×64×68×120` | `1×1×68×120` |
| P4 stride 16 | `1×64×34×60` | `1×1×34×60` |
| P5 stride 32 | `1×64×17×30` | `1×1×17×30` |

RKNN Toolkit2 2.3.2，W8A8、normal、channel PTQ；384 张训练来源的分层校准图，不使用验证/测试图。不能靠改文件名把 RK3588 模型用于 RK3576；平台错误可能表现为 `rknn_init=-6`。

## 6. 板端构建

板端只需 C++、OpenCV、RKNN Runtime 和相关解码库，无需 Python、PyTorch、ONNX Runtime 或 RKNN Toolkit。先安装板厂适配的 Runtime/driver，并准备匹配的 `rknn_api.h`。不要为加载模型盲目刷写 kernel/dtb。

在交付包根目录执行：

```bash
sha256sum -c SHA256SUMS
PKG="$PWD"
DEPLOY=/home/firefly/deployments/anti_uav_p3_student
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

必须在 aarch64 板端编译，不能复制服务器 x86_64 `.so`。安装后的模型 SHA256 应为 `{{MODEL_SHA256}}`。

## 7. 双核并行与跟踪

RK3576 使用两个独立 RKNN context，分别绑定 core 0/1，对**不同帧**做数据并行。不得使用第三核；双核提升吞吐，但不会把单帧推理延迟除以二。解码/采集、两个推理 worker、GMC 可以重叠，关联及最终结果按帧顺序输出。

以下为 30 FPS 恒帧率视频示例，源视频不同 FPS 时必须修改参数第 5 项：

```bash
DEPLOY=/home/firefly/deployments/anti_uav_p3_student
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

`--tracking motion` 是后续优化的因果运动关联实现，**不是与公开 Dist 代码完全等价的原封移植**。它使用 Kalman 预测及门控、全局一对一匹配、轨迹确认和歧义判断；GMC 根据背景估计相机运动，并受质量门控。匹配失败时不能强行承诺 ID 正确。

业务需要“不漏输出 detector 框”时，应消费 `observations`，而不是仅消费 confirmed tracks。每帧 observation 对应原 detector bbox/score，不用 Kalman 预测替换检测坐标。Pending/ambiguous 框仍保留；暂定 ID 不等于已确认身份，歧义时可能没有可靠 ID。不要通过 GT 或业务猜测强行统一 ID。

固定相机的 GMC 可用 `--gmc-mode disabled` 做对照。运动相机应先验证 GMC 质量，不要因失败就无条件使用错误变换。

## 8. 摄像头与延迟验收

先用 `v4l2-ctl` 核实节点、RAW8 格式、分辨率与真实 FPS。摄像头可使用 `--decoder v4l2 --camera-format raw8-gray --camera-policy latest --camera-dispatch independent --camera-buffers 4`，同时传入正确的 `--camera-fps`、`--frames` 和 motion nominal FPS；其余模型、双核和 Tracker 参数不变。

先做 NPU warmup，再分别测 detector-only 和 detector+关联/GMC 至少 12,000 帧，记录 FPS、温度、频率、P50/P95 及掉帧。latest 独立采集策略允许丢弃过时帧以降低排队，不是零丢帧模式。

“驱动帧时间戳到结果”不必然等于“曝光开始到结果”。只有确认驱动时间戳语义、取得曝光起始时间或使用硬件同步/外部测量，才能验收曝光到结果 ≤30 ms。文档不把软件计时冒充曝光实测。

## 9. 验收清单

1. 所有 SHA256 校验通过，RK3576 Runtime 加载成功，确认九输出而不是带 P2 的十二输出。
2. 同帧 FP32/INT8 比较 Recall、Precision、框 IoU；分别检查小目标、固定翼、叉形靶机、近距离大目标和困难背景。
3. 不存在额外面积过滤或 Tracker 静默丢 detector bbox。
4. 多目标时分别验证关联、交叉、出入画和遮挡，不以“编号一直没变”代替身份正确性。
5. 完成长测及硬件曝光端到端延迟测量后，才确认实际运行速度和延迟指标。

本次新增标签含未审核快照，正式验收前仍应完成标注质量复核。后续修复标签需生成新的冻结快照并重训，不能覆盖本包的训练来源记录。
