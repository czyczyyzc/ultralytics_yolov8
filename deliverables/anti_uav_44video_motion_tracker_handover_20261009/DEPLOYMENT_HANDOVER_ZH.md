# RK3588S 无人机检测与优化跟踪部署交接

版本：2026-10-09
适用对象：负责 RK3588S 板端集成、运行服务和验收的部署人员。

## 1. 交付结论

本交付包含两种 44 视频训练的检测器，以及同一套优化后的原生 C++ 跟踪链路：

| 方案 | 检测尺度 | 适用场景 | 当前建议 |
| --- | --- | --- | --- |
| P3 | P3/P4/P5，stride 8/16/32 | 严格实时、误检敏感、端到端延迟优先 | 默认部署候选 |
| Frozen-P3 + Add-on P2 | P2/P3/P4/P5，stride 4/8/16/32 | 4-8 px 极小目标召回优先 | 作为高召回模式单独验收 |

跟踪器不是旧 RK-BoT-SORT，也不是最初的 Dist 公开代码直接移植版。当前版本是在 Dist 的 LAPJV/矩阵基础上实现的**因果运动跟踪器**，主要特性为：

- 一对一全局匹配，先匹配 confirmed 轨迹，再处理 tentative 轨迹。
- 同时建模平滑运动和有限机动，门限随目标尺度、Kalman 不确定性及 GMC 可靠度变化。
- quality-gated GMC；GMC 失败时增加运动不确定性，不把单位矩阵误当成“相机确定静止”。
- 不等待未来帧、不做事后 ID 重映射，因此不会引入短窗回看延迟。
- 每个当前帧 detector bbox 都进入 `observations` 输出；pending、ambiguous、低分框不会被跟踪器静默删除。
- 不用预测框伪造检测。当前帧 detector 没有框时，最终输出也没有测量框。

板端完整运行只依赖 C++、RKNN Runtime 和 OpenCV，不需要 Python、PyTorch、ONNX Runtime 或 RKNN Toolkit。Python 只用于服务器训练、模型转换、离线评测和可视化。

## 2. 固定版本与文件

### 2.1 代码

Git 仓库：

```text
git@github.com:czyczyyzc/ultralytics_yolov8.git
```

本交付固定提交：

```text
ed3f8e9642ced60f5117849f05ebcabeae33c9bb
```

47 服务器代码目录：

```text
/mnt/chenziye/codes/ultralytics_yolov8
```

关键源码：

| 文件 | 用途 |
| --- | --- |
| `scripts/anti_uav/dist_native/video.cpp` | C++ 解码/摄像头、三 RKNN worker、顺序输出、统计 |
| `scripts/anti_uav/dist_native/motion_tracker.cpp` | 优化后的因果运动跟踪器 |
| `scripts/anti_uav/dist_native/motion_tracker.hpp` | 跟踪器 ABI、默认参数和 observation 状态 |
| `scripts/anti_uav/dist_native/global_assignment.hpp` | 一对一全局匹配及歧义判断 |
| `scripts/anti_uav/dist_native/gmc.cpp` | 稀疏光流、部分仿射 GMC 及质量评分 |
| `scripts/anti_uav/dist_native/camera_source.hpp` | V4L2 raw8 摄像头输入及驱动时间戳 |
| `scripts/anti_uav/rknn_yolov8_native/detector_c_api.cpp` | RKNN 检测器、预处理、后处理和 NMS |
| `scripts/anti_uav/dist_native/build.sh` | 原生程序、tracker、GMC 编译 |
| `scripts/anti_uav/build_dist_detector_on_board.sh` | RKNN detector 动态库编译 |

服务器实验目录中的 `libmotion_tracker.so` 是 x86_64，仅用于服务器回归，**不能复制到 RK3588S**。板端必须在 aarch64 环境从上述提交重新编译。

### 2.2 最新 44 视频模型

`.pt` 用于训练、评测和重新导出，板端 C++ 读取 `.rknn`。

| 模型 | 47 服务器路径 | SHA256 |
| --- | --- | --- |
| P3 `.pt` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_p3/p3/weights/best.pt` | `761ba22431d2062a9fde1cfda08c5e394e2a6f34b60b9328b50ae9860e0b4471` |
| P2+P3 `.pt` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_addon/p2/weights/best.pt` | `8f125903f223810c611bb97f5de3a51c062dab6c4cb3f67420d3d92a3100c1dd` |
| P3 RK3588 INT8 | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved_rknn_exports_20260930/44_p3/detector_960x544_int8.rknn` | `011373d915c7b05c8cf793be21da87ffdb7bdda840bcbb7fdc3233eaa20d200c` |
| P2+P3 RK3588 INT8 | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved_rknn_exports_20260930/44_p2p3/detector_960x544_int8.rknn` | `95c4cf02d73ec9b6056f96545c2d3bd4aa195cb084607711f3840912956133e5` |

本地 RKNN 副本：

```text
deliverables/anti_uav_approved40_44_rknn_20260930/44_p3/detector_960x544_int8.rknn
deliverables/anti_uav_approved40_44_rknn_20260930/44_p2p3/detector_960x544_int8.rknn
```

两份 RKNN 使用 `rknn-toolkit2==2.3.2`、`target=rk3588`、384 张灰度校准图生成。P3 有 9 个输出，P2+P3 有 12 个输出。RK3588 RKNN 不能直接用于 RK3576；平台不匹配会在 `rknn_init` 返回错误。

这两份 44 视频 RKNN 已完成转换和 Toolkit 解析，但还不能把它们标成“RK3588S 生产验收通过”。最近的 tracker 视频回归使用相同 44 视频 `.pt` 的 FP32 检测缓存。部署人员必须完成第 9 节的 INT8 板端验收。

## 3. 检测器指标与选择

以下为 47 服务器 FP32 detector-only，Video00004 + Video00009 共 3,780 帧、1,245 个 GT；输入 `960x544`，单点指标使用 `conf=0.03`、NMS IoU `0.45`，AP 使用完整排序曲线。Video00004 为独立留出测试，Video00009 曾用于 checkpoint 选择，因此合并结果不能称为完全独立测试。

| 模型 | Recall | Precision | mAP50 | mAP50-95 | 4-8 px Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| P3 FP32 | 62.33% | 45.46% | 55.40% | 30.94% | 61.98% |
| P2+P3 FP32 | 82.49% | 28.28% | 69.19% | 36.92% | 87.45% |

P2+P3 明显提高极小目标召回，但同时增加误检。最近 `000005/000006/000007` 的视觉回归也观察到 P2+P3 更容易把云纹、人物局部和草地高光当作目标。Tracker 只能关联 detector 输出，不能把稳定的 detector 误检变成真目标。

目前仅有同源图在 RK3576 目标量化模拟器上的配对诊断：P3 Recall `62.33% -> 59.52%`，P2+P3 Recall `82.49% -> 78.15%`。该结果不是 RK3588 交付二进制的板端实测，只说明 INT8 可能造成小目标召回和定位损失。

选择原则：

- 若硬性要求接近 30 ms 单帧链路延迟，先验收 P3。
- 若漏检 4-8 px 目标的代价高于误检和延迟，验收 P2+P3。
- 两个模型应作为两个独立版本部署，不能在同一个 RKNN 图中通过“关闭 P2 输出”省掉 P2 计算。

## 4. 运行架构

实时摄像头链路：

```text
V4L2 raw8 采集
  -> 当前帧独立内存
  -> 三个独立 RKNN context，分别绑定 NPU core 0/1/2
  -> 当前帧 detector bbox

同一当前帧
  -> 独立顺序 GMC 线程，与 RKNN 推理重叠
  -> warp + quality

detector 与 GMC 均完成
  -> 按驱动时间戳顺序更新单个 motion tracker
  -> observations 全量输出
```

三核模式是**跨帧数据并行**：三张相邻帧可分别在三个 NPU 核上运行。它不是把一张图拆给三个核，也不会把单帧 NPU 延迟除以三。NPU 完成顺序可以不同，但进入同一个 tracker 的顺序必须与源时间一致。

每条摄像头/视频流必须拥有独立的 GMC 和 tracker 状态。不能把多路流的帧交错送入同一个实例。

输入和后处理固定为：

```text
网络输入：宽 960 x 高 544，RGB UINT8
letterbox padding：114
detector conf：0.03
NMS IoU：0.45
max detections：100
面积过滤：无
退化框：丢弃 width<=0 或 height<=0
```

不要增加“大框面积上限”。本项目需要同时覆盖 4-8 px 小目标和占画面大部分的大目标。

## 5. 板端环境和编译

推荐环境：Linux aarch64、可工作的 RK3588S NPU driver、与 Toolkit 2.3.2 兼容的 RKNN Runtime、g++、OpenCV C++ 开发库、OpenSSL 开发库。曾验证的厂商环境包括 Ubuntu 22.04 / kernel 5.10.198 / NPU driver 0.9.8。不要仅为复刻版本号覆盖一块已经能正常运行 NPU 和摄像头的板端内核。

拉取固定代码：

```bash
cd /home/orangepi
git clone git@github.com:czyczyyzc/ultralytics_yolov8.git ultralytics_yolov8_ziye
cd /home/orangepi/ultralytics_yolov8_ziye
git fetch origin
git checkout ed3f8e9642ced60f5117849f05ebcabeae33c9bb
```

若目录已经存在：

```bash
cd /home/orangepi/ultralytics_yolov8_ziye
git fetch origin
git status --short
git checkout ed3f8e9642ced60f5117849f05ebcabeae33c9bb
```

不要删除或覆盖未提交的板端修改。工作区不干净时，应新建独立 clone 或 worktree。

安装常规编译依赖的示例：

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libopencv-dev libssl-dev
```

构建目录必须是新的：

```bash
cd /home/orangepi/ultralytics_yolov8_ziye
DEPLOY=/home/orangepi/deployments/anti_uav_motion_ed3f8e9_20261009
mkdir -p "$DEPLOY"

bash scripts/anti_uav/dist_native/build.sh "$DEPLOY"

RKNN_INCLUDE=/absolute/path/to/rknpu2/include \
  bash scripts/anti_uav/build_dist_detector_on_board.sh \
  "$DEPLOY/libanti_uav_detector.so"
```

构建后必须包含：

```text
anti_uav_dist_native
libanti_uav_detector.so
libmotion_tracker.so
libdist_gmc.so
```

检查架构、依赖和符号：

```bash
file "$DEPLOY/anti_uav_dist_native" "$DEPLOY"/*.so
ldd "$DEPLOY/anti_uav_dist_native"
ldd "$DEPLOY/libanti_uav_detector.so"
nm -D "$DEPLOY/libmotion_tracker.so" | grep motion_observations
nm -D "$DEPLOY/libdist_gmc.so" | grep gmc_quality
```

若 `gmc_quality` 缺失，程序应失败，而不是静默退回旧 GMC。若开发库不在系统目录，可使用 `scripts/anti_uav/dist_native/build_camera.sh` 的隔离前缀方式构建。

## 6. 模型传输与目录

示例目录：

```bash
mkdir -p "$DEPLOY/models/p3" "$DEPLOY/models/p2p3"
```

从 47 服务器复制：

```bash
scp root@47.107.185.207:/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved_rknn_exports_20260930/44_p3/detector_960x544_int8.rknn \
  "$DEPLOY/models/p3/"

scp root@47.107.185.207:/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved_rknn_exports_20260930/44_p2p3/detector_960x544_int8.rknn \
  "$DEPLOY/models/p2p3/"
```

传输后核对：

```bash
sha256sum "$DEPLOY/models/p3/detector_960x544_int8.rknn"
sha256sum "$DEPLOY/models/p2p3/detector_960x544_int8.rknn"
```

SHA 必须分别为：

```text
011373d915c7b05c8cf793be21da87ffdb7bdda840bcbb7fdc3233eaa20d200c
95c4cf02d73ec9b6056f96545c2d3bd4aa195cb084607711f3840912956133e5
```

## 7. 运行命令

### 7.1 30 FPS 视频

下面以 P3 为例。P2+P3 只替换 `MODEL`，其余协议保持一致。

```bash
DEPLOY=/home/orangepi/deployments/anti_uav_motion_ed3f8e9_20261009
MODEL="$DEPLOY/models/p3/detector_960x544_int8.rknn"
VIDEO=/absolute/path/to/input.mp4
OUT="$DEPLOY/runs/video_p3_$(date +%Y%m%d_%H%M%S)"
MOTION_PARAMS=0.03,0.01,0.10,1,30,1.5,180,1500,3000,240,16,3,4,0.03,0.03,1

"$DEPLOY/anti_uav_dist_native" \
  --model "$MODEL" \
  --detector-library "$DEPLOY/libanti_uav_detector.so" \
  --tracker-library "$DEPLOY/libmotion_tracker.so" \
  --gmc-library "$DEPLOY/libdist_gmc.so" \
  --tracking motion --gmc-mode estimate \
  --motion-params "$MOTION_PARAMS" \
  --video "$VIDEO" --decoder ffmpeg \
  --decode-threads 4 --decode-threading frame \
  --preprocess fused \
  --workers 3 --inflight 9 --npu-masks 0,1,2 \
  --cpus 4,5,6,7 --conf 0.03 --iou 0.45 \
  --warmup 100 --output "$OUT" --save-observations
```

`MOTION_PARAMS` 第 5 项是源 FPS。输入不是 30 FPS 时必须替换该项；VFR 视频应先明确时间戳策略，不能只按容器标称 FPS 猜测。

`--preprocess fused` 当前优化了 1920x1080 到 960x540 内容区再补成 960x544 的路径。其他尺寸会回退至 OpenCV；必须通过同帧 bbox 对比确认像素等价。

### 7.2 120 FPS raw8 灰度摄像头

先核对实际节点和格式：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video11 --all
v4l2-ctl -d /dev/video11 --list-formats-ext
```

运行示例：

```bash
DEPLOY=/home/orangepi/deployments/anti_uav_motion_ed3f8e9_20261009
MODEL="$DEPLOY/models/p3/detector_960x544_int8.rknn"
OUT="$DEPLOY/runs/camera_p3_$(date +%Y%m%d_%H%M%S)"
MOTION_PARAMS=0.03,0.01,0.10,1,120,1.5,180,1500,3000,240,16,3,4,0.03,0.03,1

"$DEPLOY/anti_uav_dist_native" \
  --model "$MODEL" \
  --detector-library "$DEPLOY/libanti_uav_detector.so" \
  --tracker-library "$DEPLOY/libmotion_tracker.so" \
  --gmc-library "$DEPLOY/libdist_gmc.so" \
  --tracking motion --gmc-mode estimate \
  --motion-params "$MOTION_PARAMS" \
  --video /dev/video11 --decoder v4l2 \
  --camera-format raw8-gray --camera-policy latest \
  --camera-dispatch direct --camera-buffers 4 \
  --camera-fps 120 --camera-start overlap \
  --preprocess fused \
  --workers 3 --inflight 3 --npu-masks 0,1,2 \
  --cpus 4,5,6,7 --conf 0.03 --iou 0.45 \
  --frames 12000 --warmup 100 \
  --output "$OUT" --save-observations
```

`--camera-fps 120` 只配置跟踪时间模型，不会修改传感器帧率。当前程序要求有效且单调递增的驱动时间戳。驱动时间戳尚未证明等于传感器曝光开始时间，因此 `driver_to_output_ms` 不能直接宣传为光子到结果延迟。

低延迟摄像头默认使用 `direct`。此前 `independent` 采集线程没有降低同板 P3 延迟，不作为默认值。

## 8. 跟踪输出接口

### 8.1 必须消费 observations

旧接口 `motion_update` 返回 confirmed-only `[id, detection_index]`，仅适合确认态评测。业务显示或平台输出若只消费该数组，就会重新出现“detector 有框、tracking 没框”。

每次 `motion_update` 成功后，必须调用：

```cpp
int count = motion_observations(handle, triples, capacity);
```

每条结果为：

```text
[id_or_zero, detection_index, status]
```

`capacity` 必须覆盖当前帧全部 detector bbox；不足时返回 `-1`，不会静默截断。

| status | 名称 | 推荐显示 | 含义 |
| ---: | --- | --- | --- |
| 0 | unassigned | `DET` | 当前没有可靠关联或出生证据，bbox 仍保留 |
| 1 | pending | `ID n?` 或 `DET` | 候选轨迹，ID 还未确认 |
| 2 | confirmed | `ID n` | 已确认轨迹在当前 detector bbox 上的身份 |
| 3 | ambiguous | `UNCERTAIN` | 存在多个合理匹配，不强行编造 ID |
| 4 | below_low | `LOW` | bbox 被保留，但低于跟踪关联阈值 |

对每个 observation，应使用 `detection_index` 取回**当前帧 detector 原始 bbox 和 score**。不能用内部 Kalman 预测框替换它。

当前 16 参数配置从首个合格观测分配 tentative ID，并在确认时保留同一 ID。确认条件仍是 4 个处理帧窗口内至少 3 次匹配，且观测均值满足 birth 阈值。`ID n?` 不能对外声称为 confirmed identity。

若 ambiguous observation 没有 ID，业务端必须仍输出 bbox；不应为满足“每框必须有数字”而随意分配新 ID。强制数字 ID 会把真正的匹配不确定性变成错误 ID shift。

### 8.2 JSONL 和 summary

添加 `--save-observations` 后：

- `observations.jsonl`：逐帧 detector bbox、score、observation ID/status、GMC warp。
- `summary.json`：模型/动态库 SHA、FPS、阶段耗时、温度频率、NPU worker 分配、observation 计数。
- `latency.csv`：摄像头模式逐帧采集、推理、GMC、关联和输出时间。

验收时必须检查：

```text
observation_boxes == detector bbox 总数
```

`displayed_tracks` 是 confirmed-only 兼容字段，不能用于验证“是否吞框”。

## 9. GMC 策略

生产默认：

```text
--gmc-mode estimate
```

GMC 在宽 320 px 缩略图上使用最多 128 个角点，并输出拟合质量。可靠 GMC 用于区分目标运动与相机运动；不可靠 GMC 不会被当成确定的零运动，而是提高跟踪器的运动不确定性。

不要因为某一帧 GMC 失败就删除 GMC。固定检测缓存的 Video00009 回归中：

| 模式 | 连续 GT 段 ID 变化，含 tentative | confirmed Recall |
| --- | ---: | ---: |
| 质量门控 GMC | 0 | 66.35% |
| 不运行 GMC | 17 | 66.16% |

`--gmc-mode unavailable` 仅用于无 GMC 环境或消融实验。它跳过 GMC 线程，保留未知相机运动不确定性，不是生产默认。

## 10. 已完成的 tracker 回归

在最新 44 视频 FP32 detector 固定输出上，优化 tracker 已完成如下审计：

| 视频/模型 | Detector bbox | Observation | Tracker 丢框 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `000002` P3 | 547 | 547 | 0 | 原 15.8 秒断 ID 故障修复，主目标保持 ID 1 |
| `000002` P2+P3 | 666 | 666 | 0 | 嵌套重复框保留为 UNCERTAIN，不伪造第二身份 |
| `000005` P3 / P2+P3 | 825 / 1,158 | 825 / 1,158 | 0 / 0 | P2+P3 主目标第 2-1084 帧连续 ID 1 |
| `000006` P3 / P2+P3 | 753 / 1,134 | 753 / 1,134 | 0 / 0 | 后段主目标同一 ID，P2+P3 detector 断帧更少 |
| `000007` P3 / P2+P3 | 1,663 / 1,922 | 1,663 / 1,922 | 0 / 0 | 草地长轨迹主要是 detector 背景误检 |

逐帧几何审计没有发现相邻帧中同一个高重叠 confirmed bbox 直接切换到另一个 confirmed ID。后三段没有完整 identity GT，因此这不是正式 HOTA、IDF1 或 MOTA 结论。

本地报告：

```text
deliverables/clip_000002_causal_global_20261008/RESULTS_ZH.md
deliverables/clips_000001_000004_causal_tracker_20261009/RESULTS_ZH.md
deliverables/clips_000005_000007_causal_tracker_20261009/RESULTS_ZH.md
```

## 11. 性能参考及限制

曾在同一 RK3588S、实时灰度摄像头、原生 C++、三核并行、12,000 帧条件下测得以下数据：

| 指标 | 旧 28 视频 P3 INT8 | 旧 28 视频 P2+P3 INT8 |
| --- | ---: | ---: |
| 完整流水线平均 FPS | 120.00 | 86.58 |
| 单帧 NPU 平均 | 21.20 ms | 31.56 ms |
| 驱动时间戳至完整结果平均 | 29.91 ms | 43.96 ms |
| 驱动时间戳至完整结果 P95 | 31.84 ms | 49.59 ms |

这些数据只用于估算 P3/P2 的计算差异，使用的是旧 28 视频 RKNN 和当时的 C++ Dist/GMC 链路。**不能作为本交付 44 视频 RKNN + motion tracker 的正式性能指标。** 最新 tracker 的 x86_64 纯关联均值低于 0.002 ms，但这也不能替代 RK3588S 全链路复测。

三核提高吞吐，不会把单帧延迟变成单核的三分之一。最终 FPS、延迟、温度和丢帧必须在目标板、目标摄像头和目标散热条件下测量。

## 12. 验收步骤

### 12.1 启动前

- 核对芯片为 RK3588/RK3588S，模型 SHA 与目标平台正确。
- 记录 kernel、`librknnrt.so` 版本、NPU driver、OpenCV 和代码提交。
- 确认 NPU 三核可用，三个 context 分别绑定 `0,1,2`。
- 核对相机节点、分辨率、stride、raw8/彩色格式和真实 FPS。
- 确认板上有合适散热，不关闭温控保护。

### 12.2 功能 smoke test

- P3 与 P2+P3 分别运行至少 2,000 帧。
- `rknn_init`、动态库加载、`gmc_quality` 和 `motion_observations` 均成功。
- 三个 `npu_worker_frame_counts` 都持续增长且负载近似均衡。
- 输出目录、`summary.json`、`observations.jsonl` 和摄像头 `latency.csv` 完整。
- `observation_boxes` 等于 detector bbox 总数，没有 tracker 吞框。

### 12.3 精度和身份

- 用 Video00004 + Video00009 对 FP32 与实际 RK3588 INT8 做同帧 bbox/score 对照。
- 分别报告 Recall、Precision、FP、mAP50、mAP50-95 和 4-8 px Recall。
- 复跑 `000002` 的 15.8 秒关键段；主目标应保持同一 confirmed ID。
- 复跑 `000005/000006/000007`；检查真实目标连续性和草地/云层困难负样本。
- 多目标场景检查一对一匹配；不得将 ambiguous bbox 强制绑定到已有 ID。

### 12.4 性能和稳定性

- 每种模型连续运行至少 12,000 帧，前 100 帧只排除统计、不排除处理。
- 报告完整 FPS、NPU/预处理/GMC/关联耗时、driver-to-output 平均/P95、sequence gaps。
- 同时记录初始/结束温度、NPU/CPU/DDR 频率及是否发生降频。
- 性能测试不得包含视频绘制、编码或网络发送；业务端到端验收则必须另行加入这些开销。

### 12.5 通过条件

建议在项目现场明确数值门限后再签字。最低功能门限为：

- 所有 detector bbox 均出现在 observations 中。
- 无崩溃、无 NaN warp、无输出乱序、无 NPU worker 长时间空闲。
- 连续可见且 detector 持续有框的单目标段不发生 confirmed ID 直接跳变。
- P3/P2+P3 的模型身份、精度和速度分别记录，不混用结果。

## 13. 常见问题

### 检测有框但界面没框

接入端只消费了 `displayed_tracks` 或 `motion_update` 的 confirmed-only 结果。改为消费 `motion_observations`/JSONL 的 `observations`。

### pending 为什么不是马上 confirmed

候选 ID 从第一帧即可产生，但 confirmed 需要多帧证据。这样可避免云纹或局部重复框立刻生成稳定假 ID。业务端可画 `ID n?`，不能把它标成 confirmed。

### 为什么不能给每个 ambiguous bbox 强制 ID

ambiguous 表示存在多个近似合理身份。强制选择会直接制造错误 ID 或 ID switch。正确行为是保留 bbox 并显示 `UNCERTAIN`。

### 跟踪为什么仍会输出草地/云层目标

这些是 detector bbox。当前设计要求 tracker 不吞 detector 框，所以稳定误检也会被关联。应通过 detector 困难负样本、置信度策略或上层告警逻辑解决，不能靠跟踪器偷偷删框。

### 检测空窗时为什么不补框

当前交付只输出当前 detector 测量，避免把 Kalman 预测伪装成检测结果。内部轨迹可以跨短空窗继续存活，检测恢复后延续 ID；若业务需要预测框，必须以独立 `predicted=true` 类型输出并单独验收。

### 是否可以关闭 GMC

可以用 `--gmc-mode unavailable` 做对照，但不是默认方案。已有标注回归中关闭 GMC 的连续身份稳定性更差。

## 14. 回滚和发布管理

- 不要覆盖旧模型、旧动态库和旧结果目录；每次发布使用新的不可变目录。
- `summary.json` 应连同模型 SHA、Git commit、Runtime/driver 版本一起归档。
- 发现量化召回下降、误检不可接受、温控降频或 observation 数不一致时，回滚到已验收版本。
- P3 和 P2+P3 使用独立服务配置及模型路径，不通过软链接无记录切换。
- 对外分发前检查 Ultralytics、AGPL-3.0 motion tracker 代码和 vendored LAP 的许可义务。

服务器报告副本建议放置于：

```text
/mnt/chenziye/codes/ultralytics_yolov8/deliverables/anti_uav_44video_motion_tracker_handover_20261009/DEPLOYMENT_HANDOVER_ZH.md
```
