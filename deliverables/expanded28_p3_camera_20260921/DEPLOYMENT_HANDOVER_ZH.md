# RK3588S 无人机检测与跟踪部署交接文档

版本：2026-09-21 · 纯 P3 / 960×544 INT8 / 原生 C++ Dist + GMC

后续采集实验：已新增可选的独立摄像头采集模式，测试记录见
`../expanded28_capture_independent_20260921/README.md`。同板纯 P3 各 12,000 帧
实测均约 120 FPS，独立采集平均延迟 30.39 ms，对照 29.91 ms，未获收益。
**本交付仍使用原 direct 模式，不替换默认程序。** 新选项需要单独实验目录中的新程序，
不可把 `--camera-dispatch independent` 直接传给本文件所列旧版实测二进制。

## 1. 交付方案

本次交付使用 **28 视频训练的 YOLOv8n P3-P5 检测器 + Dist 公开代码适配版的 C++ 实现 + C++ GMC**，无需无人机姿态输入。

检测器取自同轮训练中添加 P2 之前的 P3 权重，已独立导出为 RK3588 平台的 RKNN INT8。**计算图中不存在 Add-on P2 分支，确实省掉了 P2 的推理计算，不是仅屏蔽其输出。** 这不是重新训练出的另一套 P3 权重。

板端完整推理不使用 Python、PyTorch、ONNX Runtime 或 RKNN Toolkit。板端运行 RKNN Runtime 和 C++；PyTorch / Toolkit 仅在服务器导出模型时使用。

| 项目 | 本次部署配置 |
| --- | --- |
| 模型输入 | 宽 960 × 高 544，RGB UINT8，INT8 RKNN |
| 检测尺度 | P3/P4/P5，stride 8/16/32，共 9 个输出张量 |
| 检测阈值 | conf=0.03；NMS IoU=0.45；最多 100 框 |
| 跟踪算法 | Dist 公开代码适配版的原生 C++ 移植，无 ReID |
| 相机运动补偿 | C++ 稀疏光流 + 鲁棒部分仿射估计，缓存光流金字塔 |
| NPU 并行 | 三个独立 RKNN context，分别绑定 core 0/1/2；最多 3 个在途帧 |
| 实时取帧 | V4L2、latest 策略、4 个采集缓冲 |
| 启动优化 | 摄像头 STREAMON 与模型加载并行；不做额外 NPU 预热 |

“Dist”在本文中特指当前仓库实际执行的公开实现适配版，不宣称复现论文的全部组件；它也不是旧 RK-BoT-SORT 改名。GMC 与 Dist 都没有独立神经网络权重，交付物是 C++ 源码及动态库。

## 2. 实测性能与口径

两组均在**同一 RK3588S**、同一摄像头、同一 C++ 程序与动态库上顺序测试，各处理 **12,000 帧**；输入均为 `960×544 INT8 + Dist + GMC`、三核数据并行。除检测模型不同外，取帧策略、跟踪参数、频率策略一致。它们是实时顺序采集，不是逐像素相同的视频帧回放。

| 指标 | Frozen-P3 + Add-on P2 | 纯 P3，本次交付 |
| --- | ---: | ---: |
| 完整流水线平均 FPS | 86.58 | **120.00** |
| 单帧 NPU 推理平均耗时 | 31.56 ms | **21.20 ms** |
| 开始读帧至完整结果，平均 | 34.53 ms | **24.94 ms** |
| 驱动帧时间戳至完整结果，平均 | 43.96 ms | **29.91 ms** |
| 驱动帧时间戳至完整结果，P95 | 49.59 ms | **31.84 ms** |
| 首末处理帧之间的源帧序号跳过比例 | 27.8% | **0.0%** |
| 采样最高温度 | 45.31°C | 46.23°C |

纯 P3 平均驱动帧延迟减少约 32.0%，NPU 耗时减少约 32.8%，吞吐提高约 38.6%。12,000 帧约运行 100 秒，不等同于无限期热稳定性认证。随后 2,000 帧复测：P3 为 119.99 FPS / 29.95 ms，P3+P2 为 86.14 FPS / 44.05 ms。

必须遵守以下指标解释：

- 120 FPS 已接近摄像头供帧上限，不代表测出了模型离线最大 FPS。
- 三核流水线约每 8.33 ms 输出一个结果，但一张图从驱动时间戳到结果仍约 29.91 ms。吞吐间隔不是单帧延迟。
- 稳态统计排除最初 100 个输出帧；这 100 帧仍真实推理、跟踪并输出，不是丢掉不处理。
- “完整结果”指 C++ 检测、GMC、Dist 关联完成，不含屏幕显示、视频编码、网络传输或接收端延迟。
- 驱动时间戳使用 MONOTONIC，记录的 flags=8193；未独立验证其与曝光时刻的对应关系，不能称为曝光至结果或光子至显示延迟。
- 当前实拍是倒置且部分遮挡的室内场景，没有标注的无人机轨迹。纯 P3 长测只有 5 个检测输出及 1 个显示跟踪输出；它验证的是运行与延迟，不是检测召回率、ID 稳定性或多目标重载性能。
- 去掉 P2 可能影响 4–8 px 小目标召回率。部署前应在业务视频上另行验收精度，不能由 FPS 推断精度不变。

五次独立进程启动测试的中位数如下；不是板子断电启动测试，也未清空系统缓存：

| 首帧指标 | P3 + P2 | 纯 P3 |
| --- | ---: | ---: |
| 程序入口至首个完整结果 | 141.37 ms | **133.47 ms** |
| STREAMON 调用开始至首个结果 | 137.93 ms | 130.00 ms |
| 首次 read 调用开始至首个结果 | 36.20 ms | 26.73 ms |

相机启流本身仍约 102–104 ms，已与模型加载重叠。程序入口统计不含 shell 和动态加载器；首次 read 已晚于 STREAMON 开始，不能把 26.73 ms 当成整个相机启动时间。

## 3. 模型、代码及证据路径

### 3.1 47 服务器

服务器：`47.107.185.207`。由授权账户访问，账户凭据单独交接，不写入本文件。

代码仓库：
```text
/mnt/chenziye/codes/ultralytics_yolov8
```

**训练源权重 .pt（用于继续训练、追溯或重新导出，不在板端直接推理）：**
```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28/training_p3/p3/weights/best.pt
```

**本次纯 P3 RKNN：**
```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/expanded28_p3_camera_20260921/detector_p3_960x544_int8.rknn
```

服务器也有交付副本、文档及测速证据：
```text
/mnt/chenziye/codes/ultralytics_yolov8/deliverables/expanded28_p3_camera_20260921/
```

上述目录包含 `detector_p3_960x544_int8.rknn`、`model_manifest.json`、`detector.rkopt.json`、`p3_equivalence.json`、`comparison.json`、本文件和各次测试的原始记录。RKNN 二进制不是通过 git 自动下载的，必须另行复制。

### 3.2 当前板端

本次已验证的板端 SSH：`orangepi@192.168.144.50`。IP 是当前实验网络地址，迁移后需重新确认。

```text
代码：/home/orangepi/ultralytics_yolov8_ziye_code
模型：/home/orangepi/deployments/expanded28_p3_camera_20260921/detector_960x544_int8.rknn
部署：/home/orangepi/deployments/expanded28_p3_camera_20260921
```

当前目录不是独立压缩交付包，存在以下依赖关系：

```text
expanded28_p3_camera_20260921/bin
  -> /home/orangepi/deployments/expanded28_camera_latency_20260921/bin
expanded28_p3_camera_20260921/deps
  -> /home/orangepi/deployments/expanded28_camera_20260921/deps
RKNN Runtime
  /home/orangepi/ultralytics_yolov8_ziye/lib/librknnrt.so
RKNN 头文件
  /home/orangepi/ultralytics_yolov8_ziye/src/rknn_api.h
```

不要删除上述目标目录。`ultralytics_yolov8_ziye` 是旧发布文件目录，**git pull 要在 `_ziye_code` 下执行**。部署含 `anti_uav_dist_native`、`libanti_uav_detector.so`、`libdist_tracker.so`、`libdist_gmc.so`；三库与程序必须配套。

### 3.3 本地文档

```text
/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/expanded28_p3_camera_20260921/DEPLOYMENT_HANDOVER_ZH.md
```

### 3.4 版本和 SHA256

功能及启动脚本基准提交：`bba52c3aaf191de6f7a353f0ad582ae8043fcaaf`。实测程序编译源码提交：`9ded0bd`；后续文档及启动脚本提交没有改变该程序二进制。本次交接文档提交在此基准之后。

| 文件 | SHA256 |
| --- | --- |
| P3 best.pt | `8c551c6b115ee9d93c9625280b1d5326b8264d4040e6b69c973af3939f63863f` |
| P3 INT8 RKNN | `a4a27a12047ba1ac287ef320d398e5c935c1e8d3e43d49fddee7e02b931af20c` |
| anti_uav_dist_native | `aa9997747603e65a0e20b710f83e378ff1c76dc0448bcc2aa30baf7e8aaf2732` |
| libanti_uav_detector.so | `a04e31573f42301e59e329cc81d991dc1306eed2d9d245681edcfadc721e8e7a` |
| libdist_tracker.so | `4b981fe2c057a23833f56d09a82fb66db29df55c0d8da043a16ac2347fddbe4a` |
| libdist_gmc.so | `9c146d2c6d1c1ceef564c119b84c3d35e26a8257f43c4260ba3d2b83523fa313` |
| librknnrt.so | `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8` |

重新编译后程序哈希可能不同，应同时记录源码提交、工具链及新的哈希，不要求新环境编译文件逐字节一致。

## 4. 环境要求

| 项目 | 已验证环境 |
| --- | --- |
| 板卡 / 架构 | Orange Pi CM5，RK3588S，Linux aarch64 |
| 系统 / 内核 | Ubuntu 22.04，厂商 5.10.198 |
| RKNN | Runtime / 导出 Toolkit 2.3.2；NPU driver 0.9.8 |
| OpenCV | 4.5.4，C++；隔离依赖目录加系统传递依赖 |
| NPU / CPU / DDR | 1 GHz / 大核 2.304 GHz / 1.56 GHz，既有 performance 策略 |
| 摄像头 | 当前 `/dev/video11`，1920×1080，BA81 raw8，stride=2048，观测约 120 FPS |

相机数据按**物理灰度 raw8**处理，复制灰度至 RGB 后送入模型，不进行彩色 Bayer 去马赛克。不能把此配置直接套给彩色 BA81、NV12、YUYV 或 MJPEG 摄像头。`/dev/video11` 不是所有板卡固定编号，必须核对实际节点。

新板需要正确的厂商相机驱动、设备树及 NPU 驱动。不能仅安装 OpenCV 就获得相机支持，也不能直接将此 RK3588 RKNN 文件用于 RK3576。当前链路不是 MPP 解码 / RGA 预处理 / 全链路零拷贝；原始帧会复制到独立内存，避免摄像头 DMA 回写与并行计算竞争。

## 5. 当前板一键运行

以下命令均在板端执行。运行前确认没有其他进程占用摄像头。每次输出目录必须不存在。

```bash
cd /home/orangepi/ultralytics_yolov8_ziye_code
D=/home/orangepi/deployments/expanded28_p3_camera_20260921
uname -r
sha256sum "$D/detector_960x544_int8.rknn"
readlink -f "$D/bin"
readlink -f "$D/deps"

# 短测：2,000 帧，前 100 帧不计入稳态统计
OUT="$D/check_$(date +%Y%m%d_%H%M%S)"
bash scripts/anti_uav/dist_native/run_camera_p3_on_board.sh \
  --frames 2000 --output "$OUT"
```

复现 12,000 帧验收：

```bash
OUT="$D/acceptance_$(date +%Y%m%d_%H%M%S)"
bash scripts/anti_uav/dist_native/run_camera_p3_on_board.sh \
  --frames 12000 --warmup 100 --output "$OUT"
```

首帧测试每次启动一个新进程，至少重复五次：

```bash
OUT="$D/startup_check_$(date +%Y%m%d_%H%M%S)"
bash scripts/anti_uav/dist_native/run_camera_p3_on_board.sh \
  --frames 1 --warmup 0 --output "$OUT"
```

启动脚本继承关系为 `run_camera_p3_on_board.sh` → `run_camera_fast_start_on_board.sh` → `run_camera_on_board.sh`。默认最终参数为：

```text
--decoder v4l2 --video /dev/video11 --camera-format raw8-gray
--camera-policy latest --camera-buffers 4 --camera-fps 120
--camera-start overlap --preprocess fused
--workers 3 --inflight 3 --npu-masks 0,1,2
--cpus 4,5,6,7 --conf 0.03 --iou 0.45
--npu-warmup 0
```

`--camera-fps 120` 声明跟踪器的帧率配置，**不会把传感器设成 120 FPS**。`--warmup 100` 只是统计排除，不是 100 次预热。不要追加 `--detector-only`，否则不再运行 Dist/GMC；不要使用 `--no-pyramid-cache`，否则改变已测配置。

## 6. 三核并行与跟踪行为

处理流程：V4L2 取最新灰度帧 → 分发给空闲 RKNN worker；同时送入顺序 GMC 线程 → 等待该帧检测/GMC 完成 → 按采集顺序进行 Dist 关联 → 输出。

三个 NPU 核是**跨帧数据并行**，不是将一张图拆成三个区域，也不是三个核联合处理同一张图。各 context 有独立输入输出内存。即便 NPU 完成顺序不同，Dist 仍按时间顺序更新，禁止直接按完成先后喂给跟踪器。

GMC 使用宽 320 px 缩略图、最多 128 个角点、每 5 帧或支撑不足时刷新角点；每个处理帧仍执行必要的光流/运动估计，不是每五帧才运行一次 GMC。首帧或估计不可靠时回退单位变换。无需 IMU、无人机姿态或额外 ReID 模型。

Dist 的固定参数位于 `scripts/anti_uav/dist_native/video.cpp`，当前不是独立 YAML/CLI 配置：

| 参数 | 值 | 说明 |
| --- | ---: | --- |
| high | 0.03 | 高分检测匹配 |
| low | 0.01 | 低分补匹配阈值 |
| new_score | 0.10 | 新建轨迹阈值 |
| match | 0.8 | 第一轮匹配的距离门限，不是“要求 IoU≥0.8” |
| buffer | 30 | 按 fps/30 缩放；120 FPS 时 max_lost=120 次更新 |

重要：本次 detector conf=0.03，因此 0.01–0.03 的框不会进入跟踪器，不能宣称 low=0.01 已启用该范围的检测。新轨迹还受 0.10 阈值和确认逻辑控制；有检测但没有带 ID 框可能符合当前行为。

输出跟踪框仅来自当前帧实际匹配的检测，不会把失配后的 Kalman 预测框当成检测框绘制。全尺寸合法框没有 10%/25% 面积过滤，但仍经过置信度筛选、NMS、最多 100 框以及退化框剔除；不代表所有候选框都无条件输出。

时间模型目前按处理更新步推进，源帧跳过会记录但不会增加额外 Kalman 更新。丢帧频繁、低帧率、多路流或快速机动场景必须另外验收跟踪。每个视频流必须有独立 Dist/GMC 状态，不能将多路流混进同一个状态实例。

## 7. 输出文件和接入方式

| 输出 | 内容 |
| --- | --- |
| 标准输出 | `first_result`、每 500 帧 `progress`、最终 `complete` JSON 事件；不是逐帧检测接口 |
| summary.json | 总帧数、FPS、阶段耗时、首帧、模型哈希、worker 分配、温度频率快照 |
| latency.csv | 每帧 index、相机 sequence、时间戳、预处理/NPU/GMC/关联等耗时 |
| first_raw_gray.png | 第一帧原始灰度图，计时结束后保存 |
| observations.jsonl | 仅添加 `--save-observations` 时保存逐帧检测、跟踪 ID 和 GMC 矩阵 |

调试输出示例命令：

```bash
OUT="$D/observations_$(date +%Y%m%d_%H%M%S)"
bash scripts/anti_uav/dist_native/run_camera_p3_on_board.sh \
  --frames 2000 --output "$OUT" --save-observations
```

记录结构：`frame_index` 是零基处理序号，`boxes_xyxy_score` 是原始输入图像像素坐标下的 `[x1,y1,x2,y2,score]`；`displayed_tracks` 含 `id/detection_index/score/box`；`warp` 为上一处理帧到当前帧的 2×3 GMC 矩阵。相机原始序号在 `latency.csv` 中，与 observations 的处理序号按行对应。ID 在进程重启后重置，不是全局永久 ID。

显示时如果旋转或缩放图像，必须同步变换框坐标。需要所有检测框时消费 `boxes_xyxy_score`，需要已确认跟踪框时消费 `displayed_tracks`，两者不能混为一谈。

现有程序是指定帧数的可执行程序，不是 REST 服务、RTSP 推流服务或完善的无限循环守护进程。相机模式要求 `--frames > --warmup`，不能用 `--frames 0` 表示永远运行。JSONL 写入也不是低延迟网络 IPC，逐帧落盘会增加负担；上述性能测试未开启 observations。

产品接入建议在 `video.cpp` 完成检测/GMC/关联之后接入有界结果队列或回调，避免网络、绘制、编码阻塞顺序消费线程。该接入以及服务生命周期/日志轮转属于后续集成，不在本次已实测范围内。

可复用 C 接口位于 `tracker.cpp` 与 `gmc.cpp`：`dist_create/update/destroy/error`，`gmc_create/apply/destroy/error`。`dist_update` 输入 N×5 float 的 xyxy+score 及 6 个 double 的仿射矩阵，返回 `(track_id, detection_index)` 对；GMC 输入支持 1 通道灰度或 3 通道 BGR、带行 stride 的图像。请按源码签名及返回码集成，同一实例顺序调用。

## 8. 新板迁移与重新编译

本节是迁移操作说明，**并未声称已在另一块全新系统板上验证**。先准备相机可用、NPU 可用的兼容 aarch64 Ubuntu 环境，再迁移应用；不要覆盖 boot/kernel/DTB，也不要为追 FPS 关闭温控。

### 8.1 代码同步

仓库：`git@github.com:czyczyyzc/ultralytics_yolov8.git`，需要访问权限。保持本地修改 → commit/push → 服务器/板端 pull 的流程。先检查目标工作区是否干净，存在未提交修改时先协调，禁止强制覆盖。

```bash
git clone --filter=blob:none --sparse \
  git@github.com:czyczyyzc/ultralytics_yolov8.git \
  /home/orangepi/ultralytics_yolov8_ziye_code
cd /home/orangepi/ultralytics_yolov8_ziye_code
git sparse-checkout set scripts/anti_uav
git rev-parse HEAD
```

现有 checkout 应使用 `git pull --ff-only`，并核对提交包含基准版本。当前实验板 origin 曾设为 `git://127.0.0.1:19418/ultralytics_yolov8`，它是临时 SSH 转发，不是公共源；隧道关闭时 pull 失败是正常的。可配置可访问的正式 origin，或在本地 push 成功后传输 Git bundle，再在目标机执行 `git pull --ff-only /path/to/update.bundle main`。完整离线克隆需要包含历史的完整 bundle；增量 bundle 只能给已有基线 checkout 使用。

### 8.2 复制实际文件

复制 RKNN、匹配的 RKNN Runtime/头文件和 OpenCV 依赖，而不是只复制软链接。以下从**已能访问原板的迁移电脑**执行，示例要求预先创建本地 `handoff_assets` 空目录：

```bash
mkdir -p handoff_assets
rsync -aL orangepi@192.168.144.50:/home/orangepi/deployments/expanded28_p3_camera_20260921/ \
  handoff_assets/deployment/
rsync -aL orangepi@192.168.144.50:/home/orangepi/ultralytics_yolov8_ziye/lib/ \
  handoff_assets/rknn_lib/
rsync -aL orangepi@192.168.144.50:/home/orangepi/ultralytics_yolov8_ziye/src/rknn_api.h \
  handoff_assets/
```

`-L` 解引用软链接；不要使用会漏掉链接目标的简单目录复制。上述 deployment 可能包含测速记录，可在打包时排除结果目录，但保留模型、bin、deps 和版本记录。本次没有另行生成自包含安装包。

将这些实际文件复制到目标板选定目录。源二进制内含绝对 RPATH，OpenCV 也有系统传递依赖，**不能保证目录换个位置后直接运行**。优先按下一步重编译，并检查所有动态库的 `ldd`，不是只检查主程序。依赖不完整时应按兼容系统包补齐，不要随机混用 OpenCV / RKNN 版本。

### 8.3 编译

目标板需要 g++、C++17、OpenSSL 开发头文件/库，以及兼容的 OpenCV 4.5.4 头文件/库。隔离 deps 只提供其中部分依赖，不是完整 Linux rootfs。源码编译同时生成程序和三个动态库：

```bash
cd /home/orangepi/ultralytics_yolov8_ziye_code
D=/home/orangepi/deployments/expanded28_p3_camera_20260921
# 修改为目标板上真实存在的头文件及 Runtime 目录
export RKNN_INCLUDE=/home/orangepi/ultralytics_yolov8_ziye/src
export RKNN_LIB=/home/orangepi/ultralytics_yolov8_ziye/lib
bash scripts/anti_uav/dist_native/build_camera.sh "$D/bin" "$D/deps"
export LD_LIBRARY_PATH="$D/deps/usr/lib/aarch64-linux-gnu:$RKNN_LIB:${LD_LIBRARY_PATH:-}"
ldd "$D/bin/anti_uav_dist_native"
ldd "$D/bin/libanti_uav_detector.so"
ldd "$D/bin/libdist_tracker.so"
ldd "$D/bin/libdist_gmc.so"
```

构建末尾应通过 132 RGB、132 灰度布局及 10 个非法输入测试；`ldd` 不得有 `not found`。新板的 `bin/deps` 应为完整实际目录，或者指向明确保留的目标；不要在旧板共享 bin 软链接上直接重编译从而覆盖旧版。需要换路径时用 `ANTI_UAV_CAMERA_DEPLOY=/actual/deployment` 覆盖启动目录。

确认设备节点、灰度格式、用户对视频/NPU设备的访问权限后，先跑 120/2,000 帧功能测试，再复现 12,000 帧；不同板型、DDR 频率、驱动或相机不能套用 120 FPS 保证。

## 9. 重新导出模型（可选）

一般部署直接复制已校验 RKNN，不需要重新导出。确需重导出时，在 47 服务器使用已配置的 `.venv` 和下列命令；不要在训练权重目录原地覆盖导出物：

```bash
cd /mnt/chenziye/codes/ultralytics_yolov8
OUT="runs/anti_uav/p3_reexport_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
cp runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28/training_p3/p3/weights/best.pt "$OUT/best.pt"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python \
  scripts/anti_uav/export_detector_rkopt_onnx.py \
  --weights "$OUT/best.pt" --imgsz 544,960 --output "$OUT/detector.onnx"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python \
  scripts/anti_uav/build_rknn.py --onnx "$OUT/detector.onnx" \
  --output "$OUT/detector_p3_960x544_int8.rknn" --target rk3588 --quantize \
  --dataset deliverables/anti_uav_rk3588s_frozen_p3_addon_p2_final_20260904/metadata/calibration/dataset_no_Video00004.txt
```

CLI 尺寸是 **H,W = 544,960**，文档中展示的是 **W×H = 960×544**，不要写反。导出应确认 3 个尺度、9 个张量；均值 `[0,0,0]`，标准差 `[255,255,255]`，padding=114。原 INT8 使用 384 张校准图，未使用 Video00004 / Video00009；相同清单校验值见 `model_manifest.json`。

编译器曾报告异常权重值量化警告，已保留 build log。`p3_equivalence.json` 只证明一个固定 256×256 测试输入上 FP32 的冻结 P3 分支与本 P3 模型输出相同，不代表 INT8 精度无损或与含 P2 模型整体相同。

## 10. 验收、排错与回退

验收需记录源码提交、模型/库哈希、完整命令、相机格式、NPU/CPU/DDR 频率和温度，并保留 summary、CSV。重点核对 `frames=12000`、`measured_frames=11900`、`runtime=native_cpp_no_python`、`args.detector_only=false`、`workers=3`、`inflight=3`、`npu_core_masks=[0,1,2]`、`policy=latest`、`buffers=4`、`pyramid_cache=true`。核掩码在 JSON 中是字符串数组。

离线审计在有 Python 的工作机/服务器执行，不是板端推理依赖。同一个审计根目录只放同模型的测试，否则会拒绝混合模型：

```bash
python3 scripts/anti_uav/summarize_camera_benchmark.py \
  deliverables/expanded28_p3_camera_20260921
```

| 现象 | 处理 |
| --- | --- |
| 只有约 49 FPS | 检查是否误设为 1 worker；检查 core mask、并发数和频率 |
| FPS 接近 120，但延迟不是 8 ms | 正常：吞吐间隔和单帧延迟不同，查看 driver_to_output |
| `Output already exists` | 更换新输出目录，不要覆盖已有测量 |
| 摄像头超时/格式错误 | 检查节点、占用、驱动/DTB、raw8 格式和权限，不要盲改像素格式 |
| 动态库缺失 / fused 符号缺失 | 检查三库与程序是否同版、RPATH/LD_LIBRARY_PATH 和软链接目标 |
| 有 detection 但无 track | 检查 new_score、确认/匹配状态；区分所有检测与已确认 ID 输出 |
| ID 不稳定 | 检查按序输入、帧间跳跃、实际输入帧率及 GMC 质量；不要混流或每帧重建 tracker |
| 连续运行变慢 | 查温度、实际时钟、散热/供电和其他任务；保持温控开启 |

回退到原 P3+P2 模型，使用已有独立部署，不覆盖 P3：

```bash
cd /home/orangepi/ultralytics_yolov8_ziye_code
bash scripts/anti_uav/dist_native/run_camera_fast_start_on_board.sh \
  --frames 12000 \
  --output /home/orangepi/deployments/expanded28_camera_latency_20260921/rollback_$(date +%Y%m%d_%H%M%S)
```

以上回退命令假定未额外设置 `ANTI_UAV_CAMERA_DEPLOY`；若已设置，请先清除或明确设为 P3+P2 部署目录。保留旧模型与依赖，不做破坏性清理。

参考证据：本目录 `p3_latest3_12000/` 与相邻 `expanded28_camera_latency_20260921/overlap3_12000/`。Dist 上游版本及第三方许可见仓库 `scripts/anti_uav/dist_native/README.md` 和 `third_party/lap/LICENSE`；向外分发源码、库或固件前由交付方确认相应许可义务，本文件不代替许可审查。
