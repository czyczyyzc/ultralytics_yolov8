# RK3588S 无人机检测与跟踪部署交接

版本：2026-09-29。对象：负责 RK3588S 板端部署和验收的同事。

## 1. 先选对模型

目前有三组**不同的权重**，不能把它们的精度和速度混写：

| 版本 | P3 检测器 | Frozen-P3 + Add-on P2 检测器 | 交付状态 |
| --- | --- | --- | --- |
| 28 视频，已在 RK3588S 实测 | 已有 960×544 INT8 RKNN | 已有 960×544 INT8 RKNN | 可直接复现板端 C++/Dist/GMC 流水线；下文速度只属于此版 |
| 40 视频 + 328 张候选贴图在线增强 | `.pt` 与 INT8 RKNN 均已导出 | `.pt` 与 INT8 RKNN 均已导出 | 在当前独立 FP32 测试中优于 44 视频版；**新 RKNN 尚未完成板端精度/速度验收** |
| 最新 44 视频：保留旧 40 视频的在线贴图增强，追加 4 个原图视频 | `.pt` 与 INT8 RKNN 均已导出 | `.pt` 与 INT8 RKNN 均已导出 | FP32 独立测试误检明显增加；**仅供实验，不建议直接替换线上模型** |

建议：需要马上在板上运行时，使用已验证的 28 视频 RKNN；准备下一次精度升级时，先验收已导出的 40 视频 RKNN，不要因为 44 视频时间更新就自动切换。P3 速度更高；P2+P3 更照顾 4–8 px 目标，但可能带来更多误检。没有在同一评测中得到的 28 与 40/44 精度，不能直接排成一张排名表。

## 2. 模型文件与版本

47 服务器代码根目录：`/mnt/chenziye/codes/ultralytics_yolov8`；本次核对的仓库 HEAD：`d9c524dcb40740daadd776c0d4cd2fd1efc0d365`，44 视频训练记录的执行提交为 `f9628a044cccbd122954de3408e9f84d8cf8fd51`。两轮均为先训练 P3 15 epoch，再冻结 P3 训练 Add-on P2 15 epoch；40 视频旧清单及重复次数被保留，新增 4 个视频以原图进入训练，旧 40 视频部分继续以 `p=0.50` 使用 328 张候选贴图的在线替换缓存。**不能说全部 44 个视频都做了贴图替换。** 以下均是该服务器上的**绝对路径**。`.pt` 用于训练/导出，板端 C++ 程序读取 `.rknn`，不直接读取 `.pt`。

| 版本 | 模型 | 文件 | SHA256 |
| --- | --- | --- | --- |
| 28 视频 | P3 `.pt` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28/training_p3/p3/weights/best.pt` | `8c551c6b115ee9d93c9625280b1d5326b8264d4040e6b69c973af3939f63863f` |
| 28 视频 | P3 INT8 | `/mnt/chenziye/codes/ultralytics_yolov8/deliverables/expanded28_p3_camera_20260921/detector_p3_960x544_int8.rknn` | `a4a27a12047ba1ac287ef320d398e5c935c1e8d3e43d49fddee7e02b931af20c` |
| 28 视频 | P2+P3 `.pt` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28/training_addon/p2/weights/best.pt` | `61e9a3669f964e59e96e9b2a24bf7705b945e184ed5a0628be33ee44d8b4bb59` |
| 28 视频 | P2+P3 INT8 | `/mnt/chenziye/codes/ultralytics_yolov8/deliverables/expanded28_rk3588_dist_20260920/detector_960x544_int8.rknn` | `2afa76f9f093265b0e347d26285ed15c98c449d958eed040f27e7345f29deb53` |
| 40 视频 | P3 `.pt` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online328_20260922/training_p3/p3/weights/best.pt` | `c4b6d9a34d669694f49ee46fd22c26a6a67766b7955828b6f01d3827222fa94b` |
| 40 视频 | P2+P3 `.pt` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online328_20260922/training_addon/p2/weights/best.pt` | `6aec79be8e1562d5cbab4f7b1427d156edbd4fff8dba3095302b17b85aac6472` |
| 44 视频 | P3 `.pt` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_p3/p3/weights/best.pt` | `761ba22431d2062a9fde1cfda08c5e394e2a6f34b60b9328b50ae9860e0b4471` |
| 44 视频 | P2+P3 `.pt` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_addon/p2/weights/best.pt` | `8f125903f223810c611bb97f5de3a51c062dab6c4cb3f67420d3d92a3100c1dd` |

P3 指 P3/P4/P5 三个尺度（stride 8/16/32）；P2+P3 指在冻结的 P3 主干/检测分支之外新增 P2 路径，实际有 P2/P3/P4/P5 四个尺度（stride 4/8/16/32）。两者不是通过运行时开关启停同一个 RKNN 图：要省掉 P2 的计算，必须使用独立导出的 P3 RKNN。

40/44 视频的四份新 RKNN 已于 2026-09-30 用 Toolkit2 2.3.2、目标 `rk3588`、384 张灰度校准图生成。服务器公共前缀为：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved_rknn_exports_20260930/
```

| 版本/结构 | 前缀下相对模型路径 | RKNN SHA256 | ONNX 输出 |
| --- | --- | --- | ---: |
| 40 视频 P3 | `40_p3/detector_960x544_int8.rknn` | `b8d4517946b745d0bdc60cde45c0955b58111e31442521be96ab38e03f378fe0` | 9 |
| 40 视频 P2+P3 | `40_p2p3/detector_960x544_int8.rknn` | `5851dd1d85f511a35500461680142a01189d4f5a3043aeeed9a2e8bf3f63486c` | 12 |
| 44 视频 P3 | `44_p3/detector_960x544_int8.rknn` | `011373d915c7b05c8cf793be21da87ffdb7bdda840bcbb7fdc3233eaa20d200c` | 9 |
| 44 视频 P2+P3 | `44_p2p3/detector_960x544_int8.rknn` | `95c4cf02d73ec9b6056f96545c2d3bd4aa195cb084607711f3840912956133e5` | 12 |

本地副本在 `deliverables/anti_uav_approved40_44_rknn_20260930/<版本_结构>/`，并附 `.rkopt.json`。四份文件均完成量化编译且 Toolkit2 可解析加载；**尚未在 RK3588S 板上执行推理，不能给出它们的 INT8 精度或 FPS**。编译器提示权重离群值可能影响量化精度，部署前须按第 7 节验收。

## 3. 检测精度：40 与 44 视频同条件对比

以下为 47 服务器上的 **FP32 detector-only**，输入 960×544，NMS IoU=0.45，最多 100 框；Precision/Recall/FP/4–8 px Recall 取 `conf=0.03`。mAP50 在评测时以 `conf=0.001` 生成曲线，**不是 conf=0.03 下的单点指标**。Video00009 是权重选择用灰度验证集，表中为其原尺寸帧部分；Video00004 是未加入训练的独立测试视频。这里没有 tracker、RKNN 量化或摄像头性能。

| 数据 | 权重 | Precision | Recall | FP | mAP50 | 4–8 px Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Video00009 | 40 视频 P3 | 68.75% | 53.83% | 195 | 60.31% | 49.49% |
| Video00009 | 44 视频 P3 | 68.82% | 53.45% | 193 | 58.07% | 47.70% |
| Video00009 | 40 视频 P2+P3 | 66.19% | 75.91% | 309 | 71.95% | 77.81% |
| Video00009 | 44 视频 P2+P3 | 64.66% | 72.77% | 317 | 67.93% | 74.74% |
| Video00004 | 40 视频 P3 | 45.27% | 75.89% | 411 | 70.93% | 73.33% |
| Video00004 | 44 视频 P3 | 32.11% | 78.12% | 740 | 57.20% | 75.80% |
| Video00004 | 40 视频 P2+P3 | 36.62% | 99.55% | 772 | 95.19% | 99.51% |
| Video00004 | 44 视频 P2+P3 | 16.27% | 99.78% | 2300 | 85.63% | 99.75% |

Video00004 上新 P2+P3 的 Recall 仅增加 0.23 个百分点，FP 从 772 增至 2300；不能仅凭召回率说 44 视频模型更好。其他阈值 `0.01/0.05`、逐模型 JSON 和计算口径见：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/cross_comparison/COMPARISON.md
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/cross_comparison/results.json
```

上述测试视频没有大面积目标的充分覆盖，不可由这些分数推断“目标占画面 80%”时的性能。

## 4. 板端速度与延迟：仅限已验证的 28 视频 INT8

同一块 RK3588S、同一灰度摄像头、同一原生 C++ Dist+GMC 程序，各连续处理 12,000 帧，前 100 帧不计入稳态统计。三套独立 RKNN context 绑定 NPU core 0/1/2，跨帧并行；输入 960×544 INT8、`conf=0.03`。只更换 P3 与 P2+P3 检测图。相机顺序实时采集，不是同一批像素帧回放。

| 指标 | 28 视频 P3 | 28 视频 P2+P3 |
| --- | ---: | ---: |
| 完整流水线平均 FPS | 120.00 | 86.58 |
| 单帧 NPU 推理平均 | 21.20 ms | 31.56 ms |
| 开始读帧至完整结果，平均 | 24.94 ms | 34.53 ms |
| 驱动帧时间戳至完整结果，平均 | 29.91 ms | 43.96 ms |
| 驱动帧时间戳至完整结果，P95 | 31.84 ms | 49.59 ms |
| 处理帧之间源帧跳过比例 | 0.0% | 27.8% |

120 FPS 接近这台摄像头的供帧上限，不是 P3 模型的离线极限 FPS。时间戳不是经验证的曝光时刻；以上不含显示、编码、网络。室内测速视频没有标注无人机目标，**只能证明板端运行速度，不能证明 Dist 跟踪准确率或 28 视频模型精度**。三核提高吞吐率，不意味着单帧延迟除以 3。原始测试记录：

```text
/mnt/chenziye/codes/ultralytics_yolov8/deliverables/expanded28_p3_camera_20260921/
本地仓库 deliverables/expanded28_camera_latency_20260921/overlap3_12000/summary.json
```

## 5. C++ 跟踪链路与运行参数

代码位于服务器仓库（本地同名路径）`scripts/anti_uav/dist_native/`：`video.cpp` 是顺序调度与结果输出，`tracker.cpp` 是 Dist 公开代码适配版，`gmc.cpp` 是相机运动补偿，`build_camera.sh` 负责构建程序和动态库；检测器封装位于 `scripts/anti_uav/rknn_yolov8_native/detector_c_api.cpp`。Dist/GMC **没有额外神经网络权重**。这是公开 Dist 仓库实际执行路径的 C++ 移植，不声称实现论文中所有模块；上游代码基准 `396c359e1aa8be4fd5e81a02626cb1ee3867cf7c`，许可见 `dist_native/README.md`。

流程为 V4L2 灰度 raw8 取帧 → 960×544 RGB 输入准备 → 三个 RKNN worker 检测；GMC 在独立顺序线程与检测并行 → 按采集顺序做 Dist 关联 → 输出。NPU 完成顺序可以不同，但不能乱序更新同一个 tracker。每条视频流必须持有独立 GMC/Dist 状态。无需 Python、PyTorch、IMU 或姿态输入；Python 仅用于服务器导出与离线评测。

已测参数：`conf=0.03`、NMS IoU=0.45、最多 100 框；Dist `high=0.03, low=0.01, new_score=0.10, match=0.8, buffer=30`（按源 FPS 缩放），无 ReID。`low=0.01` 不代表 0.01–0.03 的候选会进入 tracker：检测器先以 0.03 截断。新轨迹还受 `new_score` 和确认状态约束，因此“有检测但没有 ID 框”不必然是检测丢失。GMC 使用稀疏光流和部分仿射估计；失败时回退单位变换。当前只输出本帧匹配到的跟踪框，不把 Kalman 预测框伪装成检测。

## 6. 在已验证 RK3588S 板上复现

既有板端代码目录：`/home/orangepi/ultralytics_yolov8_ziye_code`；既有 P3 部署目录：`/home/orangepi/deployments/expanded28_p3_camera_20260921`；P2+P3 部署目录：`/home/orangepi/deployments/expanded28_camera_latency_20260921`。这些是当时已验证的板端目录，**新板不能假定同一路径、摄像头节点或库依赖已经存在**。示例 SSH 地址曾为 `orangepi@192.168.144.50`，迁移后须重新确认；账户凭据不写入交接文档。上节的实测二进制源提交为 `9ded0bd`；当前仓库 HEAD 更新不等于已对重编译后的程序复测。

在既有板上，分别执行下面两项；每次使用全新的输出目录，不要并发跑两个测速进程：

```bash
cd /home/orangepi/ultralytics_yolov8_ziye_code
P3_OUT=/home/orangepi/deployments/expanded28_p3_camera_20260921/check_$(date +%Y%m%d_%H%M%S)
bash scripts/anti_uav/dist_native/run_camera_p3_on_board.sh \
  --frames 12000 --warmup 100 --output "$P3_OUT"

P2_OUT=/home/orangepi/deployments/expanded28_camera_latency_20260921/check_$(date +%Y%m%d_%H%M%S)
bash scripts/anti_uav/dist_native/run_camera_fast_start_on_board.sh \
  --frames 12000 --warmup 100 --output "$P2_OUT"
```

默认使用 `/dev/video11`、`raw8-gray`、latest 取帧、4 个缓冲、3 worker、3 in-flight、`--camera-start overlap`、fused 预处理。`--camera-fps 120` 是 tracker 的源帧率配置，不会把实际传感器设成 120 FPS。执行前核对实际相机格式、NPU 驱动、模型哈希及 `ldd`；不能把彩色 Bayer 当成物理灰度输入。输出目录中的 `summary.json`、`latency.csv` 是测速证据；逐帧检测/ID/GMC 矩阵只有显式加 `--save-observations` 才写入 `observations.jsonl`。本程序不是 REST API、RTSP 服务或已验收的无限运行守护进程。

板端已验证基线为 Ubuntu 22.04 / 厂商内核 5.10.198 / RKNN Runtime 与导出 Toolkit 2.3.2 / NPU driver 0.9.8。新板先确认内核、NPU、相机驱动和 OpenCV/RKNN 动态库匹配；勿覆盖内核或关闭温控来复刻 FPS。详细编译、依赖迁移、首帧与排错说明见 `deliverables/expanded28_p3_camera_20260921/DEPLOYMENT_HANDOVER_ZH.md`。

## 7. 新 RKNN 的复现导出与待做验收

四份新 RKNN 已生成；以下命令用于**重新导出或更换校准集时复现**，不是再次部署所必需。尚未完成的是 INT8 精度和板端验收。不要在训练目录或已有导出目录原位覆盖。以下在 47 服务器仓库执行，示例为 40 视频 P3；P2+P3 或 44 视频时替换 `PT` 和独立 `OUT`：

```bash
cd /mnt/chenziye/codes/ultralytics_yolov8
PT=/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online328_20260922/training_p3/p3/weights/best.pt
OUT=/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/export40_p3_rebuild_$(date +%Y%m%d_%H%M%S)
CAL=deliverables/anti_uav_rk3588s_frozen_p3_addon_p2_final_20260904/metadata/calibration/dataset_no_Video00004.txt
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python \
  scripts/anti_uav/export_detector_rkopt_onnx.py \
  --weights "$PT" --imgsz 544,960 --output "$OUT/detector.onnx"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python \
  scripts/anti_uav/build_rknn.py --onnx "$OUT/detector.onnx" \
  --output "$OUT/detector_960x544_int8.rknn" --target rk3588 \
  --quantize --dataset "$CAL"
sha256sum "$PT" "$OUT/detector_960x544_int8.rknn"
```

`--imgsz` 采用 **H,W=544,960**；导出后确认 P3 为 9 个输出、P2+P3 为 12 个输出。该 384 图校准清单未包含 Video00004/Video00009，但源于较早的数据分布；它可作起点，不能替代新版本的量化后精度检查。Toolkit2 在本次服务器上不支持从已导出的 `.rknn` 直接启动主机模拟推理，不能把成功 `load_rknn()` 视为板端验证。新模型应分别进行 FP32/INT8 同帧框与分数对照、Video00004/Video00009 低阈值 Precision/Recall/FP/4–8 px Recall 回归、RK3588S 目标场景实拍、至少 12,000 帧速度/温度/丢帧测试。通过后为 P3/P2+P3 分别建立独立部署目录，保存模型 SHA、C++ 构建提交、运行库版本和测试报告；不要直接覆盖已验证的 28 视频模型或把上节 28 视频 FPS 标成新模型 FPS。

## 8. 交接验收清单

- 确认部署的是 **P3 还是 P2+P3、28/40/44 哪一组权重**，并核对 `.pt`/`.rknn` SHA256；P2+P3 的 RKNN 不得误当 P3 三尺度模型加载。
- 确认输入是宽 960、高 544；记录相机原始分辨率、灰度/彩色格式、letterbox 规则、conf/NMS/max-det。
- 确认三核掩码为 `0,1,2`、三个独立 context、按采集顺序运行 GMC/Dist；记录真实 FPS、驱动时间戳至结果延迟及丢帧。
- 精度、吞吐、首帧和跟踪 ID 稳定性分别验收；不得用室内空场测速代替目标跟踪验收，也不得用 FP32 指标声称 INT8 精度相同。
- 向外分发 C++ 程序或第三方跟踪代码之前，检查 Dist/Ultralytics 与 vendored LAP 的许可义务。
