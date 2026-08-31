# Anti-UAV RK3588S 部署说明

文档版本：1.4

发布日期：2026-08-31

适用平台：Orange Pi CM5 / RK3588S / aarch64 Linux

部署基线：YOLOv8n `960x544 (WxH)` INT8 + 原生 C++ RKNN 推理 + RK-BoT-SORT + 3 NPU 核并行

模型更新说明：当前默认模型是 `neg15` recall-safe 数据上的困难正样本增量训练 epoch 1。相对原 `neg15 e10` INT8，在 Video00004、板端 `conf=0.01` 下 Recall 从 `72.54%` 提升到 `75.00%`，Precision 从 `81.05%` 提升到 `81.16%`。原 `neg15 e10` 和 `2026-08-29 final_all_gray` 模型保留作回滚，不作为默认配置。

## 1. 部署范围

本方案用于可见光或灰度视频中的无人机目标检测和连续 ID 跟踪。部署程序不依赖无人机姿态、相机姿态、IMU 或 ReID 模型，输出检测框、置信度、Track ID 和时间戳，可由上层系统继续完成目标告警、无线目标匹配或可疑目标判定。

当前生产链路固定如下：

| 项目 | 配置 |
|---|---|
| Detector | YOLOv8n，单类别 `drone` |
| 模型输入 | `960x544 (WxH)`，RKNN 中显示为 `[544, 960]` |
| 模型格式 | RKNN INT8，RKNN-Toolkit2 2.3.2 |
| 板端 Runtime | `librknnrt` 2.3.2 |
| 推理实现 | C++17，RKNN C API，native INT8 output，zero-copy I/O |
| 预处理 | OpenCV letterbox + BGR-to-RGB，直接写入 RKNN input memory |
| 并行方式 | 3 个 RKNN context，分别绑定 NPU core 0、1、2 |
| Tracker | RK-BoT-SORT detector-based tracker |
| 姿态输入 | 不需要，CameraMotion 默认为 identity |
| ReID | 不使用 |

说明：当前 C++ 链路已使用 RKNN 输入和输出 zero-copy，但图像 resize/color conversion 仍由 OpenCV 在 CPU 上完成，本版本没有启用 RGA 预处理。交付或验收时不应将其描述为 RGA 版本。

## 2. 代码与数据位置

### 2.1 Git 仓库

| 环境 | 路径 |
|---|---|
| GitHub | `git@github.com:czyczyyzc/ultralytics_yolov8.git` |
| 本地 Mac | `/Users/czyczyyzc/Documents/codes/ultralytics_yolov8` |
| 47 训练服务器 | `/mnt/chenziye/codes/ultralytics_yolov8` |
| RK3588S 推荐目录 | `/data/anti_uav/src/ultralytics_yolov8` |

部署时应固定 Git commit，并将 commit ID 写入发布清单，不要直接以可变的 `main` 作为版本号。

### 2.2 关键源码

| 文件 | 用途 |
|---|---|
| `scripts/anti_uav/rknn_yolov8_native/native_yolov8_video.cpp` | 视频读取、letterbox、RKNN zero-copy 推理、INT8 解码、NMS、多核调度和有序输出 |
| `scripts/anti_uav/rknn_yolov8_native/detector_based_tracker.hpp` | 无姿态输入 RK-BoT-SORT 实现 |
| `scripts/anti_uav/rknn_yolov8_native/build_on_board.sh` | 板端 C++ 编译脚本 |
| `scripts/anti_uav/rknn_yolov8_native/run_real_gray_rk_botsort.sh` | 批量灰度视频推理示例 |
| `scripts/anti_uav/rknn_yolov8_native/render_tracker_video.py` | 根据 tracks CSV 离线生成可视化视频 |
| `scripts/anti_uav/export_detector_rkopt_onnx.py` | `.pt` 导出 RKNN 优化的 9-output ONNX |
| `scripts/anti_uav/prepare_rknn_calibration.py` | 生成 INT8 calibration 图片和 `dataset.txt` |
| `scripts/anti_uav/build_rknn.py` | 使用 RKNN-Toolkit2 生成 `.rknn` |
| `scripts/anti_uav/rk3588_set_performance.sh` | 设置 CPU/NPU/DDR performance governor |
| `scripts/anti_uav/rk3588_performance_governor.service` | 开机自动设置 performance governor |
| `scripts/anti_uav/build_real_gray_yolo_lovo.py` | 构建 Anti-UAV300 + 灰度正样本 LOVO 数据 |
| `scripts/anti_uav/train_real_gray_yolo_lovo_fold.py` | `544x960` 灰度适配训练 |
| `scripts/anti_uav/evaluate_real_gray_yolo_lovo_fold.py` | 灰度 holdout 评测 |
| `scripts/anti_uav/mine_yolo_hard_positives.py` | 挖掘漏检、定位失败和低置信灰度正样本 |
| `scripts/anti_uav/build_hard_positive_rehearsal_mix.py` | 保持 neg15 不变并重排困难正样本 rehearsal 槽位 |
| `scripts/anti_uav/rknn_yolov8_native/run_recall_safe_deployment.sh` | 固定生产阈值的板端启动脚本 |

### 2.3 训练和验证数据

| 数据 | 服务器路径 |
|---|---|
| Anti-UAV300 | `/mnt/chenziye/datasets/anti_uav/Anti-UAV300` |
| Anti-UAV300 YOLO | `/mnt/chenziye/datasets/anti_uav/anti_uav300_yolo` |
| 灰度训练数据 | `/mnt/andrew/anti_uav_model_refinement/data/real_gray_adapt_v2_lovo_50_50` |
| 灰度 LOVO 混合数据 | `/mnt/andrew/anti_uav_model_refinement/data/real_gray_yolo_lovo_positive_mixed_v1_20260828` |
| 灰度标注与验证 registry | `/mnt/andrew/anti_uav_model_refinement/external_eval/real_gray_tracker_eval_v2` |
| 原始灰度视频 | `/mnt/andrew/anti_uav_model_refinement/external_eval/real_gray_mp4` |
| LOVO 训练结果 | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolo_lovo_mixed_20260828` |
| LOVO 汇总报告 | `deliverables/real_gray_yolo_lovo_mixed_20260828/summary.md` |

## 3. 当前交付版本

本次交付从原 `neg15 e10` 开始做困难正样本增量训练。训练集保持 `70,650` 个正样本槽位和 `12,468` 个负样本，负样本比例仍为 `15.00036%`。从 `8,424` 个唯一灰度正样本中挖掘出 `3,012` 个困难样本，并使用 `7,930` 个原有重复正样本槽位做 rehearsal。10 个增量 epoch 中选择第 1 个 epoch，对应训练文件 `weights/epoch0.pt`。

完整灰度回放包含 final 训练所覆盖的视频，只用于模型选择和适配覆盖检查，不作为独立泛化指标。旧 7-fold LOVO 的 `8.32%` macro mAP50-95 只能作为前一阶段基线；新困难正样本模型尚未完成逐 fold 独立重训，不能把旧 LOVO 数字作为新模型结果。

### 3.1 发布清单

| 项目 | 值 |
|---|---|
| 发布标识 | `neg15_hardpos_e1_544x960_20260831_v232_int8` |
| 发布类型 | recall-safe neg15 + 灰度困难正样本增量训练版 |
| Git commit | `eec8b01` |
| 训练源 checkpoint | `real_gray_yolov8n_neg15_hard_positive_20260831/.../weights/epoch0.pt` |
| 发布 PT 文件 | `yolov8n_recall_safe_neg15_hardpos_e1.pt` |
| 发布 PT SHA256 | `428807cb6214a3af8f6f87775291152058728f5042a45cfe2ffdc2b0f541a0ea` |
| RK-optimized ONNX | `yolov8n_recall_safe_neg15_hardpos_e1_544x960_rkopt.onnx` |
| ONNX SHA256 | `00867478b302bffbb221f3faab2e27f49c25706702449bf118a8d5bad89fdba5` |
| RKNN | `yolov8n_recall_safe_neg15_hardpos_e1_544x960_v232_int8.rknn` |
| RKNN SHA256 | `3d5a47132cd7b087e03170ba398db7a57af730d834ebb0c8d0c77d81e776f98e` |
| Toolkit | RKNN-Toolkit2 `2.3.2`，compiler `2.3.2` |
| 目标平台 | `rk3588`，RK3588/RK3588S |
| 输入 | RGB NHWC，`1x544x960x3`，INT8 量化 |
| 输出 | 9 个 native INT8 tensors |
| 检测类别 | `drone`，class ID `0` |

服务器发布目录：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/
  rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/
```

关键文件绝对路径：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_neg15_hard_positive_20260831/training/real_gray_yolo_recall_safe_neg15_hardpos_v1_20260831/weights/epoch0.pt
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/yolov8n_recall_safe_neg15_hardpos_e1_544x960_v232_int8.rknn
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/yolov8n_recall_safe_neg15_hardpos_e1_544x960_rkopt.onnx
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/yolov8n_recall_safe_neg15_hardpos_e1.pt
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/model.rkopt.json
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/model_build.log
```

注意：训练文件 `epoch0.pt` 表示本次增量训练完成第 1 个 epoch。10 个 checkpoint 已按相同灰度固定阈值评测，epoch 1 的 Recall/F1 最优；不要用 RGB 验证产生的 `best.pt` 或最后的 `last.pt` 替换发布模型。

建议复制到板端：

```text
/data/anti_uav/models/yolov8n_recall_safe_neg15_hardpos_e1_544x960_v232_int8.rknn
```

交付目录建议保持以下结构：

```text
anti_uav_yolov8n_neg15_hard_positive_544x960_20260831/
  models/yolov8n_recall_safe_neg15_hardpos_e1.pt
  rknn/yolov8n_recall_safe_neg15_hardpos_e1_544x960_rkopt.onnx
  rknn/yolov8n_recall_safe_neg15_hardpos_e1_544x960_v232_int8.rknn
  rknn/model.rkopt.json
  rknn/model_build.log
  COMPARISON.md
  SHA256SUMS
```

### 3.2 转换和验证状态

| 检查项 | 状态 | 说明 |
|---|---|---|
| `.pt` LOVO 独立集评测 | 待完成 | 新困难正样本版本尚未逐 fold 独立重训 |
| `.pt` Video00004 回放 | 通过 | conf0.03：P `89.53%`、R `72.54%`、F1 `80.15%` |
| ONNX 结构检查 | 通过 | 输入 `544x960`，9 outputs |
| RKNN INT8 构建 | 通过 | 448 张灰度 calibration 图片，7 段视频各含正/负样本 |
| RKNN 输出结构检查 | 通过 | P3/P4/P5 为 `68x120`、`34x60`、`17x30` |
| RK3588S 精度回归 | 通过 | conf0.01：P `81.16%`、R `75.00%`、F1 `77.96%` |
| RK3588S 完整视频验证 | 通过 | Video00004 共 2,359 帧，detector 和 Tracker 均完成 |
| 当前模型板端 FPS | 通过但余量有限 | 三核 detector `114.75 FPS`；三核 + RK-BoT-SORT `108.51 FPS` |

当前文件可用于 RK3588S 软件集成和板端回归验收。新场景正式发布前仍需完成困难正样本版本的 7-fold LOVO；任何模型替换都必须更新 SHA256，并重新执行第 15 节验收。

模型验收条件：

1. 输入必须为 `[1, 544, 960, 3]` 或等价 NHWC 描述。
2. 输出必须为 9 个 RK-optimized tensors，即 P3/P4/P5 各自的 bbox、class 和 score-sum。
3. P3/P4/P5 逻辑网格应为 `68x120`、`34x60`、`17x30`。
4. 模型必须为 INT8 native output；C++ 程序会拒绝非 INT8 native output。
5. `.pt`、ONNX 和 RKNN 必须在同一组灰度视频帧上完成结果回归。
6. 发布包必须记录模型 SHA256、Git commit、Toolkit/Runtime 版本和板端 benchmark JSON。

## 4. PC/服务器端模型生成

以下命令用于复现第 3 节的当前交付模型。RKNN-Toolkit2 安装在 x86_64 Linux 转换机上，板端只安装 RKNN Runtime。转换机应固定使用 RKNN-Toolkit2 2.3.2，不要混用 1.4.0 或其他版本生成交付模型。

### 4.1 导出 RK-optimized ONNX

```bash
cd /mnt/chenziye/codes/ultralytics_yolov8
source .venv/bin/activate

python scripts/anti_uav/export_detector_rkopt_onnx.py \
  --weights /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_neg15_hard_positive_20260831/training/real_gray_yolo_recall_safe_neg15_hardpos_v1_20260831/weights/epoch0.pt \
  --imgsz 544,960 \
  --output /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/yolov8n_recall_safe_neg15_hardpos_e1_544x960_rkopt.onnx \
  --metadata /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/model.rkopt.json \
  --opset 12
```

检查 `model.rkopt.json`，`output_count` 必须为 9，三个分支形状必须与第 3 节一致。

### 4.2 准备 INT8 calibration 数据

Calibration 图片应来自实际灰度相机，覆盖目标出现、目标消失、低对比度、海天线、云层、海面高光和不同时间段。建议使用 200 至 500 张代表性图片，并避免全部图片来自同一个连续片段。

单视频采样示例：

```bash
python scripts/anti_uav/prepare_rknn_calibration.py \
  --source /mnt/andrew/anti_uav_model_refinement/external_eval/real_gray_mp4/Video00004.mp4 \
  --output-dir /path/to/release/calibration/images \
  --dataset-txt /path/to/release/calibration/dataset.txt \
  --max-frames 128 \
  --frame-step 10 \
  --width 960 \
  --height 544 \
  --prefix Video00004
```

正式转换使用发布目录中的 `calibration/dataset.txt`：共 448 张图片，覆盖 7 段灰度视频，每段各采样 32 张有目标帧和 32 张无目标帧。预处理为 `1920x1080 -> 960x540` 等比例缩放，并在上下各填充 2 像素形成 `960x544`；`dataset.txt` 每行一个绝对图片路径。

### 4.3 生成 INT8 RKNN

```bash
/mnt/chenziye/miniconda3/envs/rknn232/bin/python scripts/anti_uav/build_rknn.py \
  --onnx /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/yolov8n_recall_safe_neg15_hardpos_e1_544x960_rkopt.onnx \
  --output /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/yolov8n_recall_safe_neg15_hardpos_e1_544x960_v232_int8.rknn \
  --target rk3588 \
  --quantize \
  --dataset /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_recall_safe_neg15_e10_544x960_20260831_v232_int8/calibration/dataset_balanced_224pos_224neg.txt \
  --mean-values 0,0,0 \
  --std-values 255,255,255 \
  --verbose 2>&1 | tee /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/model_build.log

sha256sum /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/rknn_yolov8n_neg15_hardpos_e1_544x960_20260831_v232_int8/*.rknn
```

转换日志必须显示 RKNN-Toolkit2 2.3.2。发现 outlier warning 时应做 RKNN simulator 和板端精度回归，不能只确认转换命令返回成功。

## 5. RK3588S 板端环境

已验证参考环境：

| 项目 | 版本/配置 |
|---|---|
| Board | Orange Pi CM5 RK3588S |
| Kernel | `6.1.99-rockchip-rk3588` |
| RKNN Runtime | 2.3.2 |
| NPU driver | `0.9.8` |
| NPU performance clock | 运行中应达到 1,000 MHz |
| DDR performance clock | 运行中应达到 2,112 MHz |

创建目录：

```bash
sudo mkdir -p /data/anti_uav/{bin,models,metadata,config,videos,output,logs,src}
sudo chown -R "$(id -un):$(id -gn)" /data/anti_uav
```

本次参考板识别到一张 125G 可用容量的 SD 卡，但在 `DISCARD` 后和后续 fsck 读取中均出现 MMC `-110`/I/O error，因此该卡已从 fstab 移除，最终板端验收使用系统盘上的 `/data`，可用空间约 21G。不要把该卡用于代码、环境或录像归档，需更换并完成全盘读写验证后再迁移。

更换合格 SD 卡后，建议将其挂载为 `/data`。如果卡或控制器不支持稳定的 `DISCARD`，fstab 应包含 `X-fstrim.notrim`：

```text
UUID=<replacement-card-uuid> /data ext4 defaults,noatime,nofail,X-fstrim.notrim,x-systemd.device-timeout=10 0 2
```

```bash
sudo systemctl disable --now fstrim.timer
findmnt /data
df -hT /data
```

使用 `blkid` 获取新卡 UUID，不要复用上述占位符或故障卡 UUID。不要对已有数据的卡直接执行 `mkfs`；新卡格式化和迁移后，应执行全盘容量/读写测试、`fsck -f`、模型 SHA256 校验，并重启确认自动挂载。只做少量抽样写入不足以排除坏卡或扩容假卡。

安装编译依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libopencv-dev ffmpeg git
```

安装与 Toolkit 2.3.2 匹配的 aarch64 `librknnrt.so` 和 `rknn_api.h`。如果 runtime library 不在系统库目录，可通过 `RKNN_LIB_DIR` 指定；不要为了运行 C++ 程序在板端安装完整 RKNN-Toolkit2。

建议目录：

```text
/data/anti_uav/third_party/rknn/include/rknn_api.h
/data/anti_uav/third_party/rknn/lib/librknnrt.so
```

## 6. 获取并编译代码

```bash
git clone git@github.com:czyczyyzc/ultralytics_yolov8.git \
  /data/anti_uav/src/ultralytics_yolov8

cd /data/anti_uav/src/ultralytics_yolov8
git checkout eec8b01

cd scripts/anti_uav/rknn_yolov8_native
RKNN_INCLUDE=/data/anti_uav/third_party/rknn/include \
RKNN_LIB_DIR=/data/anti_uav/third_party/rknn/lib \
./build_on_board.sh

install -m 755 build/native_yolov8_video /data/anti_uav/bin/native_yolov8_video
ldd /data/anti_uav/bin/native_yolov8_video
```

`ldd` 中的 `librknnrt.so` 和 OpenCV libraries 不得显示 `not found`。

## 7. 固化性能模式

```bash
cd /data/anti_uav/src/ultralytics_yolov8

sudo install -m 755 \
  scripts/anti_uav/rk3588_set_performance.sh \
  /usr/local/sbin/rk3588_set_performance.sh

sudo install -m 644 \
  scripts/anti_uav/rk3588_performance_governor.service \
  /etc/systemd/system/rk3588-performance-governor.service

sudo systemctl daemon-reload
sudo systemctl enable --now rk3588-performance-governor.service
sudo systemctl status rk3588-performance-governor.service --no-pager
```

重启后检查：

```bash
cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
cat /sys/class/devfreq/*npu/governor
cat /sys/class/devfreq/dmc/governor
```

输出应为 `performance`。NPU 和 DDR 实际频率应在推理负载运行时检查，而不是只在空闲状态读取。

## 8. 单视频部署测试

安装固定生产配置：

```bash
install -m 755 scripts/anti_uav/rknn_yolov8_native/run_recall_safe_deployment.sh \
  /data/anti_uav/bin/run_recall_safe_deployment.sh
install -m 644 scripts/anti_uav/rknn_yolov8_native/recall_safe_neg15.env \
  /data/anti_uav/config/recall_safe_neg15.env

/data/anti_uav/bin/run_recall_safe_deployment.sh \
  /data/anti_uav/videos/test.mp4 \
  /data/anti_uav/output/recall_safe_neg15
```

等价的完整命令如下：

```bash
MODEL=/data/anti_uav/models/yolov8n_recall_safe_neg15_hardpos_e1_544x960_v232_int8.rknn
VIDEO=/data/anti_uav/videos/test.mp4
OUT=/data/anti_uav/output/smoke

mkdir -p "$OUT"

/data/anti_uav/bin/native_yolov8_video "$MODEL" "$VIDEO" \
  --workers 3 \
  --core-mask 0_1_2 \
  --worker-cpu-base 4 \
  --queue-size 3 \
  --tracker rk_botsort \
  --source-fps 107 \
  --max-frames 1000 \
  --warmup-frames 50 \
  --conf 0.01 \
  --nms-iou 0.45 \
  --track-high-thresh 0.03 \
  --track-low-thresh 0.01 \
  --new-track-thresh 0.05 \
  --track-first-match-cost 0.92 \
  --track-second-match-cost 0.92 \
  --track-buffer-sec 1.0 \
  --track-prediction-sec 0.0 \
  --track-min-hits 2 \
  --output-json "$OUT/benchmark.json" \
  --predictions-csv "$OUT/detections.csv" \
  --tracks-csv "$OUT/tracks.csv"
```

以上阈值是当前集成基线。正式业务阈值应根据目标海域、相机和误报容忍度，在保留独立验证集的前提下重新标定，不应直接用训练集调参。

输出说明：

| 文件 | 内容 |
|---|---|
| `benchmark.json` | 输入尺寸、worker 数、总 FPS、各阶段耗时和跟踪统计 |
| `detections.csv` | 每帧最高置信度 detector box |
| `tracks.csv` | 每帧 Track ID、时间戳、box、置信度和状态 |

Tracker 模式下程序会把 detector 解码阈值自动降低到 `track-low-thresh`，高置信检测用于第一阶段关联，低置信检测用于第二阶段关联。

## 9. 三 NPU 核并行原理

`--workers 3` 不是把一帧模型切到三个 NPU 核，而是让连续三帧并行：

```text
Frame 0 -> RKNN context A -> NPU core 0 -> result 0
Frame 1 -> RKNN context B -> NPU core 1 -> result 1
Frame 2 -> RKNN context C -> NPU core 2 -> result 2
Frame 3 -> RKNN context A -> NPU core 0 -> result 3
```

C++ 程序使用 `rknn_dup_context` 创建三个 context，每个 worker 有独立 input/output memory。检测结果可能乱序完成，但 consumer 会按 `frame_index` 重新排序，再按顺序调用 RK-BoT-SORT，因此不会因为多核并行破坏跟踪时序。

参数建议：

| 场景 | workers | queue-size | 说明 |
|---|---:|---:|---|
| 低延迟实时视频 | 3 | 3 | 推荐生产配置 |
| 离线极限吞吐测试 | 3 | 12 | 吞吐略高，但缓存和延迟增加 |
| 单帧延迟测试 | 1 | 3 | 固定单个 NPU core |

三核并行提高的是连续视频吞吐，不会把单帧推理延迟缩短为三分之一。实时摄像头接入时，采集、解码、颜色转换和下游接口也必须持续达到目标帧率。

## 10. 摄像头和实时流

V4L2 摄像头示例：

```bash
/data/anti_uav/bin/native_yolov8_video \
  /data/anti_uav/models/yolov8n_recall_safe_neg15_hardpos_e1_544x960_v232_int8.rknn \
  /dev/video0 \
  --workers 3 \
  --queue-size 3 \
  --worker-cpu-base 4 \
  --tracker rk_botsort \
  --source-fps 107 \
  --conf 0.01 \
  --track-high-thresh 0.03 \
  --track-low-thresh 0.01 \
  --new-track-thresh 0.05 \
  --track-buffer-sec 1.0 \
  --track-prediction-sec 0.0 \
  --tracks-csv /data/anti_uav/output/camera_tracks.csv \
  --output-json /data/anti_uav/output/camera_summary.json
```

RTSP 可将第二个位置参数替换为实际 RTSP URL。当前程序通过 OpenCV `VideoCapture` 获取视频，生产接入前必须确认板端 OpenCV/FFmpeg 支持目标协议和编码格式。

`--source-fps 107` 用于相机或 RTSP 后端不能提供可靠时间戳时计算 tracker 的 `dt`。若相机实际帧率不是 107，应填写实测值，否则 track buffer 的时间语义会不准确。

## 11. RK-BoT-SORT 输出策略

默认配置保留 lost track 1 秒，用于目标短时漏检后恢复原 Track ID，但 `--track-prediction-sec 0.0` 不输出没有 detector 支持的预测框。这种配置可以降低海面和云层背景中产生虚假轨迹的风险。

需要在短时漏检期间继续输出预测框时，可设置：

```bash
--track-prediction-sec 0.15
```

该配置会增加预测框漂移风险，必须独立验收。无姿态输入时 `CameraMotion` 为 identity；如果相机安装在快速转动的云台上，应在后续版本接入图像稳像或相机运动补偿。

## 12. 离线可视化

推理完成后，可在有 Python、OpenCV 和 NumPy 的机器上生成结果视频：

```bash
python scripts/anti_uav/rknn_yolov8_native/render_tracker_video.py \
  --video /data/anti_uav/videos/test.mp4 \
  --tracks /data/anti_uav/output/smoke/tracks.csv \
  --benchmark /data/anti_uav/output/smoke/benchmark.json \
  --output /data/anti_uav/output/smoke/tracking_visualization.mp4
```

可视化属于离线工具，不计入实时 pipeline FPS。

## 13. 板端性能实测

当前 neg15 hard-positive e1 模型已在 Orange Pi CM5 RK3588S 实测。输入为 `1920x1080 @ 100 FPS` 的 `Video00004.mp4`，模型输入 `960x544`，queue size 为 3，CPU worker 绑定大核 4/5/6，NPU context 分别绑定 NPU0/1/2。三线程数据包含视频读取；Tracker 长测同时写入 tracks CSV。

| 配置 | 计时帧 | 完整 FPS | 平均 NPU 推理 | 说明 |
|---|---:|---:|---:|---|
| 3 contexts，detector，`conf=0.01` | 2359 | 114.75 | 19.83 ms | 336 TP / 78 FP |
| 3 contexts + RK-BoT-SORT | 2359 | 108.51 | 20.77 ms | high/low/new=`0.03/0.01/0.05` |

长测中的 Tracker association 平均只有 `0.011 ms/frame`；性能下降主要来自温度导致的 NPU/CPU 降频，不是 RK-BoT-SORT。板端风扇使用自动策略，当前散热结构下 107 FPS 需求只有约 1.5 FPS 的长测余量。生产交付前仍须用真实 107 FPS 摄像头连续运行至少 30 分钟，检查采集丢帧、端到端延迟和热稳态；建议改善散热器接触、风道或主动风扇，不能只用冷机峰值作为持续性能承诺。

原始结果位于本地交付目录 `deliverables/anti_uav_yolov8n_neg15_hard_positive_544x960_20260831/board/`。历史同结构模型的更高冷机 FPS 主要来自测试温度、视频解码和持续负载条件，不能替代当前模型的完整视频验收成绩。

## 14. 上层系统接口

`tracks.csv` 的字段为：

```text
frame,timestamp_sec,track_id,x,y,width,height,confidence,class_id,
predicted,confirmed,age,hits,time_since_update_sec
```

生产系统通常不应长期写 CSV，而应在 `ResultConsumer::consume()` 中将 `TrackOutput` 发布到消息队列、共享内存或业务 SDK。推荐上层消息至少包含：

```json
{
  "timestamp_sec": 12.345,
  "track_id": 7,
  "class_id": 0,
  "class_name": "drone",
  "bbox_xywh": [812.0, 366.0, 11.0, 8.0],
  "confidence": 0.61,
  "detector_backed": true
}
```

无线目标匹配应由上层系统基于时间同步、空间位置和无线设备 ID 完成。Detector/Tracker 只提供视觉目标和 Track ID，不直接判断船只或无人机是否合法。

## 15. 验收清单

1. 确认板型、kernel、NPU driver 和 RKNN Runtime 版本。
2. 确认模型输入为 `960x544`、INT8、9 outputs、单类别 drone。
3. 核对 Git commit `eec8b01`、RKNN SHA256 `3d5a47132cd7b087e03170ba398db7a57af730d834ebb0c8d0c77d81e776f98e` 和发布清单。
4. 重启后确认 CPU/NPU/DDR governor 仍为 performance。
5. 运行 20 帧 smoke test，确认模型可初始化且没有 output layout 错误。
6. 运行至少 1000 帧 benchmark，保存 JSON、温度和 NPU/DDR 频率。
7. 在代表性灰度视频上对比 `.pt`、ONNX 和 RKNN 输出。
8. 使用真实摄像头验证 107 FPS 采集、时间戳、延迟和丢帧情况。
9. 连续运行至少 30 分钟，检查温度降频、内存增长和视频断流恢复。
10. 检查 Track ID、短时漏检恢复、目标离开后轨迹删除和误报行为。
11. 确认生产配置使用 `queue-size 3`，不是离线 benchmark 的 `queue-size 12`。
12. 保存最终 benchmark、配置、SHA256 和部署日期，作为设备交付记录。

## 16. 常见问题

### `rknn_init failed`

优先检查 `.rknn` 是否由 RKNN-Toolkit2 2.3.2 为 `rk3588` 生成，以及板端 `librknnrt.so` 和 NPU driver 是否匹配。

### `Expected one input and nine RK-optimized YOLO outputs`

使用了普通 Ultralytics 单输出 ONNX/RKNN。必须通过 `export_detector_rkopt_onnx.py` 导出 RK-optimized 9-output 模型。

### `Native output is not INT8`

使用了 FP16 模型或转换配置错误。当前 C++ decoder 只接受 native INT8 outputs。

### 三核 FPS 没有明显提升

检查是否使用 `--workers 3`，而不是只设置 `--core-mask 0_1_2`；同时检查三个 NPU 核、CPU affinity、performance governor、视频解码和温度降频。

### 跟踪 ID 在多核模式下混乱

确认使用本仓库的 ordered consumer 版本。不能让三个 worker 分别运行 tracker，tracker 必须在检测结果按帧号重排后单线程按时间顺序执行。

### 小目标漏检较多

当前灰度目标在 `960x544` 输入下多数只有约 `4-6 px`。优先检查是否使用第 3 节灰度适配模型和正确 INT8 calibration；进一步提升需要 P2 检测头、`1280x736` 输入或原图切片，不能只通过 tracker 参数解决首次漏检。
