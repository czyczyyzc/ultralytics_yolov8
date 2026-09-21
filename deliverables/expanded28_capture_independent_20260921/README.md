# 独立摄像头采集与最新帧交接测试

日期：2026-09-21。板卡：同一 RK3588S CM5，厂商 5.10.198，RKNN Runtime 2.3.2。

## 实测结果与选择

**独立采集已经实现，但纯 P3 的实测没有延迟收益，保留 direct 为默认。** 不将结构并行化等同于性能提升。

同一新程序、同一 P3 RKNN，各处理 12,000 帧：

| 指标 | direct 原模式 | independent 独立采集 |
| --- | ---: | ---: |
| 完整 FPS | 120.00 | 120.00 |
| 驱动时间戳至完整结果，平均 | 29.91 ms | 30.39 ms |
| 驱动时间戳至完整结果，P95 | 31.89 ms | 34.85 ms |
| 驱动时间戳至取出帧，平均 | 7.392 ms | 7.400 ms |
| 取帧准备好至交给 worker，平均 | 0.0004 ms | 0.262 ms |
| NPU 推理，平均 | 21.19 ms | 21.38 ms |
| 开始 read 至完整结果，平均 | 24.93 ms | 30.80 ms |
| 源帧序号跳过 | 0 | 0 |

原模式在纯 P3 下已经跟得上 120 FPS，独立模式没有降低驱动至取帧时间，增加了约 0.26 ms 的准备好至交接等待；NPU 均值也有约 0.18 ms 差异。平均总延迟增加约 0.48 ms，不能声称优化成功。阶段统计能说明开销位置，但不是对所有调度竞争原因的独立因果证明。

独立长测完成 12,001 次读取、发布/领取 12,000 帧、槽覆盖 0 次、结束多读 1 帧。两组 GMC 都更新 12,000 次，输出序号严格递增。

2,000 帧短测与反向复测同样未显示稳定收益：

| 运行 | FPS | 驱动至结果平均 / P95 |
| --- | ---: | ---: |
| direct 前测 | 119.68 | 30.79 / 36.18 ms |
| independent 前测 | 120.00 | 31.26 / 36.78 ms |
| independent 后测 | 120.02 | 31.16 / 36.44 ms |
| direct 后测 | 120.01 | 30.24 / 33.69 ms |

每次测试的摄像头/NPU 相位和调度会有变化；不能用一次短测的小差异作普遍性能承诺。独立采集保留为明确可选功能，原交付部署和模型未覆盖。

P3+P2 补充短测各 2,000 帧，不作为其长测结论：

| 指标 | direct | independent |
| --- | ---: | ---: |
| FPS | 85.51 | 88.09 |
| 驱动至结果平均 / P95 | 44.08 / 50.36 ms | 43.99 / 49.74 ms |
| 驱动至取帧平均 | 10.57 ms | 7.41 ms |
| 图像准备好至分发平均 | 0.0004 ms | 2.87 ms |
| 源帧跳过比例 | 28.7% | 26.5% |

独立模式实际完成 2,723 次读取，发布 2,722 帧、领取 2,000 帧、覆盖 722 帧，正常结束多读 1 帧。确实实现了“采集不等待 NPU”，但大部分减少的驱动等待转移到了应用最新帧槽，平均完整延迟仅差约 0.09 ms。该短测的吞吐改善尚未经过相同长度的长期复测，不能承诺普遍提升。

## 实现与使用

新增 `--camera-dispatch independent`，保留原 `--camera-dispatch direct` 默认行为。

- direct：等待在途名额和空闲 NPU worker，再从 V4L2 取帧。
- independent：独立线程持续读取相机，不等待 NPU；将独立持有的图像放入一个最新帧槽。新帧只能替换尚未领取的槽内旧帧，不能覆盖已送入检测或 GMC 的图像。
- 调度线程同时满足 worker 空闲、在途帧少于上限、最新帧槽非空时，领取并提交该帧；不另建 FIFO。
- GMC 只处理被领取的同一组选中帧，Dist 仍按采集顺序关联，不按 NPU 完成顺序关联。未更改检测或跟踪阈值，也未修改 Kalman 时间模型。
- 退出时等待已启动线程结束后再释放摄像头、模型和动态库。正常结束可能多读一张帧，单独记为 shutdown_discarded，不算业务处理帧或中间漏帧。

独立模式仅支持 V4L2 + latest 策略，显式拒绝 fifo/fresh/非摄像头组合。本次仍执行灰度图像复制，不宣称零拷贝。采集等待槽容量为 1；在途帧上限仍为 3。

板端新增独立测试部署，没有覆盖原交付程序：

```text
/home/orangepi/deployments/expanded28_capture_independent_20260921
```

它的模型软链接指向此前纯 P3 部署，deps 指向原摄像头依赖目录；bin 是本次重新编译的实际目录。仍需保留被链接的目标目录及旧 RKNN Runtime。

```bash
cd /home/orangepi/ultralytics_yolov8_ziye_code
export ANTI_UAV_CAMERA_DEPLOY=/home/orangepi/deployments/expanded28_capture_independent_20260921
bash scripts/anti_uav/dist_native/run_camera_p3_on_board.sh \
  --camera-dispatch independent --frames 12000 \
  --output "$ANTI_UAV_CAMERA_DEPLOY/new_independent_$(date +%Y%m%d_%H%M%S)"
```

同程序对照只改为 `--camera-dispatch direct`，并使用不同输出目录。
移除环境变量后，原 P3 启动脚本仍指向原部署；新参数需要本次新程序，旧程序不认识该选项。

## 对比协议

两种模式使用同一新程序、同一组三库、同一模型；960×544 INT8，conf=0.03，NMS=0.45，3 个独立 RKNN context（0/1/2），inflight=3，4 个 V4L2 缓冲，latest 策略，overlap 启动。CPU 集合和频率策略不变，未关闭温控。每个结果含 C++ 检测、GMC、Dist。

P3 主对比各 12,000 输出帧，另有前后 2,000 帧短测；P3+P2 为独立的 2,000 帧配置对照。所有性能运行排除最初 100 帧统计，但仍处理这些帧；只有 120 帧功能检查开启 observations。

原始 `summary.json`、`latency.csv` 和灰度首帧分别保存在 `p3/` 与 `p2/` 下。`p3/comparison.json`、`p2/comparison.json` 由审计工具生成，不混合两个模型的哈希。实时顺序采集并非逐像素相同的帧回放。

## 时间与丢帧口径

首要指标是 `driver_to_output`：该帧驱动 MONOTONIC 时间戳至完成检测/GMC/Dist 的时间，而不是函数调用开始时间。

独立线程会在上一帧读取完成后立即发起下一次 read，可能等一整个相机周期。因此 `read_ms` 和 `read_to_output_ms` 可能增加，即使该图像的实际年龄没有增加；不能将这一差异误判为 NPU 变慢或相机曝光变长。

本次 CSV 增加 `ready_ms` 和 `dispatch_ms`，可以分别检查：

1. `dequeue_ms - frame_ms`：驱动时间戳至应用取出帧。
2. `ready_ms - dequeue_ms`：复制及完成取帧。
3. `dispatch_ms - ready_ms`：最新帧槽等待与调度交接。
4. `output_ms - dispatch_ms`：worker 等待、检测、并发 GMC 与顺序关联。

不能只看第 1 项下降就宣称总延迟下降；等待可能从驱动缓冲转移到了应用帧槽。

`capture_scheduler` 记录 read_completed、slot_published、slot_replaced、slot_taken、slot_remaining、shutdown_discarded。对正常独立采集运行应满足：

```text
slot_taken = 输出帧数
slot_published = slot_taken + slot_replaced + slot_remaining
read_completed = slot_published + shutdown_discarded
```

`camera.sequence_gaps` 是实际输出序列中跳过的相机序号，已包含其中的槽覆盖与驱动遗漏，不能再与 slot_replaced 相加。退出多读的尾帧在首末输出区间之外，不计入该跳帧比例。

## 验证与适用范围

- 最新帧槽测试：覆盖未领取帧、已领取帧保持不变、10 万次并发发布的顺序及计数一致性；Mac ASan/UBSan 和 TSan 分别通过，板端原生测试通过。这不是对闭源 RKNN 驱动的线程检测。
- 板端灰度/RGB 预处理测试：132 RGB 布局、132 灰度布局、10 个非法输入通过。
- 参数检查：非法模式及与 fifo/fresh/非 V4L2 的四个组合均被拒绝。
- 120 帧独立采集功能检查：观察记录顺序、GMC 更新帧数及当前检测索引关系通过。
- 审计器验证 CSV 帧数、相机序号、时间戳顺序、取帧/交接/输出顺序、帧槽计数和汇总延迟一致性；亦兼容之前的 37 组记录。

仍是室内、倒置、部分遮挡的相机画面，无标注 UAV 轨迹；结果不能证明无人机跟踪精度或多目标重载表现。驱动时间戳不是已验证的曝光时间，不含显示、编码和传输。

P3 direct / independent 长测分别记录 28 / 0 个检测框、21 / 0 个显示跟踪输出，因此本次不是活跃目标上的逐帧输出等价性或精度测试。两次均约运行 100 秒，也不是无限期热稳定性认证。

源码提交：`ddbf0d8c75571a9c419948a871808ace829f1695`。
程序 SHA256：`e404f1d2d6b0ba382215792937d552d9341d8f51e5b5aa666170fbee5d04ca3a`。
三库源码未改；本次重新编译时依赖目录 RPATH 不同，部分库二进制哈希与旧部署不同。A/B 两组使用同一批新库，不混用旧库。

审计命令（在本仓库根目录，有 Python 的工作机上）：

```bash
python3 scripts/anti_uav/summarize_camera_benchmark.py deliverables/expanded28_capture_independent_20260921/p3
python3 scripts/anti_uav/summarize_camera_benchmark.py deliverables/expanded28_capture_independent_20260921/p2
```

47 服务器结果目录：`/mnt/chenziye/codes/ultralytics_yolov8/deliverables/expanded28_capture_independent_20260921`。
