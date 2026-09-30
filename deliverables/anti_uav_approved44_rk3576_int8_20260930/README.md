# 44 视频检测器：RK3576 INT8 交接

2026-09-30 在 47 服务器使用 `rknn-toolkit2==2.3.2`，由 44 视频版的 ONNX 以 `target=rk3576` 重新量化编译。输入宽 960、高 544。**不能将同权重的 `rk3588` 模型直接用于 RK3576**；那会在 `rknn_init` 阶段报平台不匹配（例如 `-6`）。

| 模型 | 本地文件 | SHA256 | 输出数 |
| --- | --- | --- | ---: |
| 纯 P3/P4/P5 | `44_p3/detector_960x544_rk3576_int8.rknn` | `12b98639fe20bf2015c26faac97f34adf6028b3672fc935a0ba436fa316abc76` | 9 |
| Frozen-P3 + Add-on P2（P2/P3/P4/P5） | `44_p2p3/detector_960x544_rk3576_int8.rknn` | `974a58f820692feac30fcde67d2f1ef068c6d8718949f673570611ebb9e6a642` | 12 |

服务器 47 的文件在：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved_rknn_exports_rk3576_20260930/44_p3/detector_960x544_rk3576_int8.rknn
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved_rknn_exports_rk3576_20260930/44_p2p3/detector_960x544_rk3576_int8.rknn
```

源权重：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_p3/p3/weights/best.pt
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_addon/p2/weights/best.pt
```

四尺度/三尺度的目标无关 ONNX 来自 `runs/anti_uav/approved_rknn_exports_20260930/44_p2p3` 与 `44_p3`，沿用不含 Video00004/Video00009 的 384 图校准清单。编译日志在各服务器模型目录的 `build.log`。二者编译成功、模型内部 `target_platform` 均为 `rk3576`，Toolkit2 `load_rknn()` 均返回 0；本地与服务器 SHA256 相同。

**未完成的验收**：当前板端地址未连通，尚未在 RK3576 上执行 `rknn_init`/推理，也没有这两份 INT8 的精度或 FPS 实测。编译器报告量化离群值警告。44 视频权重此前在 FP32 的 Video00004 测试中误检上升，不应仅因模型现在可加载就替换生产模型。部署前请用相同灰度测试集比较 FP32/INT8 的 Recall、FP 和小目标召回，并跑完整 Dist/GMC 流水线。

板端切换时必须同时检查 NPU 并行参数：此前 RK3576 的原生视频程序使用两个 worker 与 `--core-mask 0_1`；当前 `dist_native` 若移植到 RK3576，应改为 `--workers 2 --inflight 2 --npu-masks 0,1`，而非照搬 RK3588S 的三个 worker/`0,1,2`。两种程序的参数名不同；这份导出**尚未验证 Dist/GMC 在 RK3576 上的运行**。`librknnrt`/NPU 驱动版本也应与 Toolkit2 2.3.2 生成的模型相容。

模型二进制保留在本地本目录和服务器上述路径，**不提交到 Git**；交接前按表中 SHA256 校验。
