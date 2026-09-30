# RK3588S 960x544 INT8 Detector Exports

Exported on server 47 on 2026-09-30 with `rknn-toolkit2==2.3.2`, target `rk3588`, from the `best.pt` checkpoints listed below. `40` is the recommended next candidate for on-board validation; `44` is an experimental newer training run whose FP32 holdout false positives increased substantially. These files are **converted artifacts, not approved board deployments**.

| Variant | Source checkpoint on server 47 | ONNX outputs | RKNN SHA256 |
| --- | --- | ---: | --- |
| `40_p3` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online328_20260922/training_p3/p3/weights/best.pt` | 9 | `b8d4517946b745d0bdc60cde45c0955b58111e31442521be96ab38e03f378fe0` |
| `40_p2p3` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved40_online328_20260922/training_addon/p2/weights/best.pt` | 12 | `5851dd1d85f511a35500461680142a01189d4f5a3043aeeed9a2e8bf3f63486c` |
| `44_p3` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_p3/p3/weights/best.pt` | 9 | `011373d915c7b05c8cf793be21da87ffdb7bdda840bcbb7fdc3233eaa20d200c` |
| `44_p2p3` | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_addon/p2/weights/best.pt` | 12 | `95c4cf02d73ec9b6056f96545c2d3bd4aa195cb084607711f3840912956133e5` |

Each local variant directory contains `detector_960x544_int8.rknn` and `detector.rkopt.json` (the checked ONNX output layout). Identical RKNN files and the ONNX/build logs are on server 47 at:

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved_rknn_exports_20260930/<variant>/
```

The 384-image calibration list is:

```text
/mnt/chenziye/codes/ultralytics_yolov8/deliverables/anti_uav_rk3588s_frozen_p3_addon_p2_final_20260904/metadata/calibration/dataset_no_Video00004.txt
```

The RKNN compiler completed for all four models, and Toolkit2 `load_rknn()` returned success for each file. This does **not** demonstrate board runtime compatibility or INT8 detection accuracy. The compiler reported weight outliers that may affect quantization accuracy. Host-side `init_runtime()` on an exported `.rknn` is unsupported by this Toolkit2 installation; the connected RK3588S was unavailable at export time. Before production use, run C++ RKNN inference on the board, compare INT8 predictions against the FP32 checkpoints on Video00004 and Video00009, then measure sustained FPS/latency and tracker ID behavior. Do not reuse the 28-video model's measured FPS or the FP32 precision/recall for these files.

The `.rknn` binaries are intentionally not committed to Git. The local copies remain beside this README; server copies are at the paths above. Verify SHA256 after any transfer. The full deployment handoff is `../anti_uav_p3_p2p3_dist_gmc_handover_20260929/DEPLOYMENT_HANDOVER_ZH.md`.
