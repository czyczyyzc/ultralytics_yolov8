# P3 局部放大与 Frozen-P3 + Add-on P2 对比实验

日期：2026-09-07。实验使用最新 manual_clips0123 模型，不重新训练。

## 结论

在 Video00004 上，基于上一帧跟踪结果的 ROI 放大明显改善了 P3。
4 倍 ROI 的 Precision、F1 优于本次低阈值全图 Add-on P2，但 Add-on P2 的召回率、首次发现速度和跟踪连续性更好。
不能据此称 ROI 在所有指标上替代 P2，也不能将单视频结果当成新场景泛化结论。

## 同条件设置

- 四组均输入 `960x544 (W x H)`，检测 `conf=0.01`、NMS IoU `0.45`、`max_det=100`、固定 letterbox、padding=114。
- 四组均调用实际 `detector_based_tracker.hpp` 的原生 C++ RK-BoT-SORT，无姿态输入；不是用 Python tracker 替代。
- Tracker high/low/new 为 `0.03/0.01/0.10`，两阶段匹配阈值均 `0.92`，buffer=1 秒、prediction grace=0、min_hits=3。只统计已确认且当前帧有检测支持的输出。
- 使用同一原始视频顺序推理，全部 2359 帧，100 FPS、1920x1080；448 帧有目标，1911 帧无目标。IoU >= 0.50 计 TP；重复输出和无目标帧输出均计 FP。
- P3 三组使用完全相同的权重，即此次 Add-on P2 的 P3 父模型。Video00004 被排除在此次训练数据之外。该视频已多次人工查看，因此是保留测试视频，不是全新盲测。
- 本次为 47 服务器 A100、PyTorch FP32、batch=1 参考实验。尚未验证 ROI 的 RKNN INT8 精度或板端 FPS。

## ROI 策略

原图为 1920x1080。2 倍 ROI 裁剪 960x540，4 倍 ROI 裁剪 480x270，再缩放/补边至模型输入 960x544。
这里的倍数相对于全图缩小至模型输入后的目标大小。例如原先输入尺度下 4-6 px 的目标，4 倍 ROI 后约为 16-24 px；插值不会创造原图中不存在的细节。

1. 第一帧及上一帧没有已确认、非预测轨迹时，全图检测。
2. 有有效轨迹时，以它上一帧框的中心裁剪。边缘处平移 ROI 保持尺寸，不读取当前帧 GT 或未来帧。
3. 保持此前选择的有效轨迹 ID；否则选择 hits、score 更高的轨迹。每帧只运行一个 ROI，不覆盖所有多目标。
4. 每 10 帧强制一次全图搜索；该帧不额外跑 ROI，始终一次检测推理/帧。
5. 检测框映射回原图坐标，再送入相同的 Tracker。评估阶段才读取 GT。

2 倍作为初始策略、4 倍作为敏感性对照，均在查看本次指标前固定，没有在此测试集上搜索最优窗口或训练权重。
已独立重建四组全部帧的 ROI、工作模式和 anchor ID，与保存的预测记录完全一致。

## 跟踪输出结果

| 方法，均加 RK-BoT-SORT | TP | FP | FN | Precision | Recall | F1 | ID 切换 | 断续次数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 全图 | 348 | 323 | 100 | 51.86% | 77.68% | 62.20% | 4 | 40 |
| P3 + 2 倍 ROI | 378 | 302 | 70 | 55.59% | 84.38% | 67.02% | 4 | 33 |
| P3 + 4 倍 ROI | 412 | 203 | 36 | 66.99% | 91.96% | 77.52% | 2 | 14 |
| Frozen-P3 + Add-on P2 全图 | 444 | 459 | 4 | 49.17% | 99.11% | 65.73% | 0 | 1 |

ID 切换指同一可见片段中相邻两次成功匹配输出的 ID 改变；断续指首次成功匹配后出现漏帧，再次恢复匹配。这里不是多目标标准 MOT 汇总指标。

## 小目标和原始 Detector

统一按全图输入尺度的 `sqrt(GT_width * GT_height)` 分桶；不能按 ROI 放大后的大小改变分桶，否则对比不公平。

| 方法 | 4-6 px 跟踪召回率，382 帧 | 4-6 px 检测召回率 | 全部目标检测 Precision | 全部目标检测 Recall |
|---|---:|---:|---:|---:|
| P3 全图 | 74.87% | 79.84% | 37.72% | 81.92% |
| P3 + 2 倍 ROI | 82.20% | 85.60% | 39.61% | 87.28% |
| P3 + 4 倍 ROI | 91.10% | 92.41% | 49.94% | 93.08% |
| Frozen-P3 + Add-on P2 全图 | 98.95% | 100.00% | 35.47% | 100.00% |

检测列统计送入 Tracker 前的检测框，仍来自各自在线闭环的顺序轨迹，conf=0.01。完整阈值表见实验目录 `RESULTS.md`。
该阈值表是对 conf=0.01 已生成的固定检测轨迹过滤，不是改变反馈阈值后重跑 ROI，也不是标准 mAP。

## 漏帧分析与边界

4 倍 ROI 的 36 个跟踪漏帧分解如下：1 帧 GT 未完全在 ROI 内；12 帧 GT 在 ROI 内但 detector 未匹配成功；18 帧处于全图模式且 detector 未匹配成功；5 帧全图检测成功但未输出成功匹配的确认轨迹。
“未匹配成功”也包括有预测但定位 IoU 不到 0.50，不一定完全没有框。ROI 外的一帧不能直接归因为运动过快，也可能是错误 anchor。

因此“只要目标不出框就能检出”并不成立。ROI 有助于放大已有目标，但全图搜索尚未找到目标时没有收益；错误轨迹也可能吸引后续裁剪。
本视频只有一个连续可见片段：零起始帧号 587-1034。三组 P3 首次确认匹配均延迟 7 帧，即原视频时间 70 ms；Add-on P2 为 2 帧、20 ms。这不是计算延迟，也不足以检验多次消失后重现的场景。

本次优先降低 FP、提升 F1 时，4 倍 ROI 值得继续；优先小目标召回与连续跟踪时，Add-on P2 仍更强。两者结论均限定在当前统一阈值下。
ROI 模型输入大小不变，不会自动降低单次 NPU 计算量；上一帧反馈还会影响多核并行调度，不能沿用此前约 85 FPS 的结果。

## 模型与数据追溯

服务器训练根目录：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_frozen_p3_addon_p2_manual_clips0123_20260904
```

相对于上述目录的模型路径与 SHA256：

```text
P3:
training_p3/real_gray_yolo_strict_holdout_Video00004_manual_clips0123_neg5fps_v1_20260904/weights/best.pt
e101428d2018a4cb9e25c5e16a5625a3f6f526c153a2ab5bd662b55f4eab7456

Frozen-P3 + Add-on P2:
training_addon/final/weights/best.pt
3ab1a39ad1113768f1ad51ab7343e155c91d088245f897084b0e8245ead51f65
```

视频及评估标注：

```text
/mnt/andrew/anti_uav_model_refinement/external_eval/real_gray_mp4/Video00004.mp4
/mnt/andrew/anti_uav_model_refinement/external_eval/real_gray_tracker_eval_v2/annotations/Video00004.visible.json
```

训练数据沿用此次模型的 10 个灰度训练视频和 Anti-UAV300 RGB 混合数据，不因本次 ROI 实验增减。权重、视频和 Tracker 源码哈希均在每组 `manifest.json` 中记录。

## 结果文件与复现

服务器结果：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/causal_p3_roi_vs_addon_p2_20260907
```

本地结果：

```text
/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/causal_p3_roi_vs_addon_p2_20260907
```

四宫格视频相对于结果目录：

```text
visualization/Video00004_P3_ROI2x_ROI4x_AddonP2_RKBoTSORT.mp4
```

视频左上 P3、右上 ROI 2 倍、左下 ROI 4 倍、右下 Add-on P2，均包含 Tracker。无 GT，无十字，1 px 四角预测框；放大视图从干净原图裁剪后再画框。白色大四角框表示实际推理 ROI。视频保持完整 100 FPS、2359 帧；视频播放帧率不是计算吞吐率。

每组包含 `frames.jsonl`、`manifest.json`、编译的 `librk_tracker.so`；总表在 `evaluation/summary.json`，逐帧指标在 `evaluation/per_frame.csv`，详细英文报告在 `RESULTS.md`。

在 47 服务器仓库根目录运行如下命令可重新生成。输出目录必须是新目录，避免覆盖本次结果：

```bash
EXPERIMENT_ROOT=/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/causal_roi_repeat \
bash scripts/anti_uav/run_causal_roi_experiment.sh
```

主脚本依次运行四组顺序推理、GT 事后评分及可视化。源码位置为仓库内 `scripts/anti_uav/{causal_roi_policy,run_causal_roi_comparison,summarize_causal_roi_comparison,render_causal_roi_comparison}.py`，C++ 桥接为 `scripts/anti_uav/rknn_yolov8_native/tracker_c_api.cpp`。

已通过 7 项单元测试，包括 ROI 越界、搜索回退、周期刷新、坐标映射、FP 计数、跟踪事件和非法裁剪审计；四组完整实跑和 C++ 桥接合成轨迹测试均完成。
