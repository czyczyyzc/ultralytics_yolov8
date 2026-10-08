# Motion-Aware Balanced：实现与实测对照

日期：2026-10-08。原版 Dist 保留，新增独立原生 C++ 跟踪器；不是完整 OC-SORT/IMM，也不再宣称与公开 Dist 代码行为等价。生产程序默认仍为 `--tracking dist`，新版需显式选择 `--tracking motion`。本次未重训或修改 detector，未增加 NPU tracking 模型。

## 000002：同一份检测结果，修复真实关联

输入为用户的 `000002.ts` 无重编码整理时间戳后的完整 1,800 帧；44 视频版 FP32 P3/P2+P3，960x544，detector conf=0.03、NMS IoU=0.45。新旧跟踪使用完全相同的检测框与 GMC 变换。只输出当帧检测支持的已确认轨迹，不显示预测/未确认框，也不重编号 ID。

| 模型 | 全片检测框 | 原 Dist 输出框 | 新版输出框 | 原版有跟踪帧 | 新版有跟踪帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P3 | 547 | 445 | 475 | 442 | 467 |
| P2+P3 | 666 | 513 | 536 | 447 | 473 |

框数/有输出帧数不是 Recall；该视频没有用于本次评测的完整身份 GT。更多框不能直接证明精度更高。

| 13-14 秒，30 帧 | 原 Dist 有跟踪帧 | 新版有跟踪帧 | 原版主目标 ID | 新版主目标 ID |
| --- | ---: | ---: | --- | --- |
| P3 | 23/30 | 30/30 | 1 断开，随后变为 15 | 始终为 1 |
| P2+P3 | 23/30 | 30/30 | 1 断开，随后变为 29 | 始终为 1 |

更长的 13.000-14.333 秒、帧 390-430 也逐帧检查通过：两种模型的主目标检测均被关联为 ID 1。这里的主目标采用人工查看过的连续无人机片段及对应当前检测索引检查，不是正式 MOT 身份指标，也没有把检测框当作训练 GT。

新增确认规则采用 3-of-4，并要求匹配观测的平均置信度达到原有 birth=0.10。新 ID 最早在第 3 次有效观测输出，所以启动时的两个观测可能无确认 ID；这不等于 detector 没有输出。内部预测用于关联，最终框始终来自当帧检测。

## Video00009：跨视频验证与取舍

此项固定的是旧 28 视频 P2+P3 FP32 的 14,201 帧检测缓存，不是 44 视频 detector 的新精度结果。审核有效帧 14,199、GT 框 7,982，匹配 IoU=0.5，detector conf=0.03；新旧跟踪共享检测与 GMC。这是开发期间使用的灰度验证回归，不是独立测试。

| 方法 | Recall | Precision | TP | FP | FN | F1 | 连续可见 ID 跳变诊断 | 相邻 TP 帧 ID 跳变诊断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 Dist | 66.22% | 63.89% | 5286 | 2987 | 2696 | 65.04% | 6 | 2 |
| Motion-Aware Balanced | 66.20% | 63.18% | 5284 | 3080 | 2698 | 64.65% | 2 | 0 |

**身份连续性改善，但不是所有指标都提升。** FP 增加 93（约 3.11%），Precision 和 F1 略降，Recall 基本持平。因此新版仍作为显式启用的实验版，不自动覆盖原生产部署；还需要其他未参与开发的视频及板端验收。

上述 ID 数值是“单 GT 区间内、IoU>=0.5 的目标匹配观测 ID 变化”诊断，不是正式 MOTChallenge IDSW、IDF1 或 HOTA；不能仅凭 ID 数下降排除错误合并。初版曾出现 ID 诊断明显退化，该试验也保留在服务器，未作为推荐版本。最终版修正了大协方差轨迹的评分偏好、大目标测量抖动和确认置信度证据。

## 交付与验证

代码及参数说明：`scripts/anti_uav/dist_native/MOTION_AWARE.md`。核心是 `motion_tracker.cpp`/`motion_tracker.hpp`；GMC 增加质量接口，原生 `video.cpp` 可直接调用，不依赖 Python。Python 桥和缓存重放仅用于离线验证。

本地完整视频：

```text
/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/motion_aware_tracker_20261008/000002_44video_p3_motion_aware_balanced_GMC_conf003.mp4
/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/motion_aware_tracker_20261008/000002_44video_p2p3_motion_aware_balanced_GMC_conf003.mp4
```

47 服务器完整结果和最终 x86_64 检验库：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/motion_aware_tracker_20261008/p3_final/
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/motion_aware_tracker_20261008/p2p3_final/
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/motion_aware_tracker_20261008/Video00009_final/
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/motion_aware_tracker_20261008/native/final/libmotion_tracker.so
```

本地 `metadata/FINAL_COMPARISON.json` 保存完整数字、模型/输入/库 SHA、配置和验证范围。服务器各目录另存每帧检测、关联 ID、GMC 质量、变换和原始试验。

23 项测试在本地和服务器通过，涵盖零 IoU 小目标、运动突变、短时漏检、超预算拒绝、单帧误检、歧义拒绝、远处干扰、大目标和时间戳。原 Dist 在每种模型的 1,800 帧上精确回放通过；新增 C++ GMC 质量接口完成运行检查，Linux 原生流水线/GMC 完成 C++17 语法与类型编译检查，不等于板端完整链接及 RKNN 流水线实测。

**尚未完成板端 FPS、首帧/首次确认 ID 延迟、实时摄像头或 RKNN INT8 精度验收。** 服务器离线缓存重放耗时不能当作 RK3588S FPS。上板必须重新编译 arm64 库，并使用适合芯片的 RKNN/运行库，不可复制服务器的 x86_64 `.so`。
