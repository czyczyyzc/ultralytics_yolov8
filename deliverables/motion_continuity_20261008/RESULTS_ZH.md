# 连续目标 ID 与检测框保留回归结果

日期：2026-10-08。检测输入固定为 `960x544`、`conf=0.03`、NMS IoU `0.45`。
本轮仅改原生 C++ 跟踪关联和输出协议，没有重新训练 detector，没有修改检测框或置信度。
服务器回归使用之前保存的 FP32 检测结果和完全相同的 GMC 矩阵；不是 RKNN INT8 板端速度测试。

## 结果

| 视频 / detector | 检测框总数 | 原 Dist 输出框 | 新版已确认轨迹框 | 新版保留的观测框 | 新版未保留的检测框 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `000002`，44 视频 P3 | 547 | 445 | 485 | 547 | 0 |
| `000002`，44 视频 P2+P3 | 666 | 513 | 545 | 666 | 0 |
| `Video00009`，原 28 视频 P2+P3 固定检测缓存 | 9109 | 8274 | 8382 | 9109 | 0 |

新增观测输出包含已确认、未关联、待确认、有歧义等状态，不把它们全部算作已确认跟踪。
例如 `Video00009` 的 9109 个框里，8382 个有已确认 ID，727 个没有已确认 ID。
这些 727 个框仍完整保留，显示 `PENDING` 或 `UNCERTAIN`，不伪造身份。
两帧不确定标注不参与 GT 指标，所以下表的已确认输出总数比全视频少 1。

### 连续性

| `Video00009` 单目标诊断 | 原 Dist | 上一版 Motion-Aware | 本版 |
| --- | ---: | ---: | ---: |
| 连续 GT 可见区间 ID 变化 | 6 | 2 | 0 |
| 相邻 TP 帧 ID 变化 | 2 | 0 | 0 |

本版三次 GT 匹配 ID 变化均在 GT 不可见间隔后发生，不属于连续可见区间。
没有通过 ID 重映射、强制复用旧 ID 或画预测框消除跳变。
GT 是单目标检测框标注，以上是诊断，不是正式 MOT IDSW / IDF1 / HOTA。
仅能说明当前完整视频回归通过，不能保证任意场景零跳变。

`000002` 的 13-14 秒，P3 和 P2+P3 均为 30/30 帧已确认跟踪，主目标始终为 ID 1。
更长的 13-14.333 秒检查区间（索引 390-430）同样每帧以 ID 1 关联当前主目标检测。

### 精度边界

下表仅统计已确认轨迹，不将无身份观测算成跟踪 TP。

| `Video00009` | Recall | Precision | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 Dist | 66.22% | 63.89% | 5286 | 2987 | 2696 |
| 本版 Motion-Aware | 66.20% | 63.05% | 5284 | 3097 | 2698 |

连续性显著改善，但已确认 FP 增加 110，Recall 没有提升。
有 GT 匹配检测但尚无已确认轨迹的审核帧数为 18，原 Dist 为 16；这类观测已保留，但身份尚未确认。
“没有吞掉检测框”与“每个框都有可靠 ID”是不同要求，不能混淆。
保留所有检测也会保留 detector 的误检；保留框本身不等于提高检测精度。
这些视频用于调试，结果不构成独立泛化验证，因此没有自动替换板端生产默认算法。

## 修改机制

- 相机运动不确定性随近期可信 GMC 位移自适应，并计入观测历史的速度估计误差，解决急转相机时 GMC 失败、恢复造成的断轨。
- 已确认轨迹先关联，再处理临时轨迹；仍保留运动、形状和歧义门限，不无条件认旧 ID。
- 大目标采用最近真实观测的 GMC 对齐框参与几何关联，避免预测框尺寸滞后以及边缘截断引起的错误关联。
- 所有当前 detector 观测由原生接口输出，身份不确定时 `id=null`。无检测的帧不画预测框，不进行事后 ID 修补。

原生算法是基于 Dist 矩阵/LAP 组件实现的独立 Motion-Aware 版本，不宣称与公开 Dist、完整 OC-SORT 或 IMM 等价。

## 本地文件

目录：`/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/motion_continuity_20261008/`

- `000002_P3_observations_stableID_conf003.mp4`：完整 1800 帧、60 秒、30 FPS。
- `000002_P2P3_observations_stableID_conf003.mp4`：同一视频、同一参数、完整 1800 帧。
- `Video00009_observations_stableID_conf003.mp4`：完整 14201 帧、142.01 秒、100 FPS；使用旧 28 视频 detector 缓存，便于与原跟踪器严格对比。
- `metadata/*_audit.json`：检测框保留、已确认指标、身份诊断和哈希。
- `metadata/p3/`、`metadata/p2p3/`、`metadata/Video00009/`：逐帧 `tracks.jsonl` 和回归协议。
- `metadata/*_visual_summary.json`：视频源、权重、输出 SHA256、帧数与编码参数。

视频均使用 1 px 四角框，无 GT、无十字；先裁剪未标注原图再放大、画框。
青色框是已确认 ID；黄色框是未确认/未关联观测。播放 FPS 是原视频时间轴，不是 inference FPS。

## 47 服务器

代码：`/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/dist_native/`

回归目录：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/motion_continuity_20261008/`

最终缓存：`p3_geometry_trial/`、`p2p3_geometry_trial/`、`Video00009_geometry_trial/`。
视频：对应 `p3_visual/`、`p2p3_visual/`、`Video00009_visual/`。
早期 trial 目录保留用于审计，不作为最终结果。
此次原生库：`native/geometry/libmotion_tracker.so`，为服务器 x86_64 库，不能复制到 ARM 板子直接使用。

44 视频 P3 权重：
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_p3/p3/weights/best.pt`

44 视频 P2+P3 权重：
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved44_direct_online328_20260929/training_addon/p2/weights/best.pt`

`Video00009` 的固定 detector 权重 SHA256：
`61e9a3669f964e59e96e9b2a24bf7705b945e184ed5a0628be33ee44d8b4bb59`。
这是原 28 视频模型，不可将该回归称为最新 44 视频模型精度。

## 接入注意

在板子新目录重新编译 tracker、GMC 与 native executable，显式选择 `--tracking motion`。
仅替换 `.so`、仍走 `--tracking dist` 不会启用新算法。
模型仍应使用对应芯片平台的 RKNN；本轮不替换 detector 权重。

平台/可视化应消费新 `observations` 字段；继续只读 `displayed_tracks` 就仍只得到已确认框。
原生 `motion_observations()` 返回全部 detector 索引及状态，详见：
`/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/scripts/anti_uav/dist_native/MOTION_AWARE.md`

新目标默认经过 3/4 次观测和累计置信度验证后才显示稳定 ID；确认前保留框，不显示临时 ID。
遮挡、目标交叉、极弱检测、长时间出画等身份不可辨场景允许返回不确定状态，不强行维持身份。

本地与服务器测试均为 31 passed，服务器另有 4 subtests passed。
Linux native pipeline / GMC 的 C++17 语法与类型编译通过。
尚未执行此次修改后的 RK3588 摄像头完整管线测试、FPS 或延迟复测。
