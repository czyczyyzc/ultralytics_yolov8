# 000002：GMC 与首帧 ID 的重新对照

日期：2026-10-08。
使用最新 44 视频 P3 与 Frozen-P3 + Add-on P2 的固定 FP32 检测缓存，输入 `960x544`，`conf=0.03`，NMS IoU `0.45`。
完整视频 1800 帧、60 秒、30 FPS；P3 为 547 个检测框，P2+P3 为 666 个。
本次重新运行原生 C++ Motion-Aware 关联，保持 detector 框、分数和权重不变，不重复 YOLO inference。
没有使用 GT、ID 重映射或预测框填补，也没有测试 RKNN 板端 FPS。

## 实验定义

每个 detector 进行四组主实验，并补两组关闭 GMC、保留默认不确定性的控制，共 12 组：

- 常规确认 + GMC：新轨迹阈值 `0.10`，4 帧内 3 次匹配，平均匹配置信度至少 `0.10`。
- 常规确认，不用 GMC：确认规则相同，在图像坐标中学习相机和目标合成后的运动。
- 首帧 ID + GMC：新轨迹阈值降至 `0.03`，一次观测即可获得 ID。
- 首帧 ID，不用 GMC：首帧规则相同，但不进行相机运动补偿。

所有组均保留完整 detector 观测。首帧 ID 只改变生命周期策略，不代表身份已经有足够证据；仍拒绝强行指定有歧义的身份。
无 GMC 不是“每帧 GMC 失败”：不计算补偿，并将 GMC 专用运动不确定性项设为 0，让滤波器学习完整画面位移。
因此无 GMC 的实际参数 `unknown_gmc_speed_px_s=0`；有 GMC 为默认 `1500`。
两组有 GMC 实验使用完全相同的补偿矩阵和质量分数。
新常规确认 + GMC 的逐帧观测、身份、已确认框与上一版已验证结果完全一致。

补充控制：`--gmc-mode unavailable`，不计算 GMC，恒等矩阵、质量为 0，但保留默认 `unknown_gmc_speed_px_s=1500`。
它与有 GMC 组保留相同的运动不确定性参数，用于区分删除补偿估计和同时删除容错项的影响。

## 完整结果

| Detector | ID 策略 | GMC | 有 ID 的框 | 无 ID 的框 | 全片不同 ID 数 | 13-14 秒主检测有 ID 帧数 | 13-14 秒主检测 ID |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| P3 | 常规确认 | 开 | 485 | 62 | 5 | 30/30 | 始终 1 |
| P3 | 常规确认 | 关 | 455 | 92 | 5 | 28/30 | 1 -> 5 |
| P3 | 常规确认 | 关，保留默认不确定性 | 479 | 68 | 4 | 30/30 | 始终 1 |
| P3 | 首帧 ID | 开 | 546 | 1 | 29 | 30/30 | 始终 1 |
| P3 | 首帧 ID | 关 | 545 | 2 | 46 | 30/30 | 1 -> 8 |
| P3 | 首帧 ID | 关，保留默认不确定性 | 544 | 3 | 30 | 30/30 | 始终 1 |
| P2+P3 | 常规确认 | 开 | 545 | 121 | 8 | 30/30 | 始终 1 |
| P2+P3 | 常规确认 | 关 | 514 | 152 | 9 | 28/30 | 1 -> 8 |
| P2+P3 | 常规确认 | 关，保留默认不确定性 | 537 | 129 | 8 | 30/30 | 始终 1 |
| P2+P3 | 首帧 ID | 开 | 665 | 1 | 46 | 30/30 | 始终 1 |
| P2+P3 | 首帧 ID | 关 | 662 | 4 | 68 | 30/30 | 1 -> 15 |
| P2+P3 | 首帧 ID | 关，保留默认不确定性 | 659 | 7 | 50 | 30/30 | 始终 1 |

表内普通“关”表示纯图像运动配置，不确定性项同时为 0；“关，保留默认不确定性”是仅移除 GMC 估计的补充控制。
每行有 ID 与无 ID 的框相加，都等于对应 detector 的全部框数；没有观测被丢弃。
全片不同 ID 数包括误检候选和碎片轨迹，不是无人机数量，也不是正式 IDSW。
13-14 秒诊断检查逐帧最高分检测（索引 0）及其关联；该片段已用于肉眼排查单个靶机，不是完整 GT 跟踪评测。
由于此视频没有提供完整 GT，不报告 Precision、Recall、FP 或 IDF1/HOTA。

### 关键片段

无 GMC 的常规确认组在索引 393、394（13.10、13.133 秒）有目标检测框但无确认 ID。
索引 395（13.167 秒）起，P3 变为 ID 5，P2+P3 变为 ID 8。
关闭 GMC 且使用首帧 ID 后，这段没有等待帧，但仍发生换号。
保留 GMC 的两种 ID 策略均在该片段保持 ID 1。

补充控制不计算 GMC，但两种 ID 策略均在此段保持 ID 1，且每帧都有主目标 ID。
因此不能仅根据最初四组实验断言该片段必须依赖 GMC。
合理的运动容错在这里也能维持连续性；它是否造成错关联/精度损失，仍需要有 GT 的视频验证。
取消 PENDING 可以让新 ID 更早出现，但不能单独修复错误关联导致的 ID 跳变。

### 为什么仍有一个无 ID 框

开启 GMC 的首帧 ID 组，两种 detector 均有一个 `ambiguous` 框：
索引 1240，41.333 秒，置信度约 0.03037。
该帧在画面左侧有多个重叠检测，不能可靠指定该框属于哪个既有身份，因此保留框，但不强挂旧 ID。
本组已经没有 `pending` 或 `unassigned` 输出；这个剩余框不是在等确认，而是身份关联有歧义。
若仅需要“每个框一个唯一编号”，可以额外使用 observation ID，但不能把它当作跨帧稳定的目标 ID。

## PENDING 的含义

新检测框是否值得保留、是否能确认跨帧身份，是不同决策。
原默认检测阈值为 `0.03`，新建轨迹阈值为 `0.10`；因此低分框虽然保留，但可能始终不建立轨迹。
已建立的新轨迹还要通过 3/4 次观测和平均置信度检查。
原可视化把未关联低分框也显示为 PENDING，不是所有 PENDING 都只等前两帧。
歧义关联显示 UNCERTAIN，不强行给出身份。

直接给 ID 技术上可行，首帧组就是这个实验。代价是更多低置信度检测会获得身份：
有 GMC 时，P3 的不同 ID 数由 5 增至 29；P2+P3 由 8 增至 46。
这不意味着发现了更多无人机，也不构成检测或跟踪精度提升。
默认确认配置和板端生产默认算法没有自动修改。

## 本地视频

目录：`/Users/czyczyyzc/Documents/codes/ultralytics_yolov8/deliverables/clip_000002_gmc_id_ablation_20261008/`

- `000002_P3_GMC_ID_4way_default_uncertainty.mp4`：推荐先看，P3 的有 GMC / 无 GMC 保留默认不确定性对照。
- `000002_P2P3_GMC_ID_4way_default_uncertainty.mp4`：同上，P2+P3。
- `000002_P3_GMC_ID_4way.mp4`：额外控制，右侧关闭 GMC 且去掉不确定性项。
- `000002_P2P3_GMC_ID_4way.mp4`：同上，P2+P3。
- `000002_P3_first_frame_ID_GMC.mp4`：P3 首帧 ID + GMC 单独完整版。
- `000002_P2P3_first_frame_ID_GMC.mp4`：P2+P3 首帧 ID + GMC 单独完整版。

四格位置：左上为常规确认 + GMC，右上为常规确认、关闭 GMC；左下为首帧 ID + GMC，右下为首帧 ID、关闭 GMC。
`default_uncertainty` 视频的右侧保留默认运动不确定性，其他四格视频右侧不保留，切勿混淆。
四格输出 `3200x1568`，单格/单独版 `1600x784`；均完整 1800 帧、60 秒、30 FPS。
使用 1 px 四角框、干净原图裁剪后放大，无 GT、无十字。
原始 ID 不重映射；视频播放 FPS 不等于板端 inference FPS。

`metadata/COMPARISON_EXTENDED.json` 包含完整 12 组参数、哈希、逐帧关键区间 ID、无 ID 观测清单及完整输出统计。
`metadata/COMPARISON.json` 保留最初 8 组的控制结果，不含补充的不确定性保留组。
其他 metadata 子目录保留每组 `tracks.jsonl`、协议、跟踪摘要和编码摘要。

## 47 服务器复现

实验根目录：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/clip_000002_gmc_id_ablation_20261008/`

缓存目录命名：`{p3,p2p3}_{balanced,immediate}_{estimate,disabled,unavailable}/`。
对应 `_visual/` 存放单独版视频；`p3_grid/`、`p2p3_grid/` 存放四格。
首帧配置：实验根目录 `immediate_id_config.json`。
原生关联库：`native/libmotion_tracker.so`，为服务器 x86_64 构建，不能直接复制到 ARM 板子使用。

复现脚本：
`/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/track_cached_native_dist_video.py`

审计脚本：
`/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/compare_motion_gmc_ablation.py`

无 GMC 的 `--gmc-mode disabled` / `unavailable` 是离线回归参数，目前不是板端 native executable 的参数。
首帧 ID 参数可通过 native 的既有 `--motion-params` ABI 实现，但 nominal FPS 必须与实际源 FPS 一致。
本次仅修复首帧确认模式的确认次数计数，不改变 balanced 默认的关联输出。
本地与 47 服务器均 36 tests passed；服务器另有 4 subtests passed。
