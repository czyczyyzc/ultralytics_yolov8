# 000002 因果跟踪优化结果

日期：2026-10-08  
代码提交：`74c0e3fc406ec850fdbe0827af115de131c4b225`  
协议：`960x544`、`conf=0.03`、PT FP32 detector 缓存、逐帧因果关联、无未来帧等待、无 ID 后处理重映射。

## 结论

- P3 的 547 个 detector 框全部进入跟踪 observation 输出；P2+P3 的 666 个框也全部进入输出，没有“detector 有框但 observation 丢框”。
- 原 15.8 秒故障已修复。P3 与 P2+P3 在第 470-475 帧都保持主目标为确认态 `ID 1`。
- 第 432-439 帧 detector 本身无框；第 440 帧恢复检测后仍继续 `ID 1`，未因 8 帧检测空窗换 ID。
- 两版 1800 帧结果中，相邻且都只有一个确认目标的帧之间，ID 变化均为 0。
- P2+P3 在第 474 帧对同一大目标产生完整框和嵌套局部框。完整框保持 `ID 1`；局部框保留并标记 `UNCERTAIN`，不再伪造第二个目标 ID。
- `000002` 没有完整人工 identity GT，因此以上是缓存一致性、关键事件和连续确认输出审计，不能表述为全片正式 IDF1/HOTA 或“绝对零错误 ID”。

## GMC 对照

在有单目标 GT 的 Video00009 上复测同一关联代码。该回归沿用现有
old-28 P2+P3 detector 固定缓存，只用于隔离比较 tracker/GMC，不代表
44-video detector 的检测精度：

| 模式 | 连续 GT 段 ID 变化（含 tentative） | 确认输出 TP / FP / FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| 质量门控 GMC | **0** | 5296 / 3087 / 2686 | 63.18% | 66.35% |
| 不运行 GMC | 17 | 5281 / 3025 / 2701 | 63.58% | 66.16% |

不运行 GMC 虽略降 FP，但连续身份稳定性明显变差。因此交付配置保留 GMC；GMC 失败帧由质量分数触发不确定性回退，不把 identity warp 当成“相机绝对静止”。原生程序同时保留 `--gmc-mode unavailable`，仅用于消融或无 GMC 环境，不作为默认配置。

所有 observation 输出在 Video00009 上与 detector 框完全一致：TP / FP / FN 为 5302 / 3806 / 2680。确认态指标单独统计，不能用确认过滤后的框数代表 detector 性能。

## 实现变化

- 用全局可行分配的替代解差值判断真正身份歧义，避免局部 score gap 误判。
- candidate ID 与确认状态分离；候选从首个合格观测获得 ID，达到 3/4 命中和均值置信度后以同一 ID 晋级。
- 运动门限随已观测目标尺度、Kalman 不确定性和可靠相机运动自适应；大目标快速位移不再受固定 240 px 上限截断。
- 已确认连续目标的残差用于估计未建模图像运动上限，但不会从未匹配候选学习并全局放宽门限。
- 新候选若强烈嵌套于当前已确认目标，保留原 detector 框但输出 `UNCERTAIN`；两个已确认目标不会被此规则合并。
- 输出包含每个 detector observation。候选显示 `ID n?`，确认显示 `ID n`，无可靠身份时显示 `DET` 或 `UNCERTAIN`。

## 开销

47 服务器 x86_64 上只测原生 C++ 关联与 observation ABI，不含 detector、GMC、解码或摄像头：

| 模型输出 | 平均 | P95 | P99 |
| --- | ---: | ---: | ---: |
| P3 | 0.000974 ms | 0.003012 ms | 0.004355 ms |
| P2+P3 | 0.001373 ms | 0.005206 ms | 0.007393 ms |

该结果不能替代 RK3588S 的曝光开始到结果延迟测试。算法没有等待未来帧，因此不会引入短窗确认延迟；真实 `曝光开始 -> 完整结果 <=30 ms` 仍需在板端用可靠 SOE 时间戳验证。

## 文件

- 本地 P3 视频：`deliverables/clip_000002_causal_global_20261008/visualizations/000002_P3_causal_tracker_quality_gated_GMC.mp4`
- 本地 P2+P3 视频：`deliverables/clip_000002_causal_global_20261008/visualizations/000002_FrozenP3_AddonP2_causal_tracker_quality_gated_GMC.mp4`
- 服务器实验根目录：`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/clip_000002_causal_global_20261008/`
- 服务器 x86_64 测试库：`native/release_74c0e3f/libmotion_tracker.so`
- 源码：`scripts/anti_uav/dist_native/motion_tracker.cpp`、`global_assignment.hpp`、`video.cpp`

视频均为 H.264、1600x784、30 FPS、1800 帧、60 秒：

- P3 SHA256：`59d0f3575c7751bd8924cc83d88289a4dd3fea5da624eacada156df8abda9e02`
- P2+P3 SHA256：`931aa1d9b6bec1030f8717df763fbc64749a1058e10e8d2a40452f79d0b212cf`

服务器测试库是 x86_64，不能直接复制到 RK3588S。板端必须从同一提交重新编译 ARM64 库。

## 验证

- macOS：54 tests passed。
- 47 服务器：54 tests + 4 subtests passed。
- Linux 原生视频程序：C++17 syntax check passed。
- 两段本地视频 SHA256、帧数、帧率、时长与服务器 summary 一致。
