# 公共无人机数据混合训练研究实验

## 范围与授权

本轮仅用于内部研究对照，用户已确认。数据可公开下载不等于模型已获商用交付许可；交付前仍须核验每个数据源的条款。本次不代签数据协议，不绕过登录或暂停开放限制。

47 服务器工作目录：`/mnt/chenziye/datasets/anti_uav/public_research_20261105/`。

| 数据源 | 官方入口 | 接入原则 |
| --- | --- | --- |
| DUT Anti-UAV | https://github.com/wangdongdut/DUT-Anti-UAV | 复核已有原始数据；只用官方 detection train 训练，val/test 不进入训练。Tracking 部分不擅自改为训练集。 |
| Halmstad | https://github.com/DroneDetectionThesis/Drone-detection-dataset | 复核已有原始数据；本次接入可见光 DRONE 正样本，按完整视频划分。先验证 MATLAB 标注与视频帧的对应。 |
| UAVSwarm | https://github.com/UAVSwarm/UAVSwarm-dataset | 下载官方仓库；只用人工 GT，不用 det 目录的检测器伪标签；保留官方 train/test。 |
| Det-Fly | https://github.com/Jake-WU/Det-Fly | 使用官方 OneDrive 数据链接；若链接失效或要求登录则留待提供有效访问，不使用来源不明镜像。 |
| Drone-vs-Bird | https://github.com/wosdetc/challenge | 视频需要申请并签署数据使用协议，尚未代为申请或签署，不计为已下载/已训练。 |
| LRDDv3 | https://research.coe.drexel.edu/ece/imaple/lrddv3/ | 官方暂停数据访问，暂不接入。 |

## 对照方案

69 视频基线保持运行，不修改其数据、标注、权重或配置：

`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/approved69_legacy52_dataonly_20261105/`

新增实验的固定方案：

1. 保留基线自采 69 段视频和已有 Anti-UAV300 RGB 样本，原训练清单前缀逐项保留，包括重复次数。
2. 追加通过转换与检查的公共数据正样本，每个正样本每轮追加一次。公开数据不是额外的自采视频数。
3. 从原审核负样本池中调整每轮负样本预算，使总负样本比例仍约 15%；不把未知类别或缺标图片当作负样本。旧正样本曝光不削减。
4. 不把 Halmstad 的 AIRPLANE/HELICOPTER 视频自动当负样本，避免伤害当前固定翼靶机任务。
5. 使用同一公开 `yolov8n.pt` 初始化；P3 initial 30 epochs，Frozen-P3 + Add-on P2 Teacher 12 epochs，P3 Student 蒸馏 15 epochs。无 QAT，无额外定位微调阶段。
6. 固定输入 `960x544`，batch 64、nbs 128、seed 20261015；其余基础增强和蒸馏权重沿用基线。原有安全背景缓存的在线换贴图概率为 0.5，公共 RGB 样本不强行换贴图。
7. Video00009 继续选权重，Video00004 仅测试。公共 val/test 不进入训练，也不用于选取灰度最优权重。

同轮数会因新增数据增加优化器步数，属于数据扩充研究，不是等计算量实验。

## 代码与产物

仓库：`/mnt/chenziye/codes/ultralytics_yolov8/`。

- `scripts/anti_uav/prepare_public_uav_research.py`：DUT/Halmstad 转换与来源记录。
- `scripts/anti_uav/prepare_uavswarm_research.py`：UAVSwarm 人工 GT 转换。
- `scripts/anti_uav/run_public_uav_research.py`：追加公共正样本、构建固定标签缓存、启动独立三阶段训练并与 69 视频基线比较。
- `scripts/anti_uav/run_p3_independent_curriculum_training.py`：现有训练方法，保持不改。

转换目录的 `manifest.jsonl` 记录来源、划分、原图/标签和视频组；`summary.json`、排除清单记录实际可用数量与异常。下载成功数量不能替代可安全用于训练的样本数量。

混合目录的 `manifest.json`、`public_accepted.json`、`public_excluded.json` 记录追加量、每轮正负曝光、精确像素去重和排除原因。精确像素去重不代表做过所有近重复图像或跨数据源场景去重，不能据此宣称完全不存在近重复。

训练目录的 `launch.json`、`protocol.json`、`research_status.json`、`status.json`、`training.log` 记录启动命令、方法、阶段和逐批进度。训练完成后才会生成 `comparison/results.json`，包含两组各三阶段模型在原 Video00004/Video00009 协议上的检测对照及合并指标；合并结果包含选模验证视频，不应称为纯独立测试集结果。

服务器 GitHub SSH 若不可达，仍使用本地 git push 后传输同一 commit 的 bundle，由服务器 git pull --ff-only 同步，不用散拷源代码替代版本同步。
