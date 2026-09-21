# 无人机检测训练平台接口与集成说明

版本：1.0.0　日期：2026-09-21　适用服务器：47（47.107.185.207）

## 1. 交付范围

本文用于平台前后端联调和训练 worker 开发。目标流程为：

**选择数据集版本 → 选择初始化权重和 GPU → 启动训练 → 查看 loss/validation 曲线 → 选择某轮 checkpoint → 图片/视频推理。**

底层训练和模型加载已有可运行实现；**本文 HTTP API 是平台需要实现的接口契约，当前仓库没有因此新增一个已经启动的 Web 服务。** 不应将示例接口地址、任务 ID 或数据集 ID 当作现成服务。

交付文件：

| 文件 | 用途 |
| --- | --- |
| [openapi.yaml](openapi.yaml) | OpenAPI 3.0.3 契约，可导入 Swagger Editor / 支持 OpenAPI 的接口工具 |
| [examples/read_training_metrics.py](examples/read_training_metrics.py) | 无 Torch 依赖的现有 CSV 曲线读取参考实现 |
| [examples/predict_checkpoint.py](examples/predict_checkpoint.py) | 加载可信 PT 权重、固定 544×960 输入的推理参考 worker |

本版优先复用已经验证的训练方案，不另造训练框架。默认模型为 **P3 训练完成后冻结，再训练 Add-on P2**；P3 阶段权重和最终 Add-on 阶段权重都允许单独推理。Tracker 不属于检测训练 loss，也不包含在本接口的检测精度指标中。

交接 ZIP 仅包含接口说明和示例，不是独立训练 SDK。参考脚本应在完整仓库的 `docs/anti_uav_training_platform/examples/` 位置运行，直接使用已同步的 47 服务器版本即可。

## 2. 现有代码和真实路径

本地仓库：`/Users/czyczyyzc/Documents/codes/ultralytics_yolov8`

47 服务器仓库：`/mnt/chenziye/codes/ultralytics_yolov8`

Python：`/mnt/chenziye/codes/ultralytics_yolov8/.venv/bin/python`

本次只读核对环境：Python 3.10.12、Torch 2.5.1+cu121、仓库 Ultralytics 8.2.82。必须使用本仓库自定义模块，不能用一份普通 `pip install ultralytics` 替代。以下实现依据提交 `64e16b8` 核对；每次平台任务另记录实际 Git 完整 SHA、环境版本和配置快照。

| 能力 | 仓库相对路径 | 集成说明 |
| --- | --- | --- |
| 单个数据集的两阶段训练 | `scripts/anti_uav/run_rebalanced_fullscale_training.py` | 推荐 worker 入口；不要启动双实验对比脚本代替单个任务 |
| 历史 14/28 视频对比实验 | `scripts/anti_uav/run_native_pool_comparison.py` | 参考历史协议；其中包含固定来源和两组完整训练，不是通用平台入口 |
| 灰度验证和 best 选择 | `scripts/anti_uav/gray_deployment_trainer.py` | `FixedShapeGrayP3Trainer`、`FixedShapeGrayAddOnTrainer`、`FixedShapeGrayValidator` |
| 冻结策略 | `scripts/anti_uav/frozen_p3_addon_p2_trainer.py` | 仅 P2 adapter 和 P2 分类/回归分支可训练；旧参数和 BN/EMA 缓冲保持冻结 |
| Add-on 初始化和校验 | `scripts/anti_uav/train_frozen_p3_addon_p2.py` | `initialize_addon_model`、`verify_legacy_outputs` |
| 全量标注抽帧追加 | `scripts/anti_uav/append_approved_gray_native.py` | 新 approved 视频追加到既有原图基线，保留原清单的重复曝光次数 |
| approved 标注校验 | `scripts/anti_uav/build_approved_gray_rehearsal.py` | `validate_frame_sets`、`extract_task` |
| 原图/负样本曝光采样 | `scripts/anti_uav/label_pool_sampling.py` | 候选池大小不等于每轮样本数 |
| checkpoint、CSV、callbacks | `ultralytics/engine/trainer.py` | `save_model`、`save_metrics`、`on_model_save`、`on_fit_epoch_end` |

### 2.1 可登记的现有数据和模型

以下路径已在 47 核对，平台 ID 是建议登记名，不是现成数据库记录。

| 建议 ID / 用途 | 服务器路径 |
| --- | --- |
| `ds_gray28_20260916` 数据集根目录 | `/mnt/andrew/anti_uav_model_refinement/data/real_gray_native_fullpool_28videos_20260916` |
| 训练配置 | 上述目录的 `train_hardneg_gray_monitor.yaml` |
| 数据溯源 | 上述目录的 `manifest.json`、`snapshot.json` |
| approved 标注来源 | `/mnt/andrew/anti_uav_model_refinement/data/approved_tasks/` |
| 原始新视频 | `/mnt/andrew/video-labeler/videos/` |
| 旧视频来源 | `/mnt/andrew/anti_uav_model_refinement/data/seven_old_videos/` |
| 已完成的 28 视频训练 | `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28` |
| 该任务 P3 best | 上述训练目录的 `training_p3/p3/weights/best.pt` |
| 该任务 Add-on P2 best | 上述训练目录的 `training_addon/p2/weights/best.pt` |

初始化权重 `ckpt_p3_seed` 应登记为：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_strict_holdout_Video00004_newclips01_20260902/training/strict_holdout_Video00004_neg15_newclips01_v1_20260902/weights/best.pt
```

现有入口强制要求 `--old-run`，即使加 `--skip-final-test` 也需要传入该参数。当前可用值为：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_frozen_p3_addon_p2_manual_clips0123_20260904
```

此参数在跳过最终测试时不是训练初始化来源。不要为了填参数改用它的 Add-on 权重初始化 P3。

## 3. 数据集选择与审核规则

训练实际读取 **图像及 YOLO 标签**，不直接随机读取原始 MP4。平台中的“选择视频数据集”需要先完成 approved 标注核验、抽帧、清单和 manifest 固化；耗时准备过程应由独立 worker 完成，不能放在启动训练的 HTTP 请求内。

v1 的数据集选择器只显示已经登记且 `state=ready`、与 profile 兼容的不可变版本。任意选择若干视频并自动重建任意划分，属于后续数据准备服务；现有追加脚本有原图基线、独立验证视频等限制，不能包装成无限制通用转换器。

必需内容为 `manifest.json`、训练 YAML、train/val 列表、图像、对应标签，以及 YAML 引用的负样本池。服务端记录配置、清单、标签快照和来源视频 SHA256；仅哈希一个 YAML 不能保证数据内容未变。按任务固定这些版本，运行中不得修改。

| 项目 | 规则 |
| --- | --- |
| 正样本 | approved 的 included 帧且有合法目标框；多框保留所有合法框 |
| 负样本 | 人工明确确认无目标的 `negativeFrameIndices`，对应空标签 |
| 不确定 / 未审核 | `excludedUncertainFrameIndices`、`excludedUnreviewedFrameIndices` 必须排除，不能当负样本 |
| 缺失标签 | 报错，不得静默当负样本 |
| 目标大小 | 不以面积 10%、25% 或 80% 为由删除合法框；仅校验有限数、合法类别和几何有效性 |
| 重复曝光 | 旧清单的重复次数是采样设计，不能随意 `set()` 去重 |
| 泄漏隔离 | 至少按整段视频 SHA 隔离 train/validation/test；重编码同源和同次采集仍需额外审核 |
| 初始化模型 | 审核其训练来源及可追溯配置；仅当前清单不含测试视频不足以证明无泄漏 |

当前 28 视频方案的划分：28 个自采灰度训练视频，加既有混合训练来源；`Video00009` 作为灰度验证来源，`Video00004` 为测试保留集。训练中的 `val_monitor.txt` 还含单列报告的 zoom 压力样本。**它不是完整 Video00009 全帧评测；曲线指标只针对该清单。** 后续若把 04+09 合并展示，应标注“包含用于选权重的验证视频”，不得称为全独立测试。

已核对当前采样 YAML：`anchor_slots=121353`，轮换负样本池 `25540`，每轮从池中取 `5754`，所以每轮 `127107` 个曝光位置；候选列表为 `146893` 个位置。不是 127107 张唯一图像，也不是每轮把全部负样本都训练一遍。约 15% 是整体每轮负样本目标比例，不是对每张正图保留 85% 的独立抽签概率。

平台 API 应分别展示视频数、唯一样本数、候选曝光数、每轮曝光数和排除数。没有实际统计的字段返回 `null`，不能猜测填 0。

## 4. 训练配置与现有 CLI 映射

v1 profile：`frozen_p3_addon_p2_gray_v1`。

| 字段 | v1 支持 | 底层行为 |
| --- | --- | --- |
| `dataset_id` | 选择已注册版本 | 映射 `--dataset` 和可选 `--data-yaml` |
| `initial_checkpoint_id` | 选择合规的 P3 初始化 PT | 映射 `--initial-p3`；其 `train_args.data` 和原始清单还必须可访问 |
| `epochs_per_stage` | 默认 15，两阶段相同 | 映射 `--epochs`；默认总共 15+15，不是总共 15 |
| `resource_id` | 一张经授权预约的 GPU | 映射 `--device`；不是默认占用 6 号卡 |
| 输入 | 固定 H×W=`[544,960]` | 展示为 W×H=`960×544`，不要交换 |
| batch / nbs | 固定 64 / 128 | 当前两阶段入口写死，batch 不等于 nbs；要可配需另改代码并验证 |
| 优化器 | AdamW，P3 lr0=0.0001，P2 lr0=0.001 | 其他参数以实际 `args.yaml` 为准 |
| 原版增强 | scale=.2，translate=.05，fliplr=.5，mosaic/mixup/copy_paste=0 | 不默认添加在线换贴图或额外大目标裁剪 |
| `checkpoint_publish_every` | 默认 5 | 平台按一基 epoch 选择对外发布；不改变底层每轮写盘 |
| 多 GPU | 不支持此 profile | `FixedShapeSelectionMixin` 对 WORLD_SIZE>1 会报错 |
| 任意大小/任意类别/P3-only 训练 | 不在 v1 | 不能静默忽略请求字段；返回 422 |
| 在线贴图增强 | 不在 v1 开关中 | 现有代码有能力但依赖专用缓存/配置，需单独 profile 版本和审核 |

选中的 GPU 可能已被其他任务使用。GPU 显存还剩很多不代表可以共享；默认排队、独占预约，并检查平台外进程。不要 `kill` 他人的进程。取消训练不会意味着永久修改 GPU 或系统设置。

下面是现有入口的可用调用示例，**只作 worker 实现参考，不在交付文档时自动启动训练**：

```bash
cd /mnt/chenziye/codes/ultralytics_yolov8
env -u CUDA_VISIBLE_DEVICES -u ANTI_UAV_TRUST_DATASET_CACHE \
  PYTHONPATH=/mnt/chenziye/codes/ultralytics_yolov8 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg \
  WANDB_MODE=disabled WANDB_DISABLED=true \
  YOLO_CONFIG_DIR=/mnt/chenziye/codes/ultralytics_yolov8/runs/platform/train_EXAMPLE/yolo_config \
  .venv/bin/python -u scripts/anti_uav/run_rebalanced_fullscale_training.py \
  --dataset /mnt/andrew/anti_uav_model_refinement/data/real_gray_native_fullpool_28videos_20260916 \
  --run-dir /mnt/chenziye/codes/ultralytics_yolov8/runs/platform/train_EXAMPLE \
  --initial-p3 /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_strict_holdout_Video00004_newclips01_20260902/training/strict_holdout_Video00004_neg15_newclips01_v1_20260902/weights/best.pt \
  --old-run /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/real_gray_yolov8n_frozen_p3_addon_p2_manual_clips0123_20260904 \
  --device 6 --epochs 15 --fixed-validation --skip-final-test
```

`6` 只是物理编号示例，必须由资源调度器替换为已预约 GPU。worker 先建立独有目录并准备字体/配置目录，清除不兼容的分布式环境，固定 working directory。若使用 `CUDA_VISIBLE_DEVICES=<已预约GPU>` 隔离，则进程内 `--device 0`；不能保留隔离后的可见 GPU 再传物理编号 6。

HTTP 后端使用参数数组 `subprocess.Popen(argv, cwd=..., env=..., start_new_session=True)` 或等价容器任务，不使用 `shell=True` 拼接用户输入。每任务独立运行目录、YOLO_CONFIG_DIR、日志、进程组和资源租约。不要在处理 Web 请求的进程内执行 `model.train()`。

## 5. REST API 与前端流程

基路径 `/api/v1`，Bearer 鉴权。详细字段、类型和必填项以 `openapi.yaml` 为准。列表只返回当前用户/项目授权记录；示例 ID 均需要平台先登记。

| 操作 | 接口 | 返回 |
| --- | --- | --- |
| 获取可用能力 | `GET /capabilities` | profile、后端、是否支持恢复等 |
| 数据集选择 | `GET /datasets`、`GET /datasets/{dataset_id}` | 不可变版本、划分、样本统计 |
| GPU 选择 | `GET /resources` | 资源可用/预约/外部占用状态 |
| 初始/中间模型选择 | `GET /checkpoints?job_id=...&stage=addon_p2` | 可信且 ready 的 checkpoint；不带 job_id 可含初始化模型 |
| 启动 / 查看任务 | `POST /training-jobs`、`GET /training-jobs/{job_id}` | 202 排队，随后轮询状态 |
| 曲线 / 日志 | `GET /training-jobs/{job_id}/metrics`、`.../logs` | 增量 epoch 点、游标日志 |
| 取消任务 | `POST /training-jobs/{job_id}/cancel` | 202，终止确认后才变 cancelled |
| 输入图片/视频 | `GET /media`、`POST /media` | 已授权媒体或 multipart 上传；不是权重上传 |
| checkpoint 推理 | `POST /inference-jobs`、`GET /inference-jobs/{job_id}` | 检测 JSONL、若干预测预览图 |
| 带 GT 独立评测 | `POST /evaluation-jobs`、`GET /evaluation-jobs/{job_id}` | validation/test 固定协议指标 JSON |
| 产物下载 | `GET /artifacts/{artifact_id}/download` | 经鉴权文件流 |

### 5.1 启动训练示例

```http
POST /api/v1/training-jobs
Authorization: Bearer <token>
Idempotency-Key: train-button-20260921-0001
Content-Type: application/json
```

```json
{
  "dataset_id": "ds_gray28_20260916",
  "profile_id": "frozen_p3_addon_p2_gray_v1",
  "initial_checkpoint_id": "ckpt_p3_seed",
  "resource_id": "gpu47-6",
  "epochs_per_stage": 15,
  "checkpoint_publish_every": 5
}
```

返回示例（此处为接口示意，非已启动训练）：

```json
{
  "id": "train_20260921_0001",
  "kind": "training",
  "state": "queued",
  "stage": null,
  "created_at": "2026-09-21T10:00:00+08:00",
  "updated_at": "2026-09-21T10:00:00+08:00",
  "epoch": 0,
  "epochs_per_stage": 15,
  "global_epoch": 0,
  "resource_id": "gpu47-6",
  "dataset_id": "ds_gray28_20260916",
  "resolved_config": {"input_hw": [544, 960], "batch": 64, "nbs": 128},
  "artifacts": []
}
```

同一用户/接口的同幂等键、同请求返回原任务；同键不同请求返回 409。排队重试不能重复训练。worker 领取任务需数据库原子 claim/租约，不能只依赖脚本的目录锁。

### 5.2 状态与进度

`state` 为 `queued → running → succeeded/failed`；取消为 `cancel_requested → cancelled`。终态不可被滞后的 status.json 覆盖。

| 脚本 status.stage | 平台 stage 建议 |
| --- | --- |
| 启动前数据/模型核验 | `preparing` / `validating` |
| `training_p3` | `training_p3` |
| `initialize_addon` | `initializing_addon` |
| `training_addon` | `training_addon` |
| 训练后冻结检查与产物注册 | `verifying` / `finalizing` |
| `weights_ready` | **不是单凭此文件就成功**；确认进程退出 0、两阶段产物和 frozen_checks 全通过后，state=succeeded |
| `failed` | failed，并保留错误摘要与日志 |

`status.json` 仅最新快照，每轮覆盖；曲线不能从它恢复。某些失败发生在脚本 try 范围外、取消也不保证写 status，因此还必须采集进程退出码、worker heartbeat 和日志。运行目录 PID 不能替代可重连的任务状态。

前端建议每 2–5 秒查状态/日志，每 5 秒查曲线。v1 曲线是 **epoch 粒度**：第一轮尚未完成时返回空列表，并显示“等待首轮验证”，不要补零。若需 batch loss 和步数，需要额外接入 `on_train_batch_end` 输出结构化事件；当前 CSV 不提供该能力，不能靠解析 tqdm 承诺精确进度。

两阶段轮数分别从 1 开始。API 的 `global_epoch/sequence` 在 P3 为 epoch，在 P2 为 `epochs_per_stage + epoch`；UI 显示阶段分界，**两阶段 loss 不可理解为同一条连续优化曲线**。验证期 GPU 较闲不代表卡死，ETA 应根据实际阶段历史耗时估算，无数据时返回未知。

### 5.3 取消与恢复

取消排队任务不启动进程；运行任务向其专属进程组发送温和终止信号，超时后仅结束该组；确认子进程退出后释放 GPU。保留最后一个已完成并校验的 checkpoint，不保证保存当前 batch/未完成 epoch。

v1 **不提供暂停/恢复训练按钮**。现有 `--resume-p3` 仅适用于 P3 中断且 optimizer/epoch/配置仍符合检查、尚未进入 Add-on 的情形；不是整个两阶段任务通用 resume。`train_frozen_p3_addon_p2.py --resume-model` 是从已有权重开始另一阶段，不等同于恢复原优化器进度。未来 resume API 需独立实现 Add-on 恢复、采样 epoch/随机状态和审计协议。

## 6. Loss 与 validation 指标

真实输出：

```text
<run_dir>/protocol.json
<run_dir>/status.json
<run_dir>/training_p3/p3/args.yaml
<run_dir>/training_p3/p3/results.csv
<run_dir>/training_p3/p3/weights/{best,last,epoch0,...}.pt
<run_dir>/addon_initialization.json
<run_dir>/training_addon/p2/args.yaml
<run_dir>/training_addon/p2/results.csv
<run_dir>/training_addon/p2/weights/{best,last,epoch0,...}.pt
<run_dir>/frozen_checks.json
```

CSV 表头有空格，要 strip；只读完整换行行，忽略正在写入的尾行，按 `(job, stage, epoch)` 去重。NaN/Infinity 映射 JSON `null`。CSV 仅保留约五位有效数字，不是训练时所有浮点值的完整精度。

| CSV key | 展示名称 / 口径 |
| --- | --- |
| `train/box_loss`, `train/cls_loss`, `train/dfl_loss` | 每轮训练损失，分别画线 |
| `val/box_loss`, `val/cls_loss`, `val/dfl_loss` | 验证损失；验证清单可能同时含 native/zoom |
| `native/mAP50`, `native/mAP50-95` | 原生灰度验证子集 AP；不是固定 conf=.03 下的单点 P/R |
| `native/c0.03/P`, `native/c0.03/R`, `native/c0.03/F2` | conf=.03、匹配 IoU=.5 的点指标 |
| `native/c0.03/TP`, `/FP`, `/FN`, `/FRAMES` | 聚合计数；不是错误帧数，FP 是错误框数 |
| `native/c0.03/FP1000` | 每千帧误检框数，可大于 1000，不是百分比 |
| `native/c0.03/long_4to8px/R`, `/GT` | 固定输入空间目标长边 4–8 px 的召回和 GT 数 |
| `native/c0.03/area_ge80pct/R`, `/GT` 等 | 原图框面积分桶；没有 GT 时 R=null，不是 0% |
| 对应 `c0.01` / `c0.05` | 同时展示其他低阈值点 |
| `zoom/*` | 合成放大压力验证，单独展示，不参与 best 选择 |
| `metrics/precision(B)`, `metrics/recall(B)` 等 | Ultralytics 的通用汇总，不标成固定 .03 的 native 指标 |
| `lr/pg0`, `lr/pg1`, `lr/pg2` | 参数组学习率 |

比例 API 为 0..1，前端乘 100 显示百分比；所有 Precision/Recall 卡片同时注明 split、conf、匹配 IoU、输入尺寸和 checkpoint。分桶 R 不对应一个可直接定义的分桶 FP，不能凭 R 反推分桶 Precision。

当前 best 选择为：

```text
fitness = 0.5 × native F2(conf=0.03)
        + 0.3 × native mAP50
        + 0.2 × native mAP50-95
F2 = 5PR / (4P + R)
```

验证生成候选的 conf 下限 .001、NMS IoU .45、max_det=100；AP50 的匹配 IoU=.5，AP50-95 采用 .5:.95。`best.pt` 不一定是 mAP50 最高或 loss 最低的一轮。验证函数会 pop 掉 fitness 后再写 CSV，因此 CSV 通常没有原始 fitness 列；可为画图重建 `selection/fitness_reconstructed`，但不据此重新认定 best。best/epoch 绑定以保存时回调和 checkpoint 元信息为准。

曲线接口示例：

```http
GET /api/v1/training-jobs/train_20260921_0001/metrics?after_sequence=15
```

```json
{
  "items": [{
    "sequence": 16,
    "stage": "addon_p2",
    "epoch": 1,
    "global_epoch": 16,
    "source": "results.csv",
    "values": {
      "train/box_loss": 1.372,
      "train/cls_loss": 0.98306,
      "train/dfl_loss": 1.6302,
      "native/mAP50": 0.56425,
      "native/c0.03/P": 0.50506,
      "native/c0.03/R": 0.6261,
      "native/c0.03/TP": 499,
      "native/c0.03/FP": 489,
      "native/c0.03/FN": 298,
      "native/c0.03/area_ge80pct/R": null
    }
  }],
  "latest_sequence": 16
}
```

示例数值取自现有 28 视频训练 Add-on 第 1 轮 CSV 的部分字段，不表示最终模型完整 Video00009 表现。实用解析方式：

```bash
.venv/bin/python docs/anti_uav_training_platform/examples/read_training_metrics.py \
  --run-dir runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28 \
  --epochs-per-stage 15 --after-sequence 0
```

## 7. Checkpoint 保存、发布和加载

### 7.1 保存机制和轮次

当前两阶段入口固定 `save=True, save_period=1`，所以每轮有 epoch 文件，并更新 last；fitness 最佳时更新 best。真实文件名从 0 开始：`epoch0.pt` 对应 UI 第 1 轮，`epoch9.pt` 对应第 10 轮。

框架判断是 `trainer.epoch % save_period == 0`，不是 `(epoch+1) % period`。直接把它改成 5 会得到 epoch0/5/10，即 UI 第 1/6/11 轮，而非期望的第 5/10/15 轮。

因此 v1 采用独立发布策略：底层每轮保存，平台在一基轮次 5、10、15…和阶段结束时注册 periodic 快照；best/last 别名始终可查询。**这减少前端 checkpoint 数量，不减少训练目录实际写盘数量。** 本版不自动删除训练文件；磁盘回收需单独保留策略，不能删除已被推理/恢复任务引用的对象。

### 7.2 避免半写文件和不稳定 best

现有 `save_model()` 直接写 `.pt`，不是原子替换。仅靠“文件出现了”“大小几秒没变”不足以保证可安全加载。推荐 worker 适配层在 **`on_model_save`**（同步保存已完成）中复制并发布 checkpoint；现有 callbacks 可接入，但**该发布回调需由平台适配层新增，本文没有假装原脚本已经做了**。

发布顺序：复制到隔离 artifact 临时文件，校验可加载和结构，计算 SHA256，原子 rename 成不可变对象，再事务性写入 registry，最后发送 ready 事件。复制不能用硬链接代替，因为 best/last 后续可能原位覆盖。回调处理快照后，再由独立队列异步上传；后续推理只用 immutable artifact，不使用活跃 `weights/best.pt` 路径。

最简、不改回调的首期方案只能在训练子进程彻底退出后发布 checkpoint；不要在运行中用未同步的文件扫描实现“立即推理”。若要求训练中点击刚保存的 checkpoint，则上述同步快照回调是上线前必做项。

初始化权重、P3 checkpoint、Add-on checkpoint 都需记录架构类型、来源 commit、输入形状、数据版本和原始 epoch。阶段结束 `LovoDetectionTrainer.final_eval()` 会 strip best/last 的 optimizer，并可能将 epoch 改为 -1；历史权重不能仅看 stripped `ckpt['epoch']` 推算轮数。无法可靠恢复历史 epoch 时返回 null，不能猜。

v1 `resume_supported=false` 表示没有开放恢复接口，不代表所有 epoch 文件都缺 optimizer。加载推理与恢复训练是两种能力。

### 7.3 点击某个 checkpoint 推理

```http
POST /api/v1/inference-jobs
Authorization: Bearer <token>
Idempotency-Key: infer-button-20260921-0001
Content-Type: application/json
```

```json
{
  "checkpoint_id": "ckpt_addon_ep10_immutable",
  "media_id": "media_video00009",
  "resource_id": "cpu47",
  "backend": "pytorch",
  "conf": 0.03,
  "nms_iou": 0.45,
  "preview_frames": 3
}
```

后台提交时固定 checkpoint ID+哈希；best 别名以后变化不能改变已提交任务。一个用户选择模型不应替换另一个用户正在用的全局模型。缓存模型时至少按 checkpoint SHA、代码版本、device、精度隔离，独立任务管理显存；v1 参考实现是每个推理进程独立加载。

返回普通异步 Job；完成后 artifacts 包含 `predictions.jsonl`、`summary.json` 和若干预测预览图。JSONL 每行 `frame_index` 从 0 开始，`width/height` 为原图大小，`boxes` 中 `xyxy` 为原图像素坐标，不是归一化值或网络输入坐标，空帧 `boxes=[]`。没有 GT 的素材不能返回 mAP/Precision/Recall。

参考实现可直接加载 P3 或 Add-on 的内部可信 checkpoint：

```bash
.venv/bin/python docs/anti_uav_training_platform/examples/predict_checkpoint.py \
  --checkpoint runs/anti_uav/native_fullpool_14_vs_28_20260916/expanded_28/training_addon/p2/weights/best.pt \
  --source /mnt/andrew/video-labeler/videos/20260805_multirotor_sunny_frontlight_stationary_video00009_90af30d9.mp4 \
  --output runs/platform/infer_EXAMPLE \
  --device cpu --conf 0.03 --nms-iou 0.45 --preview-frames 3
```

该脚本处理完整视频，但只保存指定数量的预览图；不宣称已实现整段带框视频编码。后续可在预测 JSONL 上单独渲染 H.264 视频，编码耗时和解码/推理耗时分开。

**固定输入注意：**当前 `ultralytics/engine/predictor.py` 的 `pre_transform` 使用 `auto=same_shapes and self.model.pt`；不能假定加 `rect=False` 就关闭 auto padding。参考 worker 显式使用 `LetterBox((544,960), auto=False)` 并断言张量形状。灰度图片最终按 3 通道 BGR/RGB 约定输入，不直接当 1 通道网络。

本仓库 `Model.predict(predictor=...)` 接收已经创建的 predictor 实例，而不是类；参考示例已按此版本构造实例并传入完整 overrides，不照搬其他 Ultralytics 版本的自定义 predictor 写法。

本接口 `.pt` 模型推理使用服务器 PyTorch，这是训练平台预览，不是板端部署。RKNN INT8 要另行导出、用合规训练校准集量化并在板端验证；不能将 .pt 换后缀当 .rknn，也不能把服务器 FPS 当 RK3588 FPS。Dist/GMC 跟踪可作为后续独立管线 profile，不混入 detector-only 的精度与耗时。

## 8. 独立评测与不泄漏测试

训练每轮已执行 validation；`evaluation-jobs` 用于明确选择 checkpoint 后的独立验证/测试，不重训、不更新 best。平台把 `split` 映射到经过登记的 YAML 的 **val 清单**；Ultralytics 不会因为接口里写 test 就自动换测试集。

worker 的核心调用为：

```python
import math
from ultralytics import YOLO
from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator

def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def evaluate(checkpoint_path, registered_split_yaml, device, output_parent, job_name):
    result = YOLO(checkpoint_path).val(
        data=registered_split_yaml, validator=FixedShapeGrayValidator,
        imgsz=[544, 960], rect=False, device=device, batch=32, workers=4,
        conf=.001, iou=.45, max_det=100, half=False, plots=False,
        project=output_parent, name=job_name, exist_ok=False,
    )
    return finite_json(result.gray_selection)
```

记录输入模型哈希、split manifest/hash、实际样本数、协议与 backend。禁止把 Video00004 的错误帧挖回训练后仍声称同一测试无泄漏；频繁参考它调方案也会使它成为开发集，需后续新独立测试。初始化模型来源同样要审计。

不要直接把旧 `evaluate_real_gray_yolo_lovo_fold.py` 作为通用多目标平台评测器：其中 `load_gt()` 只读取第一行标签。当前平台统一用上述 FixedShapeGrayValidator 路径，避免多框 GT 被漏算。

## 9. 错误、权限和运维约束

错误统一为 `{"error":{"code":"...","message":"...","request_id":"...","details":{}}}`。

| HTTP | code 示例 | 处理 |
| --- | --- | --- |
| 401/403 | UNAUTHORIZED / FORBIDDEN | 未登录或无该任务/数据/模型权限 |
| 404 | NOT_FOUND | 对象不存在或不可见，不泄露真实路径 |
| 409 | DATASET_NOT_READY / CHECKPOINT_NOT_READY / IDEMPOTENCY_CONFLICT | 数据未准备、权重未发布或幂等冲突 |
| 413 | MEDIA_TOO_LARGE | 限制媒体体积、分辨率、时长和解码工作量 |
| 422 | UNSUPPORTED_CONFIG / DATASET_LEAKAGE / INVALID_LABEL / INCOMPATIBLE_CHECKPOINT | 不支持参数或预检不通过，不静默改配置 |
| 429 | QUOTA_EXCEEDED | 用户队列/资源配额不足 |
| 500 | INTERNAL_ERROR | 仅给脱敏摘要；详细日志存授权任务日志 |

OOM、训练异常、冻结校验失败、产物损坏等发生在 202 之后，通过 Job 的 `state=failed`、`error_code` 和日志返回；不能对已成功响应的 HTTP 请求再回传一个错误状态。

服务端必须做到：ID→白名单真实路径解析，拒绝路径穿越和越界 symlink；用户不能传 shell、任意 YAML/Python、任意服务端路径或下载 URL。上传只收限额内媒体，独立 worker 解码。PT 含 pickle，仅加载本系统产生或管理员审核的可信权重；“在子进程里 torch.load”不等于安全沙箱。

建议 Web/worker 用专用低权限服务账号，只读授权数据、只写任务目录；不要把当前 root SSH 账户或密码交给浏览器。令牌不写日志或示例命令文件。artifact 下载再次校验租户/项目权限，配额和保留策略包含已发布 checkpoint 及其推理引用关系。

代码更新按现有约定：本地改动和测试 → Git commit/push → 47 git pull → 固定本次 worker commit。不能一边训练一边更新它正在 import 的代码目录；用固定 checkout/容器或至少保证运行期间版本不变。

## 10. 上线前验收清单

1. 双击启动或请求重试只创建一个训练任务；排队和 GPU 租约可追踪，未擅自共享/杀其他进程。
2. 数据集及初始化模型泄漏审计通过，不确定/未审核帧排除，空负标签不等于缺失标签。
3. P3 完成后自动初始化 P2，P2 仅训练新分支，最终 frozen_checks 对 best/last 均 bit_exact。
4. loss 和 validation 逐轮可见；P3/P2 分段、epoch 一基、NaN/null 和固定 conf 指标含义正确。
5. 发布每 5 轮 checkpoint 时 UI 展示第 5/10/15 轮而非 1/6/11；读取不与写入竞争。
6. 点击两个不同 checkpoint 得到两个固定 SHA 的推理任务；输出原图坐标、预览图及空帧记录正确。
7. P3 和 Add-on 权重均在固定 544×960 下通过图像 smoke test；视频测试验证全帧顺序和输出数量。
8. 测试指标不参与 best 选择；服务器 PT 评测、RKNN 板端精度/FPS、Tracker 指标分别展示。
9. 取消、OOM、worker 重启/失联和损坏产物不会遗留僵尸 running 或泄漏 GPU 租约；不伪装可 resume。
10. 未实现的能力在 `/capabilities` 中关闭。本文契约、离线示例验证，不等于 HTTP 服务已完成联调。

## 11. 本次交付验证记录

日期 2026-09-21；参考 worker 代码 `6ea9fdc`。仅执行只读查询和隔离 CPU 推理，没有启动训练、抢占 GPU 或修改已有模型。

| 检查 | 结果 |
| --- | --- |
| OpenAPI 3.0.3 校验（openapi-spec-validator 0.7.2） | 通过，18 个 HTTP 操作 |
| 文档 4 个 JSON 请求/响应与 schema 对照 | 通过 |
| 曲线读取单元测试 | 6 项通过，含空文件、空格、NaN/Infinity、半行、重复轮次和阶段游标 |
| 47 上真实 28 视频 results.csv | 正确读取 P3 15 + Add-on 15，共 30 个曲线点 |
| P3 best CPU 单图加载 | 通过，输入断言 `[1,3,544,960]`，输出和预览正常 |
| Add-on best CPU 单图加载 | 通过，同上 |
| Add-on `epoch9.pt`（UI 第 10 轮）CPU 视频加载 | 通过，3 帧按 0/1/2 顺序输出，4 个检测框，3 张预览 |

短视频取自 Video00009 的 9.7 秒附近并重编码，仅验证接口调用、checkpoint 加载和坐标输出，不用于精度/FPS比较。测试目录：

```text
/mnt/chenziye/codes/ultralytics_yolov8/runs/platform_handoff_smoke_20260921/p3_image
/mnt/chenziye/codes/ultralytics_yolov8/runs/platform_handoff_smoke_20260921/addon_image_v2
/mnt/chenziye/codes/ultralytics_yolov8/runs/platform_handoff_smoke_20260921/addon_epoch10_video
```

**尚未验证/尚未实现：** HTTP 服务、数据库/队列、鉴权、上传限额、GPU 租约、运行中 checkpoint 原子发布回调、端到端取消恢复、整段可视化视频编码。这些是平台团队按本契约接入的工作，不属于本次离线示例的测试结论。
