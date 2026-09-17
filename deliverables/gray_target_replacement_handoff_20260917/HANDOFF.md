# 灰度无人机靶机贴图替换数据增强：交接与操作说明

核验日期：2026-09-17。适用服务器：`47.107.185.207`。操作对象：已审核、已划入训练集的灰度无人机视频帧。

## 1. 交接结论

这套工具可从原训练视频及标注中选择正样本，用透明无人机贴图替换目标，输出**原分辨率、无绘制标记的灰度 PNG，以及重新计算的 YOLO bbox**。原训练图片、原标签、原视频和正在使用的训练清单不会被修改。

当前推荐入口是 `build_gray_replacement_batch.py`，使用同一训练视频中的已审核邻帧恢复背景，而不是挖掉矩形区域后填充独立噪声。背景配准、目标轮廓或对比度不可靠时直接跳过，不为凑数量降低质量门限。

已完成的候选批次有 **856 张图及对应标签**，来自 219 个不同原始帧、17 个训练视频，使用 53 个贴图 ID。856 张不等于 856 个独立实拍场景。该批次**尚未加入 28 视频模型的训练**，不能将现有模型提升归因于它。

本工具是离线检测训练增强，不是整段视频的时序一致目标替换，也不是三维机型、姿态和相机光学仿真。不能承诺背景完美融合或使用后一定提升精度。

## 2. 服务器、代码和数据位置

经授权的操作人员使用 `ssh root@47.107.185.207` 登录；凭据通过团队已有安全渠道获取，本文不包含密码或私钥。

| 内容 | 服务器绝对路径 |
| --- | --- |
| 代码仓库 | `/mnt/chenziye/codes/ultralytics_yolov8` |
| 批量生成入口 | `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/build_gray_replacement_batch.py` |
| Excel 素材提取、单帧合成和预览 | `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/synthesize_gray_drone_replacements.py` |
| 邻帧配准、前景轮廓和背景修复 | `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/gray_temporal_background.py` |
| 批次验收入口 | `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/verify_gray_replacement_batch.py` |
| 独立 Python 环境 | `/mnt/chenziye/codes/ultralytics_yolov8/.venv_gray_synthesis/bin/python` |
| 原始机型汇总 Excel | `/mnt/andrew/anti_uav_model_refinement/data/drone_asset_sources_20260917/drone_model_catalog_original.xlsx` |
| 当前素材索引 | `/mnt/andrew/anti_uav_model_refinement/data/drone_asset_catalog_20260916/catalog.json` |
| 已提取透明贴图 | `/mnt/andrew/anti_uav_model_refinement/data/drone_asset_catalog_20260916/cutouts/` |
| 当前 28 视频训练数据快照 | `/mnt/andrew/anti_uav_model_refinement/data/real_gray_native_fullpool_28videos_20260916/` |
| 已审核任务与标注根目录 | `/mnt/andrew/anti_uav_model_refinement/data/approved_tasks/` |
| 原视频根目录 | `/mnt/andrew/video-labeler/videos/` |
| 已完成 856 张候选批次 | `/mnt/andrew/anti_uav_model_refinement/data/real_gray_replacement_candidates_20260916/` |
| 该批次原生成日志 | `/mnt/andrew/anti_uav_model_refinement/data/real_gray_replacement_candidates_20260916.log` |
| 本交接文档 | `/mnt/chenziye/codes/ultralytics_yolov8/deliverables/gray_target_replacement_handoff_20260917/HANDOFF.md` |

### 原始 Excel 的来源

服务器文件是原文件 `无人机机型汇总(1).xlsx` 的字节一致副本，仅改用英文文件名便于终端操作。原文件与上传副本 SHA256 均应为：

```text
7f7576f1c0ec55dbb7fb047fb3bafe3cf3c13ce972a8992aa02afbae1e415baa
```

`catalog.json` 仍保留最初的本地 `xlsx` 来源路径，这是历史溯源字段，**不表示服务器需要访问 Mac**；现有批量生成只读取索引及 `cutouts/`。为保留已完成批次的哈希链，不要修改旧索引。需要重建时使用上表中的服务器 Excel 路径，并输出到新目录。

## 3. 素材范围与筛选规则

当前索引包含 334 条嵌入图像记录，其中 55 条通过原生 alpha 透明度筛选；279 条是不透明照片，需要额外抠图和人工审核，当前不会自动投入合成。55 条中，ID `24` 和 `25` 含遥控器，批量入口默认排除，剩余 **53 个可用 ID**。

透明不代表机型、方向或授权必然正确。使用前仍需检查是否为完整无人机、是否带遥控器/文字/底座、姿态是否适合远距离观察，以及素材版权是否允许计划中的训练和交付。

提取脚本读取 Excel 绘图锚点与对应行，而不是按 `image1/image2` 的文件序号猜配机型。表格约定：A 列为唯一 ID，B 列厂商，C 列类别，D 列机型，E 列实物尺寸；图片必须嵌入工作表并锚定相应数据行。默认使用第一张工作表，可用 `catalog --sheet` 指定其他表。

## 4. 合成效果与 bbox 的含义

| 项目 | 实际处理方式 |
| --- | --- |
| 背景 | 同视频已审核邻帧配准、亮度匹配，再按前景轮廓柔化修复；编辑掩码外解码像素保持一致 |
| 尺度 | 保持原目标放置中心和几何长边；保留新贴图自身宽高比，面积和短边可以变化 |
| 灰度 | 输出单通道灰度 PNG，按原目标相对背景的明暗和对比度匹配，不保留产品宣传照原色 |
| 模糊 | 默认原图坐标系下 `sigma=0.65 px` 的近似模糊，不是实测镜头 PSF |
| bbox | 根据缩放、模糊后 alpha 大于 0.15 的区域重算外接框，再转为归一化 YOLO 标签 |
| 类别 | 当前单类无人机检测，标签 class 为 `0`；Excel 机型 ID 不是检测类别 ID |
| 预览 | 原图原框与替换图新框并排展示；放大后绘制细四角框，不加十字 |

YOLO 标签每行格式为 `0 cx cy width height`，数值归一化到 `[0,1]`。**不要把原 bbox 原样复制给新贴图**，也不要为了维持原 bbox 面积而强行拉伸机型。

当前合成仅支持原图中短边至少 3 px、长边至多 160 px、单个目标且周围有足够背景上下文的正样本。复杂纹理、贴边、对比度不足、多个目标或不可靠邻帧可能被跳过。这是合成方法的适用范围，**不是检测面积过滤，也不会删除大目标原始训练数据**；80% 画面占比等大目标需要另行设计增强方案。

## 5. 直接复用现有素材：推荐操作

以下命令在服务器的 Bash 中执行。使用绝对路径；每次指定全新输出目录，不要预先创建该输出目录。

### 5.1 设置路径并检查环境

```bash
set -euo pipefail
REPO=/mnt/chenziye/codes/ultralytics_yolov8
DATA=/mnt/andrew/anti_uav_model_refinement/data
PY="$REPO/.venv_gray_synthesis/bin/python"
DATASET="$DATA/real_gray_native_fullpool_28videos_20260916"
CATALOG="$DATA/drone_asset_catalog_20260916/catalog.json"
XLSX="$DATA/drone_asset_sources_20260917/drone_model_catalog_original.xlsx"
cd "$REPO"
test -x "$PY"
test -f "$DATASET/train_hardneg.txt"
test -f "$CATALOG"
test -f "$XLSX"
"$PY" -c 'import cv2,numpy,PIL,openpyxl; print(cv2.__version__,numpy.__version__,PIL.__version__,openpyxl.__version__)'
"$PY" scripts/anti_uav/build_gray_replacement_batch.py --help
sha256sum "$XLSX"
```

已验证环境：Python 3.10.12、OpenCV 4.10.0、NumPy 1.26.4、Pillow 11.3.0、openpyxl 3.1.5。无需 GPU、CUDA、RKNN 或 YOLO 权重。不要为这项任务升级正在训练的 `.venv`。

代码已按“本地 git push，服务器 git pull”的方式同步。接手时先查看 `git status --short` 和 `git log -3 --oneline`；不要 reset/覆盖服务器上的其他工作。需要更新且网络、权限允许时使用 `git pull --ff-only`。

### 5.2 小批量试运行

```bash
OUT="$DATA/real_gray_replacement_smoke_$(date +%Y%m%d_%H%M%S)"
test ! -e "$OUT"
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nice -n 15 "$PY" scripts/anti_uav/build_gray_replacement_batch.py \
  --dataset "$DATASET" --catalog "$CATALOG" --output "$OUT" \
  --per-video 1 --variants 1 --previews 4 --seed 20260916 \
  --exclude-assets 24,25
"$PY" scripts/anti_uav/verify_gray_replacement_batch.py --batch "$OUT" \
  --report "$OUT/verification_handoff.json"
```

`--per-video 1` 是每个合格视频最多选 1 个原始帧，并非整个批次只处理 1 帧。即便只选少量帧，仍需读取并校验相关视频。试运行可能因背景质量门限产生少于预期的输出；零合格输出不能当作可用训练批次。

### 5.3 正式批量生成

下面复用已完成批次的数量参数：每个可用视频最多 20 个原帧，每帧最多 4 个不同素材变体。53 个素材不代表每帧都合成 53 次。

```bash
OUT="$DATA/real_gray_replacement_candidates_$(date +%Y%m%d_%H%M%S)"
test ! -e "$OUT"
nohup env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nice -n 15 "$PY" scripts/anti_uav/build_gray_replacement_batch.py \
  --dataset "$DATASET" --catalog "$CATALOG" --output "$OUT" \
  --per-video 20 --variants 4 --previews 16 --seed 20260916 \
  --exclude-assets 24,25 >"$OUT.log" 2>&1 < /dev/null &
echo "$!" >"$OUT.pid"
printf '输出目录：%s\n日志：%s.log\n' "$OUT" "$OUT"
```

查看 `tail -n 30 "$OUT.log"` 和 `"$PY" -m json.tool "$OUT/status.json"`。`stage=complete` 才表示完成，`failed` 时先读 `error` 和日志，不能只按已出现的图片数判断成功。断开 SSH 后需重新设置 `OUT` 为本次实际目录。

先前批次限制在 CPU 126、127 上运行约 19.6 分钟，仅作耗时参考。47 服务器本次核验有 128 个 CPU；如需限制 CPU 可在 `nice -n 15` 后加 `taskset -c 126,127`。其他机器不可照搬核心编号。不要通过同时启动很多批次挤占训练 I/O。

参数 `--per-video` 增加不同原始帧覆盖；`--variants` 增加同帧的机型变体。应优先增加多样场景，而不是大量复制同一背景。最多变体数不能超过可用素材数。修改随机种子会改变选帧与素材分配，复现实验时应固定种子及输入版本。

## 6. 从 Excel 重建素材库

只有换表、补图、复核机型或希望拿到完整原始图目录时才需要重建。旧服务器素材库主要包含 `catalog.json` 和 `cutouts/`，没有完整的 `raw/`；这不影响当前批量入口。

```bash
NEW_CATALOG="$DATA/drone_asset_catalog_rebuilt_$(date +%Y%m%d_%H%M%S)"
"$PY" scripts/anti_uav/synthesize_gray_drone_replacements.py catalog \
  --xlsx "$XLSX" --output "$NEW_CATALOG"
```

新目录包含 `catalog.json`、`raw/`、`cutouts/` 和素材预览。先人工检查索引和透明图，再把第 5 节的 `CATALOG` 改为新目录内 `catalog.json`。原索引应保留不动。

对 279 条不透明照片，现有脚本只做“需要分割”的标记，不会自动高质量抠图。可另行制作 RGBA 素材，但必须复核轮廓、旋翼、阴影和残留背景，并重新登记新索引与 `cutout_sha256`；不要只替换旧 PNG 而不更新新版本索引。`24,25` 的排除 ID 针对当前表，如新表改了 ID，应重新审查。

## 7. 输出结构与验收

| 文件/目录 | 用途 |
| --- | --- |
| `images/*.png` | 可供审核后加入检测训练的无框灰度图 |
| `labels/*.txt` | 与图片同名、更新后的 YOLO 标签 |
| `masks/*.png` | 编辑区域审计掩码，不是检测或分割训练标签 |
| `previews/*.png` | 原图/替换图/新旧 bbox 对比，仅供检查，不能加入训练 |
| `preview_contact_sheet.jpg` | 整批预览总览 |
| `train_synthetic.txt` | 成功替换图片的绝对路径列表，仅含正样本 |
| `manifest.json` | 来源、素材、框变化、邻帧信息、失败原因与所有样本记录 |
| `protocol.json` | 参数、目录、输入哈希和整视频留出规则 |
| `accepted.jsonl` | 逐条成功记录，运行中可观察进度，不替代最终完整状态 |
| `provenance/*_registry.json` | 原视频、已审核清单、COCO 文件的对应关系 |
| `summary.json` / `status.json` | 最终统计 / 当前执行状态 |

### 自动验收

```bash
"$PY" scripts/anti_uav/verify_gray_replacement_batch.py --batch "$OUT" \
  --report "$OUT/verification_handoff.json"
```

报告不得覆盖已有同名文件，复查时换一个报告名，或省略 `--report` 只打印。任一检查失败会以非零退出码停止，不会改写原图、标签或训练清单。

验收检查完成状态、源图/标签/贴图哈希、训练清单成员资格、留出视频哈希排除、输出文件完整性、PNG 灰度模式、掩码外像素不变、更新标签与记录中的新 bbox 一致、无重复输出，以及原训练配置未被修改。这是完整性检查，不能替代视觉质量检查，也不能独立证明轮廓标注准确。

### 人工验收

逐类查看小、中、大目标、不同背景和不同素材，不能只看总览。重点检查原目标是否擦干净，有无矩形色块、纹理断层、曝光台阶、双影、白边、旋翼被截断、目标过锐或过糊，以及新框是否包住可见目标。对极小目标，必须兼看原尺寸和放大图。

不要覆盖已审计批次来删除坏样本。单独维护审批后的图片清单及剔除原因，训练只读取审批清单；这样可以保留原始 manifest 与完整追溯关系。已有 856 张批次全量完整性检查通过，但仍属于待人工筛选的候选数据。

## 8. 数据来源、留出隔离与新增视频

批量入口不是“随便给一个视频文件夹就能处理”。当前 `--dataset` 必须包含本项目训练快照格式：`manifest.json`、`train_hardneg.txt`、`val_monitor.txt`、`train_hardneg_gray_monitor.yaml`，以及可访问的原图与标签。

脚本从快照中解析 `approved_data` 对应历史 manifest 的 `approved_tasks`，合并当前 `appended_videos`；按 `train_video_hashes` 做允许列表。每个来源记录需有 `task`、`video`、`sha256`、`manifest_sha256`、`image_directory`、`label_directory`。审核任务内需有 `manifest.json` 和 `coco/annotations.json`，原视频须与审核哈希相符。

只使用 `includedFrameIndices` 中且不在 `excludedUncertainFrameIndices` / `excludedUnreviewedFrameIndices` 的帧，同时要求该图已在当前训练清单。邻帧也必须经过审核。frame index 为 **0-based**，例如 `000000246.jpg` 对应视频第 246 帧索引，不是第 246 秒。

当前整视频排除：

| 用途 | 视频 | SHA256 |
| --- | --- | --- |
| 最终测试 | `Video00004` | `d70d6b1f638e1d2c4bbcd081f541b9f2cd273063b40032581ce56004ef69becc` |
| 选权重验证 | `20260805_multirotor_sunny_frontlight_stationary_video00009_90af30d9.mp4` | `2629efe7da36dfd079985344653a7a4ecaa25f651954e5258a53c50e960d6b86` |

两者均不得作为合成源帧或背景邻帧。不能先使用测试视频错误帧造数据，再把该视频称为无泄漏测试。整文件哈希不能发现所有重编码、重叠剪辑或同场景重复，新增来源还需人工排查。

28 视频快照中，本入口识别到 22 个有对应审核登记的训练视频；旧视频中缺少匹配登记的部分不会猜配标注。上一批 18 个视频有选中原帧，最终 17 个视频产生合格替换，不表示其余真实训练视频被移除。

新增视频的顺序是：完成审核交付 → 按整视频划分训练/验证/测试 → 建立新的训练快照并抽取审核帧及标签 → 检查快照来源登记与原视频一致 → 对新快照运行第 5 节。相关数据接入脚本为 `scripts/anti_uav/append_approved_gray_native.py`；先运行 `--help` 核对 snapshot、video-root、old-root 等参数，不要手工伪造旧 manifest 或修改其哈希。新上传任务不会自动加入旧的 28 视频快照。

## 9. 加入训练的方法与限制

`train_synthetic.txt` 是候选正样本清单，不是独立、正负均衡的训练集，也没有自动接入任何正在运行的训练任务。

审核通过后，创建独立实验数据配置和训练清单，保留真实数据、旧清单顺序与重复次数、困难负样本和原验证/测试划分。不要使用 `sort -u` 去重原训练清单，原来的重复次数可能承担小目标加权作用。加入合成正样本后要重新检查每 epoch 的正负抽样比例，不能把原先的 15% 负样本占比当作自动保持。

现有 28 视频训练使用标签感知抽样器，从负样本池逐 epoch 抽取负样本；**仅在文件末尾追加路径并不能保证训练器实际用到它们**。另建实验时需同步检查数据 YAML、抽样器读取的候选清单/元数据及每 epoch 正负数量，不要直接修改运行中的快照。

建议先用少量人工通过的合成样本做独立对照：真实数据基线 vs 相同设置下加入合成数据，保持 Frozen-P3 + Add-on P2 的两阶段训练、阈值和评测口径一致。检查 Video00009 验证指标及 Video00004 最终测试的 Recall、FP、Precision、mAP 和小目标分层结果。多次按 Video00004 调参会使它失去独立测试意义。

合成素材增加的是外形多样性，并不增加独立真实场景；不保证提高小目标召回，也不能靠这批受限尺寸的替换数据解决所有大目标问题。

## 10. 常见问题

| 现象 | 处理 |
| --- | --- |
| `Refusing to overwrite output` | 使用全新输出目录；不要删除已有批次来“重试”，脚本没有断点续跑保证 |
| `hash mismatch` / `COCO changed` | 检查文件版本、审核任务是否重新交付；不要绕过校验，应新建一致的数据快照 |
| 没有原视频或不能 seek | 检查 registry 指向的 MP4、权限和解码，只有 JPG+TXT 不足以完成邻帧修复 |
| 输出少于选帧数乘变体数 | 查看 manifest 的 skipped/rejected_reasons；失败、重复像素不会导出新训练图 |
| 目标纹理复杂、配准失败、邻帧同位置仍有飞机 | 属于安全跳过；不要切换为独立噪声/矩形补丁只为增加数量 |
| 大目标/多目标未生成 | 当前适用范围限制，原始训练样本仍保留；不要把限制改成删原数据 |
| 想得到连续合成视频 | 当前帧级方法不能保证时序身份/姿态/光照一致，需另外实现视频一致性流程 |
| 想使用所有 334 种素材 | 对不透明图补充分割及审核，再建新素材索引；当前不是全部可直接使用 |

不要在正式批量中使用 `synthesize --background-mode legacy-plane`。它保留用于旧方法对照，可能重新引入用户观察到的矩形/纹理不连续问题。旧 `synthesize` 原型还会保存未替换的正/负样本，不能把其输出数量当作新增成功替换数；正式批量优先使用本文的 `build_gray_replacement_batch.py`。

## 11. 验证与环境迁移

回归测试需隔离运行，避免仓库全局测试配置导入训练依赖。以下命令只复制测试脚本到临时目录，不会改训练数据：

```bash
QA=$(mktemp -d /tmp/gray-synthesis-tests.XXXXXX)
cp "$REPO/tests/test_gray_drone_replacements.py" "$REPO/tests/test_gray_replacement_batch.py" "$QA/"
PYTHONPATH="$REPO" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  "$PY" -m pytest -q -c /dev/null --confcutdir="$QA" "$QA"
```

如独立环境缺失，在新环境中按已验证版本安装，不动训练环境：

```bash
python3.10 -m venv "$REPO/.venv_gray_synthesis_new"
"$REPO/.venv_gray_synthesis_new/bin/python" -m pip install \
  numpy==1.26.4 opencv-python-headless==4.10.0.84 Pillow==11.3.0 openpyxl==3.1.5 pytest==9.1.1
```

依赖范围文件位于 `scripts/anti_uav/requirements-gray-synthesis.txt`；固定版本更适合复现。迁移到其他服务器需要同时转移 Excel/素材、原视频、审核标注、训练图片及清单，并重新解析绝对路径，不能只复制脚本和 856 张图片就期待继续生成。移动输出目录后 `train_synthetic.txt` 的绝对路径也要在新版本中更新并重新验收。

完成交接的最低要求：能找到 Excel 和素材；小批次完成；验收脚本通过；人工检查新旧 bbox 与背景；确认原训练清单未变；另建训练实验后再评估收益。

## 12. 本次交接实测记录

2026-09-17 在 47 服务器完成以下检查，而非仅整理历史说明：

| 检查 | 结果 |
| --- | --- |
| 原始 Excel 上传与 SHA256 比对 | 与本地原文件一致，156,313,158 字节 |
| 合成与批量筛选回归测试 | 20 项通过 |
| 已有候选批次重新全量验收 | 856/856 通过，约 34 秒 |
| 第 5.2 节参数的小批次实测 | 选中 18 个原帧，7 个视频各成功 1 张，11 个因背景纹理门限跳过，生成约 29 秒 |
| 小批次重新验收 | 7/7 通过 |
| 原训练配置与输入保护 | 未修改；两个合成批次均未自动加入训练 |

小批次服务器目录：

```text
/mnt/andrew/anti_uav_model_refinement/data/real_gray_replacement_handoff_smoke_20260917
```

完整性报告原件分别位于两个批次内的 `verification_handoff_20260917.json` 和 `verification_handoff.json`；交接目录也提供副本 `verification_existing_856.json`、`verification_smoke_7.json`。本次验收脚本及操作说明的首个代码提交为 `7dd52e2`，后续文档补充可从仓库历史查阅。

小批次只用于证明操作链可执行，不能据此判断全部尺寸、全部机型或全部背景的合成质量。正式候选批次仍需按第 7 节人工筛选和第 9 节独立训练对照。
