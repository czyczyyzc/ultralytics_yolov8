# Video00004 + Video00009 贴图版配对测试集

生成日期：2026-09-22。仅用于离线 detector 外观对比，未加入训练，未修改部署模型。

## 数据范围

沿用上一轮五组纯 P3 对比的同一批 3,780 帧，不另抽容易帧，也不删除替换失败的正样本。Video00004 为既有测试集 2,359 帧；Video00009 为参与选权重的 1,421 帧验证子集，并非 14,201 帧原始视频的全量导出。64 张合成放大验证图不在本数据集中。

| 来源 | 总帧数 | 正样本 | 成功替换 | 正样本保留原图 | 负样本保持不变 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Video00004 | 2,359 | 448 | 416 | 32 | 1,911 |
| Video00009 验证子集 | 1,421 | 797 | 565 | 232 | 624 |
| 合计 | 3,780 | 1,245 | 981 | 264 | 2,535 |

候选贴图共 328 张，全部至少成功使用一次。正样本成功替换率为 78.80%。每个正样本计划替换一次，不是每帧生成 328 个版本。固定随机种子为 20260922，分配在查看 detector 预测或合成是否成功之前完成；不会通过更换贴图反复尝试来挑容易样本。

## 服务器路径

完整数据位于 47 服务器：

`/mnt/andrew/anti_uav_model_refinement/external_eval/synthetic_gray_pair_20260922/`

主要入口相对此目录：

- `original_combined.yaml`：原版配对集，3,780 帧。
- `synthetic_combined.yaml`：贴图版配对集，3,780 帧，含明确记录的 264 个原图回退正样本。
- `original_Video00004.yaml`、`synthetic_Video00004.yaml`：分别评测 Video00004。
- `original_Video00009.yaml`、`synthetic_Video00009.yaml`：分别评测 Video00009 验证子集。
- `original/images/`、`original/labels/`：原图的独立副本和原始 YOLO 框。
- `synthetic/images/`、`synthetic/labels/`：合成图和更新的 YOLO 框。
- `original_replaced_positive_pairs.txt`、`synthetic_replaced_positive_pairs.txt`：同一批 981 个成功替换正样本的一一配对清单，仅用于补充分析，不包含负样本，不应据此宣称整体误检率。
- `plan.json`：随机分配、资产路径/哈希、视频与原始标注身份、受保护输入哈希。
- `manifest.json`：逐帧来源、分配机型 ID、替换状态、失败原因、原框/新框、合成参数及输出哈希。
- `summary.json`、`verification.json`、`status.json`：生成汇总、全量校验和完成状态。
- `previews/`：24 张原图/合成图对比预览；输入模型的图片不带框，只有预览画有原框与新框。

合成版中未改变的帧使用指向本数据集 `original/` 副本的相对软链接，不依赖训练目录。转移完整数据时应一起保留 `original/`、`synthetic/` 和软链接结构；各 TXT/YAML 中使用服务器绝对路径，迁移到其他根目录后必须重定位清单，不能直接照搬旧路径。

## 合成规则

只处理正样本中的原目标。使用同一原始视频中审核标注有效的邻帧做相机运动配准与背景修复，避开邻帧目标和边界。保留背景纹理，按原目标对比度生成灰度贴图；几何长边与中心沿用原目标，不人为放大。

YOLO bbox 按渲染后透明度轮廓重新计算。不同机型的纵横比不同，所以新框宽高不必等于原框；可见框也可能因模糊边缘和像素取整发生约像素级变化。类别始终是 `0: drone`，机型 ID 只是来源元数据，不会变成 328 个训练类别。

981 个替换结果为无损 PNG。合成函数保证编辑 mask 外的像素不变；其他帧的图像文件字节及标签字节保持不变。两套数据的帧数、正负样本数和每帧目标数量相同。

264 个回退正样本包括：194 个没有可靠邻帧背景，32 个背景纹理超过当前保守阈值，25 个不在当前修复器的尺寸适用范围，12 个靠近画面边缘缺少上下文，1 个贴图对比度缩放不可靠。这些样本没有被删除，也没有把它们的框过滤掉。没有启用独立噪声填充背景的旧方法来强行提高成功率。

## 评测限制

这是“同帧背景下换目标外观”的合成对比，不是真实新机型拍摄测试。328 张候选贴图已经用于此前的全部贴图训练实验，因此不能称为未见贴图/未见机型泛化测试；这会影响与未用全部贴图训练的模型之间的结果解释。

Video00009 已参与模型选择，所以合并结果也不是独立盲测。机型按帧随机变化，且 Video00009 只取标注子集，本版本不适合作为连续跟踪性能测试视频。背景修复和模糊是近似模拟，抽查不等于逐张人工审核，不能保证所有合成图完全无痕。

生成阶段只完成数据生成与校验，生成记录中的 `model_evaluation_performed=false` 描述的是当时的检查范围。2026-09-22 随后已完成五组纯 P3 的固定权重原图/合成图精度评测，报告位于仓库 `deliverables/p3_synthetic_pair_evaluation_20260922/README.md`，服务器正式运行目录为 `/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/p3_synthetic_pair_evaluation_verified_20260922/`。评测保持权重、输入、conf/NMS/匹配阈值不变，另列成功替换子集；不能丢弃失败帧后直接与原版全量指标比较。

## 代码与复现

服务器脚本：

`/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/build_gray_pair_synthetic_test.py`

生成提交：`ef7690c`。已执行本地 Git push 和服务器 fast-forward pull；本次未改动原有训练集拦截测试视频的保护。

```bash
cd /mnt/chenziye/codes/ultralytics_yolov8
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/anti_uav/build_gray_pair_synthetic_test.py \
  --reference runs/anti_uav/p3_pair_evaluation_20260922 \
  --cache /mnt/andrew/anti_uav_model_refinement/data/real_gray_online_replacement_40videos_assets328_20260922 \
  --output /mnt/andrew/anti_uav_model_refinement/external_eval/NEW_SYNTHETIC_GRAY_PAIR \
  --workers 16 --seed 20260922
```

输出目录必须不存在，不会覆盖旧结果。可加 `--smoke-per-video 8` 先跑小批次，但不能把该小批次当作全量结果。YAML 的 `train` 明确设为 null，只提供 `val` 与 `test`，不要把本测试集拿去训练。

验证：本地相关测试 38 项通过；服务器可用依赖范围内 21 项通过。服务器完整测试套件中的 Excel 读取测试因缺少 openpyxl 无法收集，未为此修改训练虚拟环境。正式生成不需要 Excel 读取，使用已有透明贴图库。全量校验覆盖全部 3,780 帧的图片完整性、哈希、YOLO 标签、负样本与回退样本的字节一致性、训练输入不变；两套数据均通过 YOLO 加载，均为 1,245 个 GT，输入验证为 544x960。

本地本目录只下载说明、汇总、校验记录和 24 张预览，完整图像数据保留在上述 47 服务器路径。
