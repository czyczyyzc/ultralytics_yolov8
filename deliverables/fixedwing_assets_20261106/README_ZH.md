# 固定翼航模贴图筛选与合并

## 数量

2026-11-06，输入目录 `/Users/czyczyyzc/Downloads/固定翼航模图片/` 共38张、8款机型。所有输入均不具备有效原生透明通道，包括一张RGBA但alpha全255的PNG。

- 17张完整机体或待确认照片进入分割，14张通过可视检查。
- 排除24张：21张为零件、特写、不完整机体或装配照；另3张实际分割后确认有分离尾翼、杯子、胶水遮挡或裁切问题。
- 旧图库328条，对PNG SHA256完全去重后317张。
- 新增14张与旧库没有原图文件级重复，新增PNG也没有重复。
- 合并兼容版342条；完全去重版331张。多视角照片仍是不同素材，不等同于新增不同机型。

| 本次机型 | 新增素材 |
| --- | ---: |
| ATOMRC Dolphin | 3 |
| Freewing Avanti S 80mm EDF | 2 |
| HEEWING T1 Ranger | 1 |
| Lazy Bee | 1 |
| Reptile S800 V2 | 1 |
| SonicModell AR Wing Pro White Falcon | 1 |
| Volantex Ranger 1600 V757-7 | 1 |
| ZOHD Drift Dark Breeze | 4 |

## 服务器路径

服务器：`root@47.107.185.207`。

```text
新合并图库根目录：
/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_catalog_20261106/

342条兼容旧ID的清单：
/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_catalog_20261106/catalog.json

331张PNG完全去重清单及旧ID别名：
/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_catalog_20261106/catalog_unique.json

仅本次14张新增素材的清单：
/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_catalog_20261106/new_assets.json

透明PNG：
/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_catalog_20261106/cutouts/

兼容342条素材的69视频背景缓存视图：
/mnt/andrew/anti_uav_model_refinement/data/real_gray_background_approved69_assets342_20261106/

新增14张独立打包：
/mnt/andrew/anti_uav_model_refinement/data/fixedwing_new14_rgba_20261106.tar.gz

输入原图（38张，未修改）：
/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_import_20261106_input/

17张分割候选及原图/深色背景预览（含未通过项，不要整库用于训练）：
/mnt/andrew/anti_uav_model_refinement/data/fixedwing_asset_candidates_20261106/
```

原328条图库没有覆盖或改写，既有训练配置未切换。已有模型实验仍使用其原328条配置，不能把其指标归因于这14张新素材。

在后续新训练配置中，将 `online_replacement.cache` 指向上述342条素材的背景缓存视图即可沿用原57,461条安全背景。当前视图绑定的是 `catalog.json` 的342条，不是331条；如需训练使用去重版，须另外生成对应331条的索引和校验值，不能只替换文件导致ID/哈希不一致。本次未启动新训练。

## 方法与检查

沿用已固定版本的BiRefNet分割模型：`e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`。模型仅预测alpha透明度；原图按EXIF显示方向解码，RGB像素不生成、不重绘、不锐化。Freewing两张照片先限定完整飞机ROI，减少桌椅和人物干扰。

检查了全部38张原图、17张透明度预览，以及Avanti和Volantex细节。336等同款不同角度照片保留；340、342、354三个分割输出明确排除。最终新增ID为335、336、337、338、339、341、348、349、356、358、369、370、371、372。

通过原合成代码执行14张素材×小/中/大三个目标尺度，共42次检查：所有案例合成成功、原图未改、编辑ROI外像素不变、更新后的YOLO bbox有限且在图像范围内；合成函数还校验了ROI内编辑mask外像素不变及bbox坐标往返。结果见合并目录 `composition_QA.json`。这不是检测精度或完美边界保证，细线和草地接触边缘仍可能有残留。

源图使用权未作为本次处理的一部分完成核验。保留所有来源文件名；Freewing文件名标有Fred Hsu及CC-BY-SA-3.0，后续使用与交付需保留对应来源并核验授权。

## 代码与本地产物

处理代码已本地push，再由服务器pull同步：

```text
/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/import_fixedwing_assets.py
/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/configs/fixedwing_assets_20261106.json
/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/configs/fixedwing_assets_20261106_review.json
```

`segment` 子命令生成候选，不自动认可；`publish` 只合并已通过复核的候选，校验旧/新PNG、重复与ID冲突，保留原图库。

本文档同目录提供 `fixedwing_new14_rgba_20261106.tar.gz`、解压后的 `fixedwing_new14_rgba_20261106/`、`accepted_contact_sheet.jpg`、`summary.json`、`composition_QA.json`。包中只有14张新增PNG及其清单，不是342张全库；完整合并图库在服务器上述位置。
