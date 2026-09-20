# Drone Catalog: 279 Alpha-Only Cutout Candidates

## Locations

- Server: `root@47.107.185.207`
- Output: `/mnt/andrew/anti_uav_model_refinement/data/drone_asset_matted279_20260920/`
- Original workbook: `/mnt/andrew/anti_uav_model_refinement/data/drone_asset_sources_20260917/drone_model_catalog_original.xlsx`
- Source catalog: `/mnt/andrew/anti_uav_model_refinement/data/drone_asset_catalog_20260916/catalog.json`
- Script: `/mnt/chenziye/codes/ultralytics_yolov8/scripts/anti_uav/matte_drone_catalog.py`
- Model: `/mnt/andrew/anti_uav_model_refinement/data/drone_asset_sources_20260917/birefnet-e2bf8e4/`

The 279 records are the opaque images in the 334-entry workbook, not the existing
55 native-alpha entries (53 enabled, 2 excluded). Duplicate source images share
segmentation results but retain separate model IDs. Refer to `summary.json` for
completed counts and `catalog.json` for all file hashes and automatic flags.

## Files

- `index.html`: per-image source/checkerboard/dark-background comparison gallery.
- `contact_sheets/`: overview pages; the checkerboard is a preview only.
- `raw/`: byte-identical embedded workbook images.
- `rgba/<id>.png`: full source resolution; RGB channels are byte-identical to the
  decoded original RGB. Only alpha changes. Verified after PNG encoding.
- `masks/<id>.png`: source-resolution, 8-bit soft alpha mask.
- `cutouts/<id>.png`: tight crops of the transparent results, suitable for the
  existing compositing API after acceptance review. Bounds are in `catalog.json`.
- `previews/<id>.jpg`: labeled comparison cards. Do not use these for training.
- `records/<id>.json`: identity, source and output hashes, crop bounds, flags.
- `run_config.json`: pinned model files, source workbook and script hashes.

## Method and Limits

BiRefNet predicts alpha at 1024 x 1024; masks are resized to source resolution.
The source image is never regenerated, recolored, sharpened, or geometrically
changed. Alpha below 4/255 is suppressed to remove negligible background haze.
Soft boundary alpha is otherwise retained. No automatic largest-component
deletion is performed, because disconnected thin rotors can be legitimate.

The pretrained segmentation model is
[ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet), pinned at
`e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` (MIT model card).
This license does not replace the rights/usage conditions of the original
third-party drone photographs.

Automatic flags identify empty/tiny masks, near-full masks, significant separate
components, and border contact. Passing these checks is NOT proof of a perfect
cutout. Inspect wings/rotors, white halos, transparent plastics, shadows, product
stands, controllers, text, or multiple drones. Low-resolution source images remain
low-resolution. Thin translucent structures may still contain original background
color; RGB preservation does not promise zero halo on every new background.

**This is an isolated candidate library, not an automatic training update.**
The existing 53-asset catalog, cached backgrounds, current data manifests and
training jobs are not changed. A later training integration must create an
explicit reviewed catalog/version and update its expected hashes; do not overwrite
the old catalog or point the training loader blindly at all candidates.

## Reproduce or Resume

Use the existing repository Python plus the isolated dependency overlay. This
avoids changing the packages used by current model training/RKNN jobs.

```bash
cd /mnt/chenziye/codes/ultralytics_yolov8
DATA=/mnt/andrew/anti_uav_model_refinement/data
SOURCES=$DATA/drone_asset_sources_20260917

# Dependency versions are also recorded in scripts/anti_uav/requirements-drone-matting.txt.
# Model code/weights were downloaded at the pinned revision and reviewed locally.
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=$SOURCES/matting_dependencies \
.venv/bin/python -u scripts/anti_uav/matte_drone_catalog.py \
  --catalog $DATA/drone_asset_catalog_20260916/catalog.json \
  --workbook $SOURCES/drone_model_catalog_original.xlsx \
  --model $SOURCES/birefnet-e2bf8e4 \
  --revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 \
  --output $DATA/drone_asset_matted279_20260920 \
  --resume
```

Recheck that GPU 0 is free before rerunning. `--resume` checks source/model/script
identity and per-file hashes; a changed configuration requires a fresh output
directory. `--limit 15` chooses 15 evenly spaced catalog records for a pilot;
remove the limit with `--resume` to finish the same library.
