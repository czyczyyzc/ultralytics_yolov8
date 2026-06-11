# Anti-UAV Alerting Scripts

These scripts support a defensive, alerting-only anti-UAV workflow.

## NanoTrack Notes

If you want a NanoTrack backend inside this repository, the most useful references are:

- `HonglinChu/SiamTrackers/NanoTrack` for the upstream PyTorch tracker and training code
- `Try2ChangeX/NanoTrack_RK3588_python` for RK3588-specific RKNN runtime layout and post-processing

In other words:

- for local replay and fine-tuning, use upstream NanoTrack
- for later RK3588 deployment, use the RK3588 repo as the export/runtime reference

This repository also carries a minimal vendored NanoTrack-compatible snapshot under
`third_party/nanotrack_vendor`. That snapshot is enough for local replay and Anti-UAV300
fine-tuning without any server-side GitHub dependency.

## `prepare_anti_uav300.sh`

Downloads `Anti-UAV300`, extracts it, and converts the videos plus JSON annotations into a YOLO detection dataset.

Example:

```bash
bash scripts/anti_uav/prepare_anti_uav300.sh
```

Useful overrides:

- `DATA_ROOT=/mnt/chenziye/datasets/anti_uav`
- `FRAME_STEP=2`
- `NEGATIVE_FRAME_STEP=8`
- `EXPORT_MODALITIES="rgb ir"`
- `DOWNLOAD_URL=https://huggingface.co/datasets/VoyageWang/antiuav/resolve/main/Anti-UAV300.zip`

## `setup_nanotrack.sh`

Sparse-checks out the upstream NanoTrack workspace under `third_party/SiamTrackers/NanoTrack`
and optionally downloads the matching pretrained checkpoint.

You no longer need this step for the default training flow because `train_nanotrack.sh`
uses the vendored `third_party/nanotrack_vendor` snapshot by default.

Example:

```bash
bash scripts/anti_uav/setup_nanotrack.sh
```

Useful overrides:

- `VARIANT=v1|v2|v3`
- `NANOTRACK_ROOT=/path/to/NanoTrack`
- `DOWNLOAD_PRETRAINED=0`
- `BUILD_EXT=1`

## `convert_anti_uav300_nanotrack.py`

Converts `Anti-UAV300` into a NanoTrack-style `crop511` dataset with modality-specific
`train.json` and `val.json` files.

Example:

```bash
python scripts/anti_uav/convert_anti_uav300_nanotrack.py \
  --source-root /mnt/chenziye/datasets/anti_uav/Anti-UAV300 \
  --output-root /mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack \
  --modalities rgb ir \
  --frame-step 1 \
  --background-frame-step 6 \
  --distractor-frame-step 2
```

Output layout:

- `.../rgb/crop511/<sequence>/<frame>.00.x.jpg`
- `.../rgb/crop511/<sequence>/<frame>.__neg__.x.jpg`
- `.../rgb/crop511/<sequence>/<frame>.__bg__.x.jpg`
- `.../rgb/train.json`
- `.../rgb/val.json`
- `.../rgb/split_manifest.json`
- `.../ir/crop511/<sequence>/<frame>.00.x.jpg`
- `.../ir/crop511/<sequence>/<frame>.__neg__.x.jpg`
- `.../ir/crop511/<sequence>/<frame>.__bg__.x.jpg`
- `.../ir/train.json`
- `.../ir/val.json`

Notes:

- `00` is the positive target track.
- `__neg__` stores same-scene distractor negatives sampled near the UAV without overlapping it.
- `__bg__` stores no-target/background negatives from frames where the UAV is absent.
- `split_manifest.json` preserves the deterministic train/val split so training and validation stay aligned.

## `train_detect.sh`

Trains a YOLO detector on the converted Anti-UAV300 dataset.

Example:

```bash
MODALITY=rgb \
MODEL=checkpoints/yolov8n.pt \
DEVICE=0 \
bash scripts/anti_uav/train_detect.sh
```

Supported `MODALITY` values:

- `rgb`
- `ir`
- `full`

## `train_rgb_8gpu_a100.sh`

Starts Anti-UAV300 RGB detector training on `8x A100 80G` with a more realistic
throughput-oriented default than the generic launcher.

Example:

```bash
bash scripts/anti_uav/train_rgb_8gpu_a100.sh
```

Useful overrides:

- `BATCH=256` default, increase only after checking the first epoch speed and stability
- `DEVICE=0,1,2,3,4,5,6,7`
- `WORKERS=64`
- `NAME=yolov8n_anti_uav300_rgb_8gpu`

## `train_rgb_8gpu_recommended.sh`

Starts the recommended follow-up RGB run for `8x A100 80G` after the earlier
`batch=512` experiment that peaked very early and then regressed.

Default training arguments:

- `EPOCHS=50`
- `PATIENCE=12`
- `IMGSZ=960`
- `BATCH=128`
- `NBS=128`
- `DEVICE=0,1,2,3,4,5,6,7`
- `WORKERS=64`

Example:

```bash
bash scripts/anti_uav/train_rgb_8gpu_recommended.sh
```

## `eval_tracker.sh`

Runs the alerting-only tracker stack on one Anti-UAV sequence for replay evaluation.

Example:

```bash
MODEL=runs/anti_uav/yolov8n_anti_uav300_rgb/weights/best.pt \
SEQUENCE_ROOT=/mnt/chenziye/datasets/anti_uav/Anti-UAV-RGBT/0001 \
TRACKER=template_match \
bash scripts/anti_uav/eval_tracker.sh
```

To evaluate with NanoTrack:

```bash
MODEL=runs/anti_uav/yolov8n_anti_uav300_rgb_8gpu_b128_e50_nbs128/weights/best.pt \
SEQUENCE_ROOT=/mnt/chenziye/datasets/anti_uav/Anti-UAV300/test-dev/20190925_101846_1_1 \
TRACKER=nanotrack \
EXTRA_ARGS="--nanotrack-root /mnt/chenziye/codes/ultralytics_yolov8/third_party/nanotrack_vendor --nanotrack-config /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/nanotrack_rgb_v2_anti_uav300/config.yaml --nanotrack-snapshot /mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/nanotrack_rgb_v2_anti_uav300/snapshots/best.pth" \
bash scripts/anti_uav/eval_tracker.sh
```

## `replay_eval.py`

Offline replay and evaluation for:

- Anti-UAV style JSON annotations
- Drone-vs-Bird custom TXT annotations
- Generic JSONL bbox annotations

It reports frame-level precision/recall, IoU quality, alert precision, and can save an annotated replay video plus
error logs for false positives, false negatives, and bad alerts.

Example:

```bash
python scripts/anti_uav/replay_eval.py \
  --model yolov8n.pt \
  --sequence-root /data/Anti-UAV/seq_001 \
  --dataset-format anti-uav-json \
  --modality rgb \
  --auto-confirm \
  --tile-size 640 \
  --save-video runs/anti_uav/replay.mp4 \
  --summary-json runs/anti_uav/replay_summary.json \
  --error-log runs/anti_uav/replay_errors.jsonl
```

Useful tuning flags:

- `--conf`
- `--detect-interval`
- `--tracker-score-thresh`
- `--min-confidence`
- `--area-min-px`
- `--area-max-ratio`
- `--aspect-min`
- `--aspect-max`
- `--border-margin`
- `--disable-roi-redetect`
- `--disable-full-frame-fallback`
- `--nanotrack-root`
- `--nanotrack-config`
- `--nanotrack-snapshot`
- `--nanotrack-device`

## RK3588 persistent performance governor

If you want repeatable RK3588 detector benchmarks after reboot, use:

- `scripts/anti_uav/rk3588_set_performance.sh`
- `scripts/anti_uav/rk3588_performance_governor.service`

The script pins:

- CPU `cpufreq` governors to `performance`
- `dmc` governor to `performance`
- `npu` governor to `performance`

Typical board-side install:

```bash
sudo install -m 755 scripts/anti_uav/rk3588_set_performance.sh /usr/local/sbin/rk3588_set_performance.sh
sudo install -m 644 scripts/anti_uav/rk3588_performance_governor.service /etc/systemd/system/rk3588-performance-governor.service
sudo systemctl daemon-reload
sudo systemctl enable --now rk3588-performance-governor.service
```

## `batch_replay_eval.py`

Runs `replay_eval.py` over an entire Anti-UAV split and aggregates the results.

Example:

```bash
python scripts/anti_uav/batch_replay_eval.py \
  --model runs/anti_uav/yolov8n_anti_uav300_rgb_8gpu/weights/best.pt \
  --dataset-root /mnt/chenziye/datasets/anti_uav/Anti-UAV300 \
  --split test-dev \
  --modality rgb \
  --output-root runs/anti_uav/testdev_rgb_batch \
  --save-video \
  --auto-confirm \
  --conf 0.35 \
  --detect-interval 4 \
  --tracker-score-thresh 0.35 \
  --min-confidence 0.35
```

## `build_rknn.py`

Generic ONNX to RKNN export helper for RK3588 deployment.

Example:

```bash
python scripts/anti_uav/build_rknn.py \
  --onnx runs/anti_uav/best_rknnopt.onnx \
  --output runs/anti_uav/best.rknn \
  --target rk3588 \
  --quantize \
  --dataset /path/to/calibration.txt
```

## `benchmark_rknn.py`

Benchmarks RKNN forward latency on images or video frames and writes a JSON summary.

Example:

```bash
python scripts/anti_uav/benchmark_rknn.py \
  --model runs/anti_uav/best.rknn \
  --source /path/to/video.mp4 \
  --input-size 640,640 \
  --output-json runs/anti_uav/rknn_benchmark.json \
  --preview-dir runs/anti_uav/rknn_preview
```

## `train_nanotrack.sh`

Runs a NanoTrack fine-tuning job against the converted `Anti-UAV300` crop dataset.

Example:

```bash
MODALITY=rgb \
VARIANT=v2 \
DEVICE=0 \
bash scripts/anti_uav/train_nanotrack.sh
```

Useful overrides:

- `NANOTRACK_ROOT=/path/to/custom/nanotrack/root`
- `PRETRAINED=/path/to/nanotrackv2.pth`
- `EPOCHS=40`
- `BATCH_SIZE=64`
- `VIDEOS_PER_EPOCH=120000`
- `NEG_RATIO=0.35`
- `NEG_SAME_SEQ_PROB=0.75`
- `NEG_BACKGROUND_PROB=0.35`
- `BACKGROUND_FRAME_STEP=6`
- `DISTRACTOR_FRAME_STEP=2`
- `DEVICE=cuda:0`
- `SAVE_EVERY=5`
- `NAME=nanotrack_rgb_v2_anti_uav300`

This launcher keeps the scope on tracking-model fine-tuning only. It does not add any actuation or interception logic.

Example RGB and IR jobs on one machine:

```bash
MODALITY=rgb DEVICE=cuda:0 NAME=nanotrack_rgb_v2_anti_uav300 bash scripts/anti_uav/train_nanotrack.sh
MODALITY=ir DEVICE=cuda:1 NAME=nanotrack_ir_v2_anti_uav300 bash scripts/anti_uav/train_nanotrack.sh
```

## `nanotrack_val_eval.py`

Runs tracker-only validation replays for NanoTrack checkpoints on Anti-UAV300 splits and reports
tracking stability metrics that are more useful than raw training loss when picking checkpoints.

Example:

```bash
python scripts/anti_uav/nanotrack_val_eval.py \
  --source-root /mnt/chenziye/datasets/anti_uav/Anti-UAV300 \
  --converted-root /mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack \
  --modality rgb \
  --split val \
  --config runs/anti_uav/nanotrack_rgb_v2_anti_uav300/config.yaml \
  --snapshot runs/anti_uav/nanotrack_rgb_v2_anti_uav300/snapshots/best.pth \
  --nanotrack-root third_party/nanotrack_vendor \
  --output-json runs/anti_uav/nanotrack_rgb_v2_anti_uav300/val_eval.json
```

Useful flags:

- `--score-threshold`
- `--iou-threshold`
- `--center-threshold`
- `--max-sequences`
- `--per-sequence-dir`
- `--dry-run`

The aggregate report includes `success_rate`, `center_precision`, `absent_fp_rate`, and a
`composite` score intended for checkpoint ranking.

## `nanotrack_checkpoint_sweep.py`

Evaluates a directory of NanoTrack checkpoints with `nanotrack_val_eval.py` and ranks them by
validation metrics instead of training loss.

Example:

```bash
python scripts/anti_uav/nanotrack_checkpoint_sweep.py \
  --snapshot-dir runs/anti_uav/nanotrack_rgb_v2_anti_uav300/snapshots \
  --config runs/anti_uav/nanotrack_rgb_v2_anti_uav300/config.yaml \
  --source-root /mnt/chenziye/datasets/anti_uav/Anti-UAV300 \
  --converted-root /mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack \
  --modality rgb \
  --split val \
  --metric composite \
  --include-best \
  --include-last \
  --output-json runs/anti_uav/nanotrack_rgb_v2_anti_uav300/checkpoint_sweep.json
```

Useful flags:

- `--pattern epoch_*.pth`
- `--metric composite|success_rate|avg_iou|precision|recall`
- `--max-sequences`
- `--dry-run`

## `export_nanotrack_rk3588.py`

Exports the vendored NanoTrack model into three ONNX components aligned with the
`Try2ChangeX/NanoTrack_RK3588_python` runtime split:

- `nanotrack_t_backbone.onnx`
- `nanotrack_x_backbone.onnx`
- `nanotrack_head.onnx`

Example:

```bash
python scripts/anti_uav/export_nanotrack_rk3588.py \
  --cfg runs/anti_uav/nanotrack_rgb_v2_anti_uav300/config.yaml \
  --snapshot runs/anti_uav/nanotrack_rgb_v2_anti_uav300/snapshots/best.pth \
  --output-dir runs/anti_uav/nanotrack_rgb_v2_anti_uav300/rk3588_onnx
```

Useful flags:

- `--device cpu`
- `--opset 12`
- `--template-size 127`
- `--search-size 255`
- `--dry-run`

The generated manifest records the feature tensor shapes so the later RKNN conversion
and board-side runtime can be checked against the expected `T-backbone / X-backbone / Head`
interfaces before building `.rknn` artifacts.

## `setup_rk3588_board.sh`

Bootstraps a Rockchip RK3588 board for board-side deployment:

- creates or reuses a venv under `/data/venvs/anti_uav_rk3588`
- installs `onnxruntime`, `yacs`, and `rknn-toolkit-lite2`
- clones `Try2ChangeX/NanoTrack_RK3588_python`
- downloads the RK-optimized YOLOv8n ONNX from the bundled RKNN model-zoo example

Example on the board after syncing this repository to `/data/codes/ultralytics_yolov8`:

```bash
bash scripts/anti_uav/setup_rk3588_board.sh
```

Useful overrides:

- `VENV_DIR=/data/venvs/custom_env`
- `NANOTRACK_ROOT=/data/codes/NanoTrack_RK3588_python`
- `MODEL_DIR=/data/models/anti_uav`
- `YOLO_ONNX_PATH=/data/models/anti_uav/custom_detector.onnx`
- `DOWNLOAD_YOLO_ONNX=0`

## `anti_uav_rk3588.py`

Runs the alerting-only perception stack on RK3588 without requiring PyTorch on the board.

Detector backends:

- `.onnx` via `onnxruntime`
- `.rknn` via `rknn-toolkit-lite2`

Tracker backends:

- `nanotrack_rknn` using the `Try2ChangeX/NanoTrack_RK3588_python` split runtime
- `template_match` as a zero-dependency fallback

Example with the default RK-optimized YOLOv8n ONNX plus NanoTrack RKNN:

```bash
source /data/venvs/anti_uav_rk3588/bin/activate
python scripts/anti_uav/anti_uav_rk3588.py \
  --model /data/models/anti_uav/yolov8n_rkopt.onnx \
  --source /path/to/video.mp4 \
  --tracker nanotrack_rknn \
  --nanotrack-root /data/codes/NanoTrack_RK3588_python \
  --class-names rknn_model_zoo/examples/yolov8/model/coco_80_labels_list.txt \
  --target-class-names airplane,bird \
  --save-output /data/models/anti_uav/preview.mp4
```

For a custom anti-UAV detector, pass your own class list and model path instead:

```bash
python scripts/anti_uav/anti_uav_rk3588.py \
  --model /data/models/anti_uav/custom_best.rknn \
  --source /path/to/video.mp4 \
  --tracker nanotrack_rknn \
  --nanotrack-root /data/codes/NanoTrack_RK3588_python \
  --class-names drone \
  --target-class-names drone
```

Notes:

- The board-side script accepts either a final `.rknn` detector or an `.onnx` detector for CPU fallback.
- The official RKNN stack uses `RKNN-Toolkit2` on a computer for model conversion and `RKNN-Toolkit-Lite2` on the board for inference.

## `prepare_rknn_calibration.py`

Samples frames from a video into JPEGs plus a `dataset.txt` manifest suitable for RKNN quantization.

Example:

```bash
python scripts/anti_uav/prepare_rknn_calibration.py \
  --source /mnt/chenziye/datasets/anti_uav/Anti-UAV300/test-dev/20190925_101846_1_1/RGB.mp4 \
  --output-dir runs/anti_uav/calibration_rgb_1_1 \
  --dataset-txt runs/anti_uav/calibration_rgb_1_1/dataset.txt \
  --max-frames 64 \
  --frame-step 30 \
  --width 960 \
  --height 960
```

## `export_detector_rknn_legacy.sh`

Host-side helper for boards that still run the older RKNN runtime line such as `librknnrt 1.4.0`.

It does two things:

1. exports a custom Ultralytics detector checkpoint to the repo's RKNN-optimized ONNX layout
2. sanitizes the ONNX graph for RKNN `v1.4.0` compatibility and builds an old-format `.rknn` using a separate `python=3.8` RKNN-Toolkit2 environment

Typical FP32 build on the conversion host:

```bash
CONDA_BIN=/mnt/chenziye/miniconda3/bin/conda \
EXPORT_PYTHON=/mnt/chenziye/codes/ultralytics_yolov8/.venv/bin/python \
RKNN_WHEEL=/path/to/rknn_toolkit2-1.4.0_22dcfef4-cp38-cp38-linux_x86_64.whl \
MODEL=/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/yolov8n_anti_uav300_rgb_8gpu_b128_e50_nbs128/weights/best.pt \
IMGSZ=960 \
RKNN_OUT=/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/yolov8n_anti_uav300_rgb_8gpu_b128_e50_nbs128/weights/best_v140_fp.rknn \
bash scripts/anti_uav/export_detector_rknn_legacy.sh
```

INT8 build with calibration sampled from a video:

```bash
CONDA_BIN=/mnt/chenziye/miniconda3/bin/conda \
EXPORT_PYTHON=/mnt/chenziye/codes/ultralytics_yolov8/.venv/bin/python \
RKNN_WHEEL=/path/to/rknn_toolkit2-1.4.0_22dcfef4-cp38-cp38-linux_x86_64.whl \
MODEL=/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/yolov8n_anti_uav300_rgb_8gpu_b128_e50_nbs128/weights/best.pt \
IMGSZ=960 \
QUANTIZE=1 \
CALIB_SOURCE=/mnt/chenziye/datasets/anti_uav/Anti-UAV300/test-dev/20190925_101846_1_1/RGB.mp4 \
RKNN_OUT=/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/yolov8n_anti_uav300_rgb_8gpu_b128_e50_nbs128/weights/best_v140_i8.rknn \
bash scripts/anti_uav/export_detector_rknn_legacy.sh
```

Notes:

- This flow is specifically for older board runtimes that reject newer RKNN model versions.
- The script separates the export environment from the RKNN conversion environment on purpose: modern Ultralytics export can stay on Python 3.10, while RKNN `v1.4.0` conversion stays on Python 3.8.
- The legacy helper writes an extra `*_v140.onnx` artifact after stripping unsupported `MaxPool.dilations=[1,1]` attributes.
- The official `rknn-toolkit2 v1.4.0` wheel has non-PEP-440 metadata, so the helper unpacks it directly into the target environment instead of relying on `pip install`.
