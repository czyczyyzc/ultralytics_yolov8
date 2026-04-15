# Anti-UAV Alerting Scripts

These scripts support a defensive, alerting-only anti-UAV workflow.

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

## `eval_tracker.sh`

Runs the alerting-only tracker stack on one Anti-UAV sequence for replay evaluation.

Example:

```bash
MODEL=runs/anti_uav/yolov8n_anti_uav300_rgb/weights/best.pt \
SEQUENCE_ROOT=/mnt/chenziye/datasets/anti_uav/Anti-UAV-RGBT/0001 \
TRACKER=template_match \
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
