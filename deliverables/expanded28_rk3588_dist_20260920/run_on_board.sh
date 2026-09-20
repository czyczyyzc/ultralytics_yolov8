#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${ANTI_UAV_REPO:-$HOME/ultralytics_yolov8_ziye}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec "$HERE/venv/bin/python" "$REPO/scripts/anti_uav/run_rknn_dist_pipeline.py" \
  --model "$HERE/detector_960x544_int8.rknn" \
  --library "$HERE/libanti_uav_detector.so" \
  --upstream "$HERE/Dist-Tracker" \
  --video "$HERE/Video00009_original.mp4" \
  --conf 0.03 --iou 0.45 "$@"
