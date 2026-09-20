#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${ANTI_UAV_BASE:-$HOME/deployments/expanded28_dist_20260920}"
REPO="${ANTI_UAV_REPO:-$HOME/ultralytics_yolov8_ziye}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Later command-line options override this explicitly reduced-cost GMC profile.
exec "$BASE/venv/bin/python" "$REPO/scripts/anti_uav/run_rknn_dist_optimized.py" \
  --model "$BASE/detector_960x544_int8.rknn" \
  --library "$HERE/libanti_uav_detector.so" \
  --upstream "$BASE/Dist-Tracker" \
  --video "$BASE/Video00009_original.mp4" \
  --workers 3 --core-mode split --contexts shared --preprocess cached \
  --cpus 4,5,6,7 --worker-affinity pinned --dispatch ready --inflight 9 \
  --conf 0.03 --iou 0.45 --gmc compact --gmc-width 320 \
  --gmc-corners 128 --gmc-refresh 5 --gmc-resize-first "$@"
