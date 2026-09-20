#!/usr/bin/env bash
set -euo pipefail
DEPLOY=${ANTI_UAV_LATENCY_DEPLOY:-/home/orangepi/deployments/expanded28_dist_latency_20260920}
BASE=${ANTI_UAV_BASE_DEPLOY:-/home/orangepi/deployments/expanded28_dist_20260920}
exec "$DEPLOY/anti_uav_dist_native" \
  --model "$BASE/detector_960x544_int8.rknn" \
  --detector-library "$DEPLOY/libanti_uav_detector.so" \
  --tracker-library "$DEPLOY/libdist_tracker.so" \
  --gmc-library "$DEPLOY/libdist_gmc.so" \
  --video "$BASE/Video00009_original.mp4" \
  --decoder ffmpeg --decode-threads 4 --decode-threading frame \
  --preprocess fused "$@"
