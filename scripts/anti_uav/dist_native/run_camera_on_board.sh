#!/usr/bin/env bash
set -euo pipefail
DEPLOY=${ANTI_UAV_CAMERA_DEPLOY:-/home/orangepi/deployments/expanded28_camera_20260921}
export LD_LIBRARY_PATH="$DEPLOY/deps/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
exec "$DEPLOY/bin/anti_uav_dist_native" \
    --model "$DEPLOY/detector_960x544_int8.rknn" \
    --detector-library "$DEPLOY/bin/libanti_uav_detector.so" \
    --tracker-library "$DEPLOY/bin/libdist_tracker.so" \
    --gmc-library "$DEPLOY/bin/libdist_gmc.so" \
    --video /dev/video11 --decoder v4l2 --camera-format raw8-gray \
    --camera-policy latest --camera-buffers 4 --camera-fps 120 \
    --preprocess fused --workers 3 --inflight 3 --frames 2000 --warmup 100 "$@"
