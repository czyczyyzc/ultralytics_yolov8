#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RKNN_INCLUDE="${RKNN_INCLUDE:-/home/orangepi/ultralytics_yolov8/rknn_model_zoo/3rdparty/rknpu2/include}"
OUTPUT="${1:-$HERE/rknn_yolov8_native/build/libanti_uav_detector.so}"
mkdir -p "$(dirname "$OUTPUT")"
rga_flags=()
if [[ "${AU_ENABLE_RGA:-0}" == 1 ]]; then
  rga_flags=(-DAU_ENABLE_RGA -lrga)
fi
g++ -std=c++17 -O3 -DNDEBUG -march=armv8-a+simd -fPIC -shared \
  -I"$RKNN_INCLUDE" $(pkg-config --cflags opencv4) \
  "$HERE/rknn_yolov8_native/detector_c_api.cpp" -o "$OUTPUT" \
  $(pkg-config --libs opencv4) "${rga_flags[@]}" -lrknnrt -pthread
