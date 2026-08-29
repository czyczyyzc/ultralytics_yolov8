#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ -z "${RKNN_INCLUDE:-}" ]]; then
  for candidate in \
    /usr/include \
    /usr/local/include \
    /usr/include/rknn \
    /usr/local/include/rknn \
    "${SCRIPT_DIR}/third_party/rknn/include" \
    /home/orangepi/Andrew/anti_uav_rk3588_bundle/profile_rgbbase_ostrack_v4/c_bench/include; do
    if [[ -f "${candidate}/rknn_api.h" ]]; then
      RKNN_INCLUDE="${candidate}"
      break
    fi
  done
fi

if [[ -z "${RKNN_INCLUDE:-}" || ! -f "${RKNN_INCLUDE}/rknn_api.h" ]]; then
  echo "Unable to find rknn_api.h. Set RKNN_INCLUDE=/path/to/rknn/include." >&2
  exit 1
fi

linker_flags=()
if [[ -n "${RKNN_LIB_DIR:-}" ]]; then
  linker_flags+=("-L${RKNN_LIB_DIR}" "-Wl,-rpath,${RKNN_LIB_DIR}")
fi

mkdir -p "${BUILD_DIR}"
g++ \
  -std=c++17 \
  -O3 \
  -DNDEBUG \
  -march=armv8-a+simd \
  -I"${RKNN_INCLUDE}" \
  $(pkg-config --cflags opencv4) \
  "${SCRIPT_DIR}/native_yolov8_video.cpp" \
  -o "${BUILD_DIR}/native_yolov8_video" \
  $(pkg-config --libs opencv4) \
  "${linker_flags[@]}" \
  -lrknnrt \
  -pthread

echo "RKNN include: ${RKNN_INCLUDE}"
echo "Built ${BUILD_DIR}/native_yolov8_video"
