#!/usr/bin/env bash
# Isolated OpenCV headers/libraries, without replacing the board's system packages.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?Output directory}
PREFIX=${2:?Extracted Ubuntu arm64 dependency prefix}
RKNN_INCLUDE=${RKNN_INCLUDE:?Path containing rknn_api.h}
RKNN_LIB=${RKNN_LIB:?Path containing librknnrt.so}
mkdir -p "$OUT"
inc=(-I"$PREFIX/usr/include/opencv4")
lib=(-L"$PREFIX/usr/lib/aarch64-linux-gnu" -Wl,-rpath,"$PREFIX/usr/lib/aarch64-linux-gnu")
cv=(-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio)
flags=(-std=c++17 -O3 -DNDEBUG -ffp-contract=off)
g++ "${flags[@]}" -fPIC -shared "$HERE/tracker.cpp" "$HERE/third_party/lap/lapjv.cpp" -o "$OUT/libdist_tracker.so"
g++ "${flags[@]}" -fPIC -shared "${inc[@]}" "$HERE/gmc.cpp" -o "$OUT/libdist_gmc.so" \
    "${lib[@]}" -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_calib3d
g++ "${flags[@]}" -march=armv8-a+simd -fPIC -shared "${inc[@]}" -I"$RKNN_INCLUDE" \
    "$HERE/../rknn_yolov8_native/detector_c_api.cpp" -o "$OUT/libanti_uav_detector.so" \
    "${lib[@]}" "${cv[@]}" -L"$RKNN_LIB" -Wl,-rpath,"$RKNN_LIB" -lrknnrt -pthread
g++ "${flags[@]}" "${inc[@]}" "$HERE/video.cpp" -o "$OUT/anti_uav_dist_native" \
    "${lib[@]}" "${cv[@]}" -lcrypto -ldl -pthread
g++ "${flags[@]}" "$HERE/fused_half_rgb_test.cpp" -o "$OUT/fused_half_rgb_test"
"$OUT/fused_half_rgb_test"
g++ "${flags[@]}" "$HERE/latest_frame_slot_test.cpp" -pthread -o "$OUT/latest_frame_slot_test"
"$OUT/latest_frame_slot_test"
