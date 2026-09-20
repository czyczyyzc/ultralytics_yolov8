#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?Output directory required}"
mkdir -p "$OUT"
g++ -std=c++17 -O3 -DNDEBUG -ffp-contract=off -fPIC -shared \
  "$HERE/tracker.cpp" "$HERE/third_party/lap/lapjv.cpp" -o "$OUT/libdist_tracker.so"
if [[ -f "$HERE/gmc.cpp" ]]; then
  g++ -std=c++17 -O3 -DNDEBUG -ffp-contract=off -fPIC -shared \
    $(pkg-config --cflags opencv4) "$HERE/gmc.cpp" -o "$OUT/libdist_gmc.so" \
    $(pkg-config --libs opencv4)
fi
if [[ -f "$HERE/video.cpp" ]]; then
  g++ -std=c++17 -O3 -DNDEBUG -ffp-contract=off \
    $(pkg-config --cflags opencv4) "$HERE/video.cpp" -o "$OUT/anti_uav_dist_native" \
    $(pkg-config --libs opencv4) -lcrypto -ldl -pthread
fi
