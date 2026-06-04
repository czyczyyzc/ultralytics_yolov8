#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/build/rk_yolov8_postprocess.so}"

mkdir -p "$(dirname "${OUT}")"
g++ -O3 -std=c++17 -fPIC -shared \
  "${SCRIPT_DIR}/rk_yolov8_postprocess.cpp" \
  -o "${OUT}"

echo "${OUT}"
