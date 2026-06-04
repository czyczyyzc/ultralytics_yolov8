#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/build/rk_letterbox_preprocess.so}"

mkdir -p "$(dirname "${OUT}")"

CXX="${CXX:-g++}"
CXXFLAGS=(-O3 -std=c++17 -fPIC -shared)
DEFINES=()
PKG_CFLAGS=()
PKG_LIBS=()

if pkg-config --exists librga; then
  DEFINES+=(-DWITH_RGA)
  read -r -a RGA_CFLAGS <<<"$(pkg-config --cflags librga)"
  read -r -a RGA_LIBS <<<"$(pkg-config --libs librga)"
  PKG_CFLAGS+=("${RGA_CFLAGS[@]}")
  PKG_LIBS+=("${RGA_LIBS[@]}")
fi

if pkg-config --exists opencv4; then
  DEFINES+=(-DWITH_OPENCV)
  read -r -a OPENCV_CFLAGS <<<"$(pkg-config --cflags opencv4)"
  read -r -a OPENCV_LIBS <<<"$(pkg-config --libs opencv4)"
  PKG_CFLAGS+=("${OPENCV_CFLAGS[@]}")
  PKG_LIBS+=("${OPENCV_LIBS[@]}")
fi

if [[ "${#DEFINES[@]}" -eq 0 ]]; then
  echo "Neither librga nor opencv4 was found by pkg-config; cannot build preprocess library." >&2
  exit 1
fi

"${CXX}" "${CXXFLAGS[@]}" "${DEFINES[@]}" "${PKG_CFLAGS[@]}" \
  "${SCRIPT_DIR}/rk_letterbox_preprocess.cpp" \
  -o "${OUT}" \
  "${PKG_LIBS[@]}"

echo "${OUT}"
