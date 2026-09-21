#!/usr/bin/env bash
# Real three-scale P3-P5 RKNN graph; no Add-on P2 computation.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export ANTI_UAV_CAMERA_DEPLOY=${ANTI_UAV_CAMERA_DEPLOY:-/home/orangepi/deployments/expanded28_p3_camera_20260921}
exec bash "$HERE/run_camera_fast_start_on_board.sh" "$@"
