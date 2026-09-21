#!/usr/bin/env bash
# Same throughput-oriented capture settings; overlap sensor startup with model load.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export ANTI_UAV_CAMERA_DEPLOY=${ANTI_UAV_CAMERA_DEPLOY:-/home/orangepi/deployments/expanded28_camera_latency_20260921}
exec bash "$HERE/run_camera_on_board.sh" --camera-start overlap "$@"
