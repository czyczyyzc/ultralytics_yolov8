#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/chenziye/datasets/anti_uav/Anti-UAV300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_yolo}"
MODEL="${MODEL:-${REPO_ROOT}/checkpoints/yolov8n.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-960}"

# 8x A100 80G has ample headroom for YOLOv8n at 960px. Start at 256 total batch
# to improve utilization without jumping straight to an overly aggressive regime.
BATCH="${BATCH:-256}"
DEVICE="${DEVICE:-0,1,2,3,4,5,6,7}"
WORKERS="${WORKERS:-64}"
PROJECT="${PROJECT:-${REPO_ROOT}/runs/anti_uav}"
NAME="${NAME:-yolov8n_anti_uav300_rgb_8gpu}"
CONVERT_LABELS="${CONVERT_LABELS:-0}"

exec env \
  PYTHON_BIN="${PYTHON_BIN}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  MODALITY=rgb \
  MODEL="${MODEL}" \
  EPOCHS="${EPOCHS}" \
  IMGSZ="${IMGSZ}" \
  BATCH="${BATCH}" \
  DEVICE="${DEVICE}" \
  WORKERS="${WORKERS}" \
  PROJECT="${PROJECT}" \
  NAME="${NAME}" \
  CONVERT_LABELS="${CONVERT_LABELS}" \
  bash "${SCRIPT_DIR}/train_detect.sh"
