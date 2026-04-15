#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/chenziye/datasets/anti_uav/Anti-UAV300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_yolo}"
MODEL="${MODEL:-${REPO_ROOT}/checkpoints/yolov8n.pt}"

# Recommended first retraining pass after the over-regularized 512-batch run.
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-12}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-128}"
NBS="${NBS:-128}"
DEVICE="${DEVICE:-0,1,2,3,4,5,6,7}"
WORKERS="${WORKERS:-64}"
PROJECT="${PROJECT:-${REPO_ROOT}/runs/anti_uav}"
NAME="${NAME:-yolov8n_anti_uav300_rgb_8gpu_b128_e50_nbs128}"
CONVERT_LABELS="${CONVERT_LABELS:-0}"

exec env \
  PYTHON_BIN="${PYTHON_BIN}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  MODALITY=rgb \
  MODEL="${MODEL}" \
  EPOCHS="${EPOCHS}" \
  PATIENCE="${PATIENCE}" \
  IMGSZ="${IMGSZ}" \
  BATCH="${BATCH}" \
  NBS="${NBS}" \
  DEVICE="${DEVICE}" \
  WORKERS="${WORKERS}" \
  PROJECT="${PROJECT}" \
  NAME="${NAME}" \
  CONVERT_LABELS="${CONVERT_LABELS}" \
  bash "${SCRIPT_DIR}/train_detect.sh"
