#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/chenziye/datasets/uav_vaste}"
DATA="${DATA:-${REPO_ROOT}/ultralytics/cfg/datasets/UAVVaste.yaml}"
MODEL="${MODEL:-${REPO_ROOT}/checkpoints/yolov8n.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
PROJECT="${PROJECT:-${REPO_ROOT}/runs/uavvaste}"
NAME="${NAME:-yolov8n_uavvaste}"
CONVERT_LABELS="${CONVERT_LABELS:-1}"
OVERWRITE_LABELS="${OVERWRITE_LABELS:-0}"

CONVERTER="${REPO_ROOT}/scripts/uavvaste/convert_uavvaste.py"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root not found: ${DATASET_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "Model checkpoint not found: ${MODEL}" >&2
  exit 1
fi

if [[ ! -f "${DATA}" ]]; then
  echo "Dataset yaml not found: ${DATA}" >&2
  exit 1
fi

if [[ ! -f "${CONVERTER}" ]]; then
  echo "Converter script not found: ${CONVERTER}" >&2
  exit 1
fi

if [[ "${CONVERT_LABELS}" == "1" ]]; then
  if [[ ! -d "${DATASET_ROOT}/labels" || ! -f "${DATASET_ROOT}/train.txt" || ! -f "${DATASET_ROOT}/val.txt" || "${OVERWRITE_LABELS}" == "1" ]]; then
    echo "Converting UAVVaste COCO annotations to YOLO labels and split txt files..."
    convert_args=("${CONVERTER}" --root "${DATASET_ROOT}")
    if [[ "${OVERWRITE_LABELS}" == "1" ]]; then
      convert_args+=(--overwrite)
    fi
    "${PYTHON_BIN}" "${convert_args[@]}"
  else
    echo "Existing YOLO labels and split files found. Skipping conversion."
  fi
fi

echo "Training configuration:"
echo "  MODEL=${MODEL}"
echo "  DATA=${DATA}"
echo "  EPOCHS=${EPOCHS}"
echo "  IMGSZ=${IMGSZ}"
echo "  BATCH=${BATCH}"
echo "  DEVICE=${DEVICE}"
echo "  WORKERS=${WORKERS}"
echo "  PROJECT=${PROJECT}"
echo "  NAME=${NAME}"

if command -v yolo >/dev/null 2>&1; then
  yolo detect train \
    model="${MODEL}" \
    data="${DATA}" \
    epochs="${EPOCHS}" \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    device="${DEVICE}" \
    workers="${WORKERS}" \
    project="${PROJECT}" \
    name="${NAME}"
else
  echo "'yolo' command not found, falling back to Python API."
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<PY
from ultralytics import YOLO

model = YOLO(r"${MODEL}")
model.train(
    data=r"${DATA}",
    epochs=int("${EPOCHS}"),
    imgsz=int("${IMGSZ}"),
    batch=int("${BATCH}"),
    device="${DEVICE}",
    workers=int("${WORKERS}"),
    project=r"${PROJECT}",
    name=r"${NAME}",
)
PY
fi
