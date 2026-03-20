#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA="${DATA:-${REPO_ROOT}/ultralytics/cfg/datasets/UAVVaste.yaml}"
MODEL="${MODEL:-${REPO_ROOT}/runs/uavvaste/yolov8n_uavvaste/weights/best.pt}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
SPLIT="${SPLIT:-val}"
PROJECT="${PROJECT:-${REPO_ROOT}/runs/uavvaste_val}"
NAME="${NAME:-yolov8n_uavvaste_val}"

if [[ ! -f "${MODEL}" ]]; then
  echo "Model checkpoint not found: ${MODEL}" >&2
  exit 1
fi

if [[ ! -f "${DATA}" ]]; then
  echo "Dataset yaml not found: ${DATA}" >&2
  exit 1
fi

echo "Validation configuration:"
echo "  MODEL=${MODEL}"
echo "  DATA=${DATA}"
echo "  IMGSZ=${IMGSZ}"
echo "  BATCH=${BATCH}"
echo "  DEVICE=${DEVICE}"
echo "  WORKERS=${WORKERS}"
echo "  SPLIT=${SPLIT}"
echo "  PROJECT=${PROJECT}"
echo "  NAME=${NAME}"

if command -v yolo >/dev/null 2>&1; then
  yolo detect val \
    model="${MODEL}" \
    data="${DATA}" \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    device="${DEVICE}" \
    workers="${WORKERS}" \
    split="${SPLIT}" \
    project="${PROJECT}" \
    name="${NAME}"
else
  echo "'yolo' command not found, falling back to Python API."
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<PY
from ultralytics import YOLO

model = YOLO(r"${MODEL}")
model.val(
    data=r"${DATA}",
    imgsz=int("${IMGSZ}"),
    batch=int("${BATCH}"),
    device="${DEVICE}",
    workers=int("${WORKERS}"),
    split="${SPLIT}",
    project=r"${PROJECT}",
    name=r"${NAME}",
)
PY
fi
