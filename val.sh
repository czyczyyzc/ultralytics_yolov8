#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA="${DATA:-${ROOT_DIR}/ultralytics/cfg/datasets/visdrone.yaml}"
MODEL="${MODEL:-${ROOT_DIR}/runs/visdrone/yolov8n_visdrone/weights/best.pt}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
SPLIT="${SPLIT:-val}"
PROJECT="${PROJECT:-${ROOT_DIR}/runs/visdrone_val}"
NAME="${NAME:-yolov8n_visdrone_val}"

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
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<PY
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
