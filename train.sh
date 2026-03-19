#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/chenziye/datasets/vis_drone}"
DATA="${DATA:-${ROOT_DIR}/ultralytics/cfg/datasets/visdrone.yaml}"
MODEL="${MODEL:-${ROOT_DIR}/checkpoints/yolov8n.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
PROJECT="${PROJECT:-${ROOT_DIR}/runs/visdrone}"
NAME="${NAME:-yolov8n_visdrone}"
CONVERT_LABELS="${CONVERT_LABELS:-1}"
OVERWRITE_LABELS="${OVERWRITE_LABELS:-0}"

CONVERTER="${ROOT_DIR}/ultralytics/data/scripts/convert_visdrone.py"
TRAIN_SPLIT="${DATASET_ROOT}/VisDrone2019-DET-train"
VAL_SPLIT="${DATASET_ROOT}/VisDrone2019-DET-val"

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
  if [[ ! -d "${TRAIN_SPLIT}/labels" || ! -d "${VAL_SPLIT}/labels" || "${OVERWRITE_LABELS}" == "1" ]]; then
    echo "Converting VisDrone annotations to YOLO labels..."
    convert_args=(
      "${CONVERTER}"
      --root "${DATASET_ROOT}"
      --splits VisDrone2019-DET-train VisDrone2019-DET-val
    )
    if [[ "${OVERWRITE_LABELS}" == "1" ]]; then
      convert_args+=(--overwrite)
    fi
    "${PYTHON_BIN}" "${convert_args[@]}"
  else
    echo "Existing YOLO labels found. Skipping conversion."
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
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<PY
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
