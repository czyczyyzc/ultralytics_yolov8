#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/chenziye/datasets/anti_uav/Anti-UAV300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_yolo}"
MODALITY="${MODALITY:-rgb}"
MODEL="${MODEL:-${REPO_ROOT}/checkpoints/yolov8n.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-16}"
PROJECT="${PROJECT:-${REPO_ROOT}/runs/anti_uav}"
NAME="${NAME:-yolov8n_anti_uav300_${MODALITY}}"
FRAME_STEP="${FRAME_STEP:-2}"
NEGATIVE_FRAME_STEP="${NEGATIVE_FRAME_STEP:-8}"
MIN_BOX_SIZE="${MIN_BOX_SIZE:-2}"
CONVERT_LABELS="${CONVERT_LABELS:-1}"
OVERWRITE_EXPORT="${OVERWRITE_EXPORT:-0}"

CONVERTER="${REPO_ROOT}/scripts/anti_uav/convert_anti_uav300.py"

case "${MODALITY}" in
  rgb) DATA="${DATA:-${REPO_ROOT}/ultralytics/cfg/datasets/AntiUAV300RGB.yaml}" ;;
  ir) DATA="${DATA:-${REPO_ROOT}/ultralytics/cfg/datasets/AntiUAV300IR.yaml}" ;;
  full) DATA="${DATA:-${REPO_ROOT}/ultralytics/cfg/datasets/AntiUAV300Full.yaml}" ;;
  *)
    echo "Unsupported MODALITY=${MODALITY}. Use rgb, ir, or full." >&2
    exit 1
    ;;
esac

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Extracted Anti-UAV300 root not found: ${SOURCE_ROOT}" >&2
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
  expected_train="${OUTPUT_ROOT}/train_${MODALITY}.txt"
  expected_val="${OUTPUT_ROOT}/val_${MODALITY}.txt"
  if [[ ! -f "${expected_train}" || ! -f "${expected_val}" || "${OVERWRITE_EXPORT}" == "1" ]]; then
    echo "Converting Anti-UAV300 videos to YOLO labels..."
    convert_args=(
      "${CONVERTER}"
      --source-root "${SOURCE_ROOT}"
      --output-root "${OUTPUT_ROOT}"
      --frame-step "${FRAME_STEP}"
      --negative-frame-step "${NEGATIVE_FRAME_STEP}"
      --min-box-size "${MIN_BOX_SIZE}"
    )
    if [[ "${MODALITY}" != "full" ]]; then
      convert_args+=(--modalities "${MODALITY}")
    fi
    if [[ "${OVERWRITE_EXPORT}" == "1" ]]; then
      convert_args+=(--overwrite)
    fi
    "${PYTHON_BIN}" "${convert_args[@]}"
  else
    echo "Existing exported YOLO dataset found. Skipping conversion."
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
