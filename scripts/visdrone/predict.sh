#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/chenziye/datasets/vis_drone}"
MODEL="${MODEL:-${REPO_ROOT}/runs/visdrone/yolov8n_visdrone/weights/best.pt}"
SOURCE="${SOURCE:-${DATASET_ROOT}/VisDrone2019-DET-val/images}"
IMGSZ="${IMGSZ:-960}"
DEVICE="${DEVICE:-0}"
CONF="${CONF:-0.25}"
PROJECT="${PROJECT:-${REPO_ROOT}/runs/visdrone_predict}"
NAME="${NAME:-yolov8n_visdrone_pred}"
SAVE_TXT="${SAVE_TXT:-0}"
SAVE_CONF="${SAVE_CONF:-0}"
SAVE_TXT_BOOL="False"
SAVE_CONF_BOOL="False"

if [[ "${SAVE_TXT}" == "1" ]]; then
  SAVE_TXT_BOOL="True"
fi
if [[ "${SAVE_CONF}" == "1" ]]; then
  SAVE_CONF_BOOL="True"
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "Model checkpoint not found: ${MODEL}" >&2
  exit 1
fi

if [[ ! -e "${SOURCE}" ]]; then
  echo "Prediction source not found: ${SOURCE}" >&2
  exit 1
fi

echo "Prediction configuration:"
echo "  MODEL=${MODEL}"
echo "  SOURCE=${SOURCE}"
echo "  IMGSZ=${IMGSZ}"
echo "  DEVICE=${DEVICE}"
echo "  CONF=${CONF}"
echo "  PROJECT=${PROJECT}"
echo "  NAME=${NAME}"

if command -v yolo >/dev/null 2>&1; then
  predict_args=(
    detect predict
    model="${MODEL}"
    source="${SOURCE}"
    imgsz="${IMGSZ}"
    device="${DEVICE}"
    conf="${CONF}"
    project="${PROJECT}"
    name="${NAME}"
  )
  if [[ "${SAVE_TXT}" == "1" ]]; then
    predict_args+=(save_txt=True)
  fi
  if [[ "${SAVE_CONF}" == "1" ]]; then
    predict_args+=(save_conf=True)
  fi
  yolo "${predict_args[@]}"
else
  echo "'yolo' command not found, falling back to Python API."
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<PY
from ultralytics import YOLO

model = YOLO(r"${MODEL}")
model.predict(
    source=r"${SOURCE}",
    imgsz=int("${IMGSZ}"),
    device="${DEVICE}",
    conf=float("${CONF}"),
    project=r"${PROJECT}",
    name=r"${NAME}",
    save_txt=${SAVE_TXT_BOOL},
    save_conf=${SAVE_CONF_BOOL},
)
PY
fi
