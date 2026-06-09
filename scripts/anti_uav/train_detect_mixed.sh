#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$([[ -x "${DEFAULT_PYTHON_BIN}" ]] && echo "${DEFAULT_PYTHON_BIN}" || echo python)}"

ANTIUAV_SOURCE_ROOT="${ANTIUAV_SOURCE_ROOT:-/mnt/chenziye/datasets/anti_uav/Anti-UAV300/train}"
ANTIUAV_YOLO_ROOT="${ANTIUAV_YOLO_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_yolo_trainonly}"
EXTRA_YOLO_ROOT="${EXTRA_YOLO_ROOT:-/mnt/hanlue/hanlue_multirotor_assets_phase1_20260525}"
EXTRA_YOLO_ROOTS="${EXTRA_YOLO_ROOTS:-${EXTRA_YOLO_ROOT}}"
MERGED_YOLO_ROOT="${MERGED_YOLO_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_plus_hanlue_yolo_trainonly}"
MODEL="${MODEL:-${REPO_ROOT}/checkpoints/yolov8n.pt}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-12}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-128}"
NBS="${NBS:-128}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-16}"
PROJECT="${PROJECT:-${REPO_ROOT}/runs/anti_uav}"
NAME="${NAME:-yolov8n_anti_uav300_plus_hanlue_rgb}"
FRAME_STEP="${FRAME_STEP:-2}"
NEGATIVE_FRAME_STEP="${NEGATIVE_FRAME_STEP:-8}"
MIN_BOX_SIZE="${MIN_BOX_SIZE:-2}"
OVERWRITE_EXPORT="${OVERWRITE_EXPORT:-0}"

ANTI_CONVERTER="${REPO_ROOT}/scripts/anti_uav/convert_anti_uav300.py"
MERGER="${REPO_ROOT}/scripts/anti_uav/prepare_detector_mixed_dataset.py"

if [[ ! -d "${ANTIUAV_SOURCE_ROOT}" ]]; then
  echo "Anti-UAV300 root not found: ${ANTIUAV_SOURCE_ROOT}" >&2
  exit 1
fi
read -r -a EXTRA_YOLO_ROOT_ARRAY <<< "${EXTRA_YOLO_ROOTS}"
if [[ "${#EXTRA_YOLO_ROOT_ARRAY[@]}" -eq 0 ]]; then
  echo "No extra YOLO roots configured." >&2
  exit 1
fi
for extra_root in "${EXTRA_YOLO_ROOT_ARRAY[@]}"; do
  if [[ ! -d "${extra_root}" ]]; then
    echo "Extra YOLO root not found: ${extra_root}" >&2
    exit 1
  fi
done

if [[ ! -f "${ANTIUAV_YOLO_ROOT}/train_rgb.txt" || "${OVERWRITE_EXPORT}" == "1" ]]; then
  "${PYTHON_BIN}" "${ANTI_CONVERTER}" \
    --source-root "${ANTIUAV_SOURCE_ROOT}" \
    --output-root "${ANTIUAV_YOLO_ROOT}" \
    --modalities rgb \
    --frame-step "${FRAME_STEP}" \
    --negative-frame-step "${NEGATIVE_FRAME_STEP}" \
    --min-box-size "${MIN_BOX_SIZE}" \
    $([[ "${OVERWRITE_EXPORT}" == "1" ]] && echo --overwrite)
fi

MERGE_ARGS=(
  "${MERGER}"
  --antiuav-root "${ANTIUAV_YOLO_ROOT}"
  --output-root "${MERGED_YOLO_ROOT}"
  --modality rgb
)
for extra_root in "${EXTRA_YOLO_ROOT_ARRAY[@]}"; do
  MERGE_ARGS+=(--extra-yolo-root "${extra_root}")
done
"${PYTHON_BIN}" "${MERGE_ARGS[@]}"

DATA_YAML="${MERGED_YOLO_ROOT}/AntiUAV300PlusHanlueRGB.yaml"

if command -v yolo >/dev/null 2>&1; then
  yolo detect train \
    model="${MODEL}" \
    data="${DATA_YAML}" \
    epochs="${EPOCHS}" \
    patience="${PATIENCE}" \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    nbs="${NBS}" \
    device="${DEVICE}" \
    workers="${WORKERS}" \
    project="${PROJECT}" \
    name="${NAME}"
else
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<PY
from ultralytics import YOLO
model = YOLO(r"${MODEL}")
model.train(
    data=r"${DATA_YAML}",
    epochs=int("${EPOCHS}"),
    patience=int("${PATIENCE}"),
    imgsz=int("${IMGSZ}"),
    batch=int("${BATCH}"),
    nbs=int("${NBS}"),
    device="${DEVICE}",
    workers=int("${WORKERS}"),
    project=r"${PROJECT}",
    name=r"${NAME}",
)
PY
fi
