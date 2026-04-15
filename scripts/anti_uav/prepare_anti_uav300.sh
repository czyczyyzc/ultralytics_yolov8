#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-/mnt/chenziye/datasets/anti_uav}"
ZIP_NAME="${ZIP_NAME:-Anti-UAV300.zip}"
EXTRACTED_ROOT="${EXTRACTED_ROOT:-${DATA_ROOT}/Anti-UAV300}"
DOWNLOAD_URL="${DOWNLOAD_URL:-https://huggingface.co/datasets/VoyageWang/antiuav/resolve/main/Anti-UAV300.zip}"
GDOWN_BIN="${GDOWN_BIN:-gdown}"
FRAME_STEP="${FRAME_STEP:-2}"
NEGATIVE_FRAME_STEP="${NEGATIVE_FRAME_STEP:-8}"
MIN_BOX_SIZE="${MIN_BOX_SIZE:-2}"
OVERWRITE_EXPORT="${OVERWRITE_EXPORT:-0}"
EXPORT_MODALITIES="${EXPORT_MODALITIES:-rgb ir}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/anti_uav300_yolo}"

CONVERTER="${REPO_ROOT}/scripts/anti_uav/convert_anti_uav300.py"

mkdir -p "${DATA_ROOT}"

if [[ ! -f "${DATA_ROOT}/${ZIP_NAME}" ]]; then
  if ! command -v "${GDOWN_BIN}" >/dev/null 2>&1; then
    echo "Installing gdown..."
    "${PYTHON_BIN}" -m pip install -q gdown
  fi
  echo "Downloading Anti-UAV300 archive..."
  "${GDOWN_BIN}" "${DOWNLOAD_URL}" -O "${DATA_ROOT}/${ZIP_NAME}"
fi

if [[ ! -d "${EXTRACTED_ROOT}" ]]; then
  echo "Extracting ${ZIP_NAME}..."
  unzip -q "${DATA_ROOT}/${ZIP_NAME}" -d "${DATA_ROOT}"
fi

if [[ ! -d "${EXTRACTED_ROOT}" && -d "${DATA_ROOT}/Anti-UAV-RGBT" ]]; then
  EXTRACTED_ROOT="${DATA_ROOT}/Anti-UAV-RGBT"
fi

convert_args=(
  "${CONVERTER}"
  --source-root "${EXTRACTED_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --frame-step "${FRAME_STEP}"
  --negative-frame-step "${NEGATIVE_FRAME_STEP}"
  --min-box-size "${MIN_BOX_SIZE}"
  --modalities ${EXPORT_MODALITIES}
)

if [[ "${OVERWRITE_EXPORT}" == "1" ]]; then
  convert_args+=(--overwrite)
fi

echo "Exporting YOLO detection dataset..."
"${PYTHON_BIN}" "${convert_args[@]}"
