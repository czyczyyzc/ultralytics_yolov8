#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/chenziye/datasets/anti_uav/Anti-UAV300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack}"
MODALITY="${MODALITY:-rgb}"
VARIANT="${VARIANT:-v2}"
FRAME_STEP="${FRAME_STEP:-1}"
MIN_BOX_SIZE="${MIN_BOX_SIZE:-2}"
NANOTRACK_ROOT="${NANOTRACK_ROOT:-${REPO_ROOT}/third_party/SiamTrackers/NanoTrack}"
PREPARE_NANOTRACK="${PREPARE_NANOTRACK:-1}"
PRETRAINED="${PRETRAINED:-}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VIDEOS_PER_EPOCH="${VIDEOS_PER_EPOCH:-0}"
DEVICE="${DEVICE:-0}"
NAME="${NAME:-nanotrack_${MODALITY}_${VARIANT}_anti_uav300}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/anti_uav/${NAME}}"
OVERWRITE_EXPORT="${OVERWRITE_EXPORT:-0}"

CONVERTER="${REPO_ROOT}/scripts/anti_uav/convert_anti_uav300_nanotrack.py"
SETUP_SCRIPT="${REPO_ROOT}/scripts/anti_uav/setup_nanotrack.sh"
CONFIG_WRITER="${REPO_ROOT}/scripts/anti_uav/write_nanotrack_config.py"

if [[ "${PREPARE_NANOTRACK}" == "1" ]]; then
  VARIANT="${VARIANT}" NANOTRACK_ROOT="${NANOTRACK_ROOT}" "${SETUP_SCRIPT}"
fi

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Extracted Anti-UAV300 root not found: ${SOURCE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${NANOTRACK_ROOT}/nanotrack" ]]; then
  echo "NanoTrack workspace not found: ${NANOTRACK_ROOT}" >&2
  exit 1
fi

convert_args=(
  "${CONVERTER}"
  --source-root "${SOURCE_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --modalities "${MODALITY}"
  --frame-step "${FRAME_STEP}"
  --min-box-size "${MIN_BOX_SIZE}"
)
if [[ "${OVERWRITE_EXPORT}" == "1" ]]; then
  convert_args+=(--overwrite)
fi

modality_root="${OUTPUT_ROOT}/${MODALITY}"
train_json="${modality_root}/train.json"
crop_root="${modality_root}/crop511"

if [[ ! -f "${train_json}" || "${OVERWRITE_EXPORT}" == "1" ]]; then
  "${PYTHON_BIN}" "${convert_args[@]}"
fi

mkdir -p "${RUN_ROOT}"
CONFIG_PATH="${RUN_ROOT}/config.yaml"
LOG_DIR="${RUN_ROOT}/logs"
SNAPSHOT_DIR="${RUN_ROOT}/snapshots"

if [[ -z "${PRETRAINED}" ]]; then
  PRETRAINED="${NANOTRACK_ROOT}/models/pretrained/nanotrack${VARIANT}.pth"
fi

"${PYTHON_BIN}" "${CONFIG_WRITER}" \
  --output "${CONFIG_PATH}" \
  --dataset-name "ANTIUAV300_${MODALITY^^}" \
  --crop-root "${crop_root}" \
  --train-json "${train_json}" \
  --variant "${VARIANT}" \
  --pretrained "${PRETRAINED}" \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --videos-per-epoch "${VIDEOS_PER_EPOCH}"

export CUDA_VISIBLE_DEVICES="${DEVICE}"
export PYTHONPATH="${NANOTRACK_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

(
  cd "${NANOTRACK_ROOT}"
  "${PYTHON_BIN}" ./bin/train.py --cfg "${CONFIG_PATH}"
)
