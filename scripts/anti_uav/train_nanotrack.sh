#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$([[ -x "${DEFAULT_PYTHON_BIN}" ]] && echo "${DEFAULT_PYTHON_BIN}" || echo python)}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/chenziye/datasets/anti_uav/Anti-UAV300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack}"
MODALITY="${MODALITY:-rgb}"
VARIANT="${VARIANT:-v2}"
FRAME_STEP="${FRAME_STEP:-1}"
MIN_BOX_SIZE="${MIN_BOX_SIZE:-2}"
NANOTRACK_ROOT="${NANOTRACK_ROOT:-${REPO_ROOT}/third_party/nanotrack_vendor}"
PRETRAINED="${PRETRAINED:-}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VIDEOS_PER_EPOCH="${VIDEOS_PER_EPOCH:-0}"
DEVICE="${DEVICE:-cuda:0}"
NAME="${NAME:-nanotrack_${MODALITY}_${VARIANT}_anti_uav300}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/anti_uav/${NAME}}"
OVERWRITE_EXPORT="${OVERWRITE_EXPORT:-0}"
SAVE_EVERY="${SAVE_EVERY:-5}"
BACKGROUND_FRAME_STEP="${BACKGROUND_FRAME_STEP:-6}"
DISTRACTOR_FRAME_STEP="${DISTRACTOR_FRAME_STEP:-2}"
NEG_RATIO="${NEG_RATIO:-0.35}"
NEG_SAME_SEQ_PROB="${NEG_SAME_SEQ_PROB:-0.75}"
NEG_BACKGROUND_PROB="${NEG_BACKGROUND_PROB:-0.35}"

CONVERTER="${REPO_ROOT}/scripts/anti_uav/convert_anti_uav300_nanotrack.py"
CONFIG_WRITER="${REPO_ROOT}/scripts/anti_uav/write_nanotrack_config.py"
TRAINER="${REPO_ROOT}/scripts/anti_uav/train_nanotrack_local.py"

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
  --background-frame-step "${BACKGROUND_FRAME_STEP}"
  --distractor-frame-step "${DISTRACTOR_FRAME_STEP}"
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
  candidate="${NANOTRACK_ROOT}/models/pretrained/nanotrack${VARIANT}.pth"
  if [[ -f "${candidate}" ]]; then
    PRETRAINED="${candidate}"
  fi
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
  --videos-per-epoch "${VIDEOS_PER_EPOCH}" \
  --neg-ratio "${NEG_RATIO}" \
  --neg-same-seq-prob "${NEG_SAME_SEQ_PROB}" \
  --neg-background-prob "${NEG_BACKGROUND_PROB}"

export PYTHONPATH="${NANOTRACK_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

launch_args=(
  --cfg "${CONFIG_PATH}"
  --device "${DEVICE}"
  --save-every "${SAVE_EVERY}"
)

if [[ "${DEVICE}" == cuda:* && "${DEVICE}" == *,* ]]; then
  visible_devices="${DEVICE#cuda:}"
  IFS=',' read -r -a gpu_ids <<< "${visible_devices}"
  world_size="${#gpu_ids[@]}"
  if [[ "${world_size}" -lt 2 ]]; then
    echo "Expected at least 2 GPUs in DEVICE for DDP launch, got: ${DEVICE}" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${visible_devices}"
  launch_args[3]="cuda"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${world_size}" "${TRAINER}" "${launch_args[@]}"
else
  "${PYTHON_BIN}" "${TRAINER}" "${launch_args[@]}"
fi
