#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$([[ -x "${DEFAULT_PYTHON_BIN}" ]] && echo "${DEFAULT_PYTHON_BIN}" || echo python)}"

ANTIUAV_SOURCE_ROOT="${ANTIUAV_SOURCE_ROOT:-/mnt/chenziye/datasets/anti_uav/Anti-UAV300/train}"
ANTIUAV_OUTPUT_ROOT="${ANTIUAV_OUTPUT_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_nanotrack_trainonly}"
EXTRA_TRACKER_ROOT="${EXTRA_TRACKER_ROOT:-/mnt/hanlue/hanlue_multirotor_assets_phase1_20260525/tracker_sequences}"
EXTRA_IMAGE_ROOT="${EXTRA_IMAGE_ROOT:-/mnt/hanlue/hanlue_multirotor_assets_phase1_20260525/images}"
EXTRA_OUTPUT_ROOT="${EXTRA_OUTPUT_ROOT:-/mnt/chenziye/datasets/anti_uav/hanlue_nanotrack}"
MERGED_OUTPUT_ROOT="${MERGED_OUTPUT_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_plus_hanlue_nanotrack}"
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
NAME="${NAME:-nanotrack_rgb_v2_anti_uav300_plus_hanlue}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/anti_uav/${NAME}}"
OVERWRITE_EXPORT="${OVERWRITE_EXPORT:-0}"
SAVE_EVERY="${SAVE_EVERY:-5}"
FRAME_RANGE="${FRAME_RANGE:-45}"
BASE_LR="${BASE_LR:-0.0035}"
BACKGROUND_FRAME_STEP="${BACKGROUND_FRAME_STEP:-6}"
DISTRACTOR_FRAME_STEP="${DISTRACTOR_FRAME_STEP:-2}"
TRANSITION_WINDOW="${TRANSITION_WINDOW:-8}"
HARD_NEGATIVE_ERRORS="${HARD_NEGATIVE_ERRORS:-}"
NEG_RATIO="${NEG_RATIO:-0.30}"
NEG_SAME_SEQ_PROB="${NEG_SAME_SEQ_PROB:-0.75}"
NEG_BACKGROUND_PROB="${NEG_BACKGROUND_PROB:-0.15}"
NEG_TRANSITION_PROB="${NEG_TRANSITION_PROB:-0.12}"
NEG_HARD_PROB="${NEG_HARD_PROB:-0.35}"
TRANSITION_TEMPLATE_PROB="${TRANSITION_TEMPLATE_PROB:-0.30}"
FAST_MOTION_PROB="${FAST_MOTION_PROB:-0.40}"
FAST_MOTION_MIN_GAP="${FAST_MOTION_MIN_GAP:-12}"

ANTI_CONVERTER="${REPO_ROOT}/scripts/anti_uav/convert_anti_uav300_nanotrack.py"
EXTRA_CONVERTER="${REPO_ROOT}/scripts/anti_uav/convert_tracker_sequences_nanotrack.py"
MERGER="${REPO_ROOT}/scripts/anti_uav/merge_nanotrack_datasets.py"
CONFIG_WRITER="${REPO_ROOT}/scripts/anti_uav/write_nanotrack_config.py"
TRAINER="${REPO_ROOT}/scripts/anti_uav/train_nanotrack_local.py"

if [[ ! -d "${ANTIUAV_SOURCE_ROOT}" ]]; then
  echo "Anti-UAV300 root not found: ${ANTIUAV_SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${EXTRA_TRACKER_ROOT}" ]]; then
  echo "Extra tracker root not found: ${EXTRA_TRACKER_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${NANOTRACK_ROOT}/nanotrack" ]]; then
  echo "NanoTrack workspace not found: ${NANOTRACK_ROOT}" >&2
  exit 1
fi

anti_train_json="${ANTIUAV_OUTPUT_ROOT}/${MODALITY}/train.json"
if [[ ! -f "${anti_train_json}" || "${OVERWRITE_EXPORT}" == "1" ]]; then
  anti_args=(
    "${ANTI_CONVERTER}"
    --source-root "${ANTIUAV_SOURCE_ROOT}"
    --output-root "${ANTIUAV_OUTPUT_ROOT}"
    --modalities "${MODALITY}"
    --frame-step "${FRAME_STEP}"
    --min-box-size "${MIN_BOX_SIZE}"
    --background-frame-step "${BACKGROUND_FRAME_STEP}"
    --distractor-frame-step "${DISTRACTOR_FRAME_STEP}"
    --transition-window "${TRANSITION_WINDOW}"
  )
  if [[ -n "${HARD_NEGATIVE_ERRORS}" ]]; then
    read -r -a hard_negative_args <<< "${HARD_NEGATIVE_ERRORS}"
    anti_args+=(--hard-negative-errors "${hard_negative_args[@]}")
  fi
  if [[ "${OVERWRITE_EXPORT}" == "1" ]]; then
    anti_args+=(--overwrite)
  fi
  "${PYTHON_BIN}" "${anti_args[@]}"
fi

extra_train_json="${EXTRA_OUTPUT_ROOT}/${MODALITY}/train.json"
if [[ ! -f "${extra_train_json}" || "${OVERWRITE_EXPORT}" == "1" ]]; then
  extra_args=(
    "${EXTRA_CONVERTER}"
    --source-root "${EXTRA_TRACKER_ROOT}"
    --image-root "${EXTRA_IMAGE_ROOT}"
    --output-root "${EXTRA_OUTPUT_ROOT}"
  )
  if [[ "${OVERWRITE_EXPORT}" == "1" ]]; then
    extra_args+=(--overwrite)
  fi
  "${PYTHON_BIN}" "${extra_args[@]}"
fi

merged_train_json="${MERGED_OUTPUT_ROOT}/${MODALITY}/train.json"
if [[ ! -f "${merged_train_json}" || "${OVERWRITE_EXPORT}" == "1" ]]; then
  merge_args=(
    "${MERGER}"
    --input-roots "${ANTIUAV_OUTPUT_ROOT}/${MODALITY}" "${EXTRA_OUTPUT_ROOT}/${MODALITY}"
    --output-root "${MERGED_OUTPUT_ROOT}/${MODALITY}"
  )
  if [[ "${OVERWRITE_EXPORT}" == "1" ]]; then
    merge_args+=(--overwrite)
  fi
  "${PYTHON_BIN}" "${merge_args[@]}"
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
  --dataset-name "ANTIUAV300PLUS_${MODALITY^^}" \
  --crop-root "${MERGED_OUTPUT_ROOT}/${MODALITY}/crop511" \
  --train-json "${MERGED_OUTPUT_ROOT}/${MODALITY}/train.json" \
  --variant "${VARIANT}" \
  --pretrained "${PRETRAINED}" \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --videos-per-epoch "${VIDEOS_PER_EPOCH}" \
  --frame-range "${FRAME_RANGE}" \
  --base-lr "${BASE_LR}" \
  --neg-ratio "${NEG_RATIO}" \
  --neg-same-seq-prob "${NEG_SAME_SEQ_PROB}" \
  --neg-background-prob "${NEG_BACKGROUND_PROB}" \
  --neg-transition-prob "${NEG_TRANSITION_PROB}" \
  --neg-hard-prob "${NEG_HARD_PROB}" \
  --fast-motion-prob "${FAST_MOTION_PROB}" \
  --fast-motion-min-gap "${FAST_MOTION_MIN_GAP}" \
  --transition-template-prob "${TRANSITION_TEMPLATE_PROB}" \
  --transition-frame-window "${TRANSITION_WINDOW}"

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
  export CUDA_VISIBLE_DEVICES="${visible_devices}"
  launch_args[3]="cuda"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${world_size}" "${TRAINER}" "${launch_args[@]}"
else
  "${PYTHON_BIN}" "${TRAINER}" "${launch_args[@]}"
fi
