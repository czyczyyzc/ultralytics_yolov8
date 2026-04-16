#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-${REPO_ROOT}/runs/anti_uav/yolov8n_anti_uav300_rgb/weights/best.pt}"
SEQUENCE_ROOT="${SEQUENCE_ROOT:-}"
MODALITY="${MODALITY:-rgb}"
TRACKER="${TRACKER:-template_match}"
INPUT_MODE="${INPUT_MODE:-rgb}"
DETECT_INTERVAL="${DETECT_INTERVAL:-8}"
MAX_LOST="${MAX_LOST:-30}"
TILE_SIZE="${TILE_SIZE:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-}"
SUMMARY_JSON="${SUMMARY_JSON:-${REPO_ROOT}/runs/anti_uav/replay_summary.json}"
ERROR_LOG="${ERROR_LOG:-${REPO_ROOT}/runs/anti_uav/replay_errors.jsonl}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${SEQUENCE_ROOT}" ]]; then
  echo "Please set SEQUENCE_ROOT to one Anti-UAV sequence directory." >&2
  exit 1
fi

replay_args=(
  "${REPO_ROOT}/scripts/anti_uav/replay_eval.py"
  --model "${MODEL}"
  --sequence-root "${SEQUENCE_ROOT}"
  --dataset-format anti-uav-json
  --modality "${MODALITY}"
  --tracker "${TRACKER}"
  --input-mode "${INPUT_MODE}"
  --detect-interval "${DETECT_INTERVAL}"
  --max-lost "${MAX_LOST}"
  --auto-confirm
  --summary-json "${SUMMARY_JSON}"
  --error-log "${ERROR_LOG}"
)

if [[ "${TILE_SIZE}" != "0" ]]; then
  replay_args+=(--tile-size "${TILE_SIZE}")
fi

if [[ -n "${SAVE_VIDEO}" ]]; then
  replay_args+=(--save-video "${SAVE_VIDEO}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${EXTRA_ARGS})
  replay_args+=("${extra_args[@]}")
fi

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" "${replay_args[@]}"
