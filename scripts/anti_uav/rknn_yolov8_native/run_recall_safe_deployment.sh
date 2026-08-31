#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 VIDEO_OR_STREAM [OUTPUT_DIR]" >&2
    exit 2
fi

source_uri=$1
output_dir=${2:-/data/anti_uav/output/recall_safe_neg15}
config_path=${ANTI_UAV_CONFIG:-/data/anti_uav/config/recall_safe_neg15.env}

if [[ ! -r "$config_path" ]]; then
    echo "Deployment config is not readable: $config_path" >&2
    exit 1
fi

# The config is a trusted, board-local release artifact maintained with this script.
source "$config_path"

required=(BINARY MODEL WORKERS CORE_MASK WORKER_CPU_BASE QUEUE_SIZE SOURCE_FPS DETECTOR_CONF NMS_IOU)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required config value: $name" >&2
        exit 1
    fi
done

if [[ ! -x "$BINARY" ]]; then
    echo "Inference binary is not executable: $BINARY" >&2
    exit 1
fi
if [[ ! -r "$MODEL" ]]; then
    echo "RKNN model is not readable: $MODEL" >&2
    exit 1
fi

mkdir -p "$output_dir"

exec "$BINARY" "$MODEL" "$source_uri" \
    --workers "$WORKERS" \
    --core-mask "$CORE_MASK" \
    --worker-cpu-base "$WORKER_CPU_BASE" \
    --queue-size "$QUEUE_SIZE" \
    --tracker rk_botsort \
    --source-fps "$SOURCE_FPS" \
    --conf "$DETECTOR_CONF" \
    --nms-iou "$NMS_IOU" \
    --track-high-thresh "${TRACK_HIGH_THRESH:-0.03}" \
    --track-low-thresh "${TRACK_LOW_THRESH:-0.01}" \
    --new-track-thresh "${NEW_TRACK_THRESH:-0.05}" \
    --track-first-match-cost "${TRACK_FIRST_MATCH_COST:-0.92}" \
    --track-second-match-cost "${TRACK_SECOND_MATCH_COST:-0.92}" \
    --track-buffer-sec "${TRACK_BUFFER_SEC:-1.0}" \
    --track-prediction-sec "${TRACK_PREDICTION_SEC:-0.0}" \
    --track-min-hits "${TRACK_MIN_HITS:-2}" \
    --output-json "$output_dir/benchmark.json" \
    --predictions-csv "$output_dir/detections.csv" \
    --tracks-csv "$output_dir/tracks.csv"
