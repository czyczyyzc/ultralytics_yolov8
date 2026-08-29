#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 BINARY MODEL_RKNN VIDEO_DIR OUTPUT_DIR" >&2
    exit 2
fi

binary=$1
model=$2
video_dir=$3
output_dir=$4

mkdir -p "$output_dir/logs"

for video in "$video_dir"/Video*.mp4; do
    stem=$(basename "$video" .mp4)
    tracks_csv="$output_dir/$stem.csv"
    detector_csv="$output_dir/${stem}_detector.csv"
    summary_json="$output_dir/$stem.json"
    log_path="$output_dir/logs/$stem.log"

    if [[ -s "$tracks_csv" && -s "$detector_csv" && -s "$summary_json" ]]; then
        echo "Skipping completed $stem"
        continue
    fi

    echo "Running $stem"
    "$binary" "$model" "$video" \
        --output-json "$summary_json" \
        --predictions-csv "$detector_csv" \
        --tracker rk_botsort \
        --tracks-csv "$tracks_csv" \
        --core-mask 0_1_2 \
        --workers 3 \
        --queue-size 3 \
        --worker-cpu-base 4 \
        --warmup-frames 50 \
        --conf 0.45 \
        --nms-iou 0.45 \
        --track-high-thresh 0.45 \
        --track-low-thresh 0.10 \
        --new-track-thresh 0.50 \
        --track-first-match-cost 0.92 \
        --track-second-match-cost 0.92 \
        --track-buffer-sec 1.0 \
        --track-prediction-sec 0.0 \
        --track-min-hits 2 \
        2>&1 | tee "$log_path"
done
