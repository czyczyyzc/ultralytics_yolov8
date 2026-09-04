#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 VIDEO_OR_STREAM [OUTPUT_DIR]" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
package_root=$(cd "$script_dir/.." && pwd)
source_uri=$1
output_dir=${2:-"$package_root/output/$(date +%Y%m%d_%H%M%S)"}
config_file=${CONFIG_FILE:-"$package_root/config/rk_botsort_visualization.env"}

if [[ -f "$config_file" ]]; then
    # shellcheck disable=SC1090
    source "$config_file"
fi

binary=${NATIVE_BINARY:-"$script_dir/native_yolov8_video"}
model=${MODEL_RKNN:-"$package_root/models/frozen_p3_addon_p2_960x544_v232_int8.rknn"}
mkdir -p "$output_dir"

summary_json="$output_dir/summary.json"
detections_csv="$output_dir/detections.csv"
tracks_csv="$output_dir/tracks.csv"
confirmed_tracks_csv="$output_dir/tracks_confirmed.csv"

command=(
    "$binary" "$model" "$source_uri"
    --output-json "$summary_json"
    --predictions-csv "$detections_csv"
    --tracker rk_botsort
    --tracks-csv "$tracks_csv"
    --core-mask "${NPU_CORE_MASK:-0_1_2}"
    --workers "${NPU_WORKERS:-3}"
    --queue-size "${QUEUE_SIZE:-12}"
    --worker-cpu-base "${WORKER_CPU_BASE:-4}"
    --warmup-frames "${WARMUP_FRAMES:-100}"
    --conf "${DETECTOR_CONF:-0.01}"
    --nms-iou "${NMS_IOU:-0.45}"
    --track-high-thresh "${TRACK_HIGH_THRESH:-0.03}"
    --track-low-thresh "${TRACK_LOW_THRESH:-0.01}"
    --new-track-thresh "${NEW_TRACK_THRESH:-0.10}"
    --track-first-match-cost "${TRACK_FIRST_MATCH_COST:-0.92}"
    --track-second-match-cost "${TRACK_SECOND_MATCH_COST:-0.92}"
    --track-buffer-sec "${TRACK_BUFFER_SEC:-1.0}"
    --track-prediction-sec "${TRACK_PREDICTION_SEC:-0.0}"
    --track-min-hits "${TRACK_MIN_HITS:-3}"
)

if [[ -n "${SOURCE_FPS:-}" ]]; then
    command+=(--source-fps "$SOURCE_FPS")
fi
if [[ -n "${MAX_FRAMES:-}" ]]; then
    command+=(--max-frames "$MAX_FRAMES")
fi

"${command[@]}"

# Keep the raw tracker output for diagnostics and produce the visualization/API view separately.
awk -F, '
    NR == 1 {
        for (column = 1; column <= NF; ++column) {
            if ($column == "confirmed") confirmed_column = column
        }
        if (!confirmed_column) exit 2
        print
        next
    }
    $confirmed_column == 1 { print }
' "$tracks_csv" > "$confirmed_tracks_csv"

printf 'summary: %s\nconfirmed tracks: %s\n' "$summary_json" "$confirmed_tracks_csv"
