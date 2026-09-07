#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
training_root=${TRAINING_ROOT:-"$repo_root/runs/anti_uav/real_gray_yolov8n_frozen_p3_addon_p2_manual_clips0123_20260904"}
experiment_root=${EXPERIMENT_ROOT:-"$repo_root/runs/anti_uav/causal_p3_roi_vs_addon_p2_20260907"}
video_path=${VIDEO_PATH:-/mnt/andrew/anti_uav_model_refinement/external_eval/real_gray_mp4/Video00004.mp4}
gt_path=${GT_PATH:-/mnt/andrew/anti_uav_model_refinement/external_eval/real_gray_tracker_eval_v2/annotations/Video00004.visible.json}
device=${DEVICE:-0}
p3_model=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["p3_model"])' "$training_root/training_addon/final/training_manifest.json")
addon_model="$training_root/training_addon/final/weights/best.pt"
"$python_bin" -m unittest scripts.anti_uav.test_causal_roi_comparison -v
mkdir -p "$experiment_root"
for arm in p3_full p3_roi2x p3_roi4x addon_full; do
    model="$p3_model"
    zoom=1
    case "$arm" in
        p3_roi2x) zoom=2 ;;
        p3_roi4x) zoom=4 ;;
        addon_full) model="$addon_model" ;;
    esac
    "$python_bin" scripts/anti_uav/run_causal_roi_comparison.py \
        --model "$model" --video "$video_path" --output "$experiment_root/$arm" \
        --name "$arm" --zoom "$zoom" --refresh-interval 10 --device "$device"
done
"$python_bin" scripts/anti_uav/summarize_causal_roi_comparison.py \
    --root "$experiment_root" --ground-truth "$gt_path"
"$python_bin" scripts/anti_uav/render_causal_roi_comparison.py --root "$experiment_root"
