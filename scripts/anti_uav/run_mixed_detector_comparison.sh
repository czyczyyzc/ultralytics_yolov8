#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$([[ -x "${DEFAULT_PYTHON_BIN}" ]] && echo "${DEFAULT_PYTHON_BIN}" || echo python)}"

MODE="${MODE:-all}" # all, train, eval
MODEL="${MODEL:-${REPO_ROOT}/checkpoints/yolov8n.pt}"
IMGSZ="${IMGSZ:-960}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-12}"
BATCH="${BATCH:-128}"
NBS="${NBS:-128}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-16}"
PROJECT="${PROJECT:-${REPO_ROOT}/runs/anti_uav}"
NAME="${NAME:-yolov8n_anti_uav300_hanlue_old_new_rgb_${IMGSZ}}"

ANTIUAV_SOURCE_ROOT="${ANTIUAV_SOURCE_ROOT:-/mnt/chenziye/datasets/anti_uav/Anti-UAV300/train}"
ANTIUAV_DATASET_ROOT="${ANTIUAV_DATASET_ROOT:-/mnt/chenziye/datasets/anti_uav/Anti-UAV300}"
ANTIUAV_YOLO_ROOT="${ANTIUAV_YOLO_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_yolo_trainonly}"
OLD_HANLUE_ROOT="${OLD_HANLUE_ROOT:-/mnt/hanlue/hanlue_multirotor_assets_phase1_20260525}"
NEW_HANLUE_ROOT="${NEW_HANLUE_ROOT:-/mnt/hanlue/hanlue_tracking_v2_full_hdri4k_20260608}"
NEW_HANLUE_DET_ROOT="${NEW_HANLUE_DET_ROOT:-${NEW_HANLUE_ROOT}/detection_yolo}"
MERGED_YOLO_ROOT="${MERGED_YOLO_ROOT:-/mnt/chenziye/datasets/anti_uav/anti_uav300_hanlue_old_new_yolo_trainonly}"

EVAL_ROOT="${EVAL_ROOT:-${PROJECT}/${NAME}_eval}"
EVAL_BATCH="${EVAL_BATCH:-32}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE%%,*}}"
RUN_DETECTION_EVAL="${RUN_DETECTION_EVAL:-1}"
RUN_TRACKING_EVAL="${RUN_TRACKING_EVAL:-1}"
TRACK_LIMIT="${TRACK_LIMIT:-0}"
ANTI_TRACK_SPLIT="${ANTI_TRACK_SPLIT:-test-dev}"
TRACKER="${TRACKER:-nanotrack}"
NANOTRACK_ROOT="${NANOTRACK_ROOT:-${REPO_ROOT}/third_party/nanotrack_vendor}"
NANOTRACK_SNAPSHOT="${NANOTRACK_SNAPSHOT:-${REPO_ROOT}/runs/anti_uav/nanotrack_rgb_v2_anti_uav300_8gpu_absentaware_trainonly_907a622/snapshots/epoch_025.pth}"
PRESENCE_MODEL="${PRESENCE_MODEL:-${REPO_ROOT}/runs/anti_uav/presence_pair_trainonly24_a52f825_model/pair_presence_edl.pt}"

TRAIN_SCRIPT="${SCRIPT_DIR}/train_detect_mixed.sh"
BATCH_REPLAY="${SCRIPT_DIR}/batch_replay_eval.py"
BATCH_TRACKER="${SCRIPT_DIR}/batch_tracker_sequence_eval.py"

run_python() {
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" "$@"
}

write_eval_yamls() {
  local eval_yaml_root="${MERGED_YOLO_ROOT}/eval_yamls"
  mkdir -p "${eval_yaml_root}"
  run_python - <<PY
from pathlib import Path

eval_root = Path(r"${eval_yaml_root}")
anti_root = Path(r"${ANTIUAV_YOLO_ROOT}")
old_root = Path(r"${OLD_HANLUE_ROOT}")
new_root = Path(r"${NEW_HANLUE_DET_ROOT}")

def image_files(path: Path):
    if not path.exists():
        return []
    return sorted(p.resolve() for p in path.iterdir() if p.is_file())

def paired_images(root: Path, split: str):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if image_dir.exists():
        images = image_files(image_dir)
    else:
        image_dir = root / "images"
        label_dir = root / "labels"
        images = [p for p in image_files(image_dir) if p.stem.startswith(f"{split}_")]
    return [p for p in images if (label_dir / f"{p.stem}.txt").exists()]

def write_list(path: Path, items):
    path.write_text("\\n".join(str(item) for item in items) + ("\\n" if items else ""), encoding="utf-8")

def write_yaml(path: Path, root: Path, train: str, val: str):
    path.write_text(
        f"path: {root}\\n"
        f"train: {train}\\n"
        f"val: {val}\\n\\n"
        "names:\\n"
        "  0: drone\\n",
        encoding="utf-8",
    )

write_yaml(eval_root / "anti_uav_rgb.yaml", anti_root, "train_rgb.txt", "val_rgb.txt")
write_yaml(eval_root / "hanlue_old_rgb.yaml", old_root, "images/train", "images/val")

new_train = paired_images(new_root, "train")
new_val = paired_images(new_root, "val")
write_list(eval_root / "hanlue_new_train.txt", new_train)
write_list(eval_root / "hanlue_new_val.txt", new_val)
write_yaml(eval_root / "hanlue_new_rgb.yaml", eval_root, "hanlue_new_train.txt", "hanlue_new_val.txt")

print({
    "eval_yaml_root": str(eval_root),
    "hanlue_new_train": len(new_train),
    "hanlue_new_val": len(new_val),
})
PY
}

train_detector() {
  EXTRA_YOLO_ROOTS="${OLD_HANLUE_ROOT} ${NEW_HANLUE_DET_ROOT}" \
  ANTIUAV_SOURCE_ROOT="${ANTIUAV_SOURCE_ROOT}" \
  ANTIUAV_YOLO_ROOT="${ANTIUAV_YOLO_ROOT}" \
  MERGED_YOLO_ROOT="${MERGED_YOLO_ROOT}" \
  MODEL="${MODEL}" \
  EPOCHS="${EPOCHS}" \
  PATIENCE="${PATIENCE}" \
  IMGSZ="${IMGSZ}" \
  BATCH="${BATCH}" \
  NBS="${NBS}" \
  DEVICE="${DEVICE}" \
  WORKERS="${WORKERS}" \
  PROJECT="${PROJECT}" \
  NAME="${NAME}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${TRAIN_SCRIPT}"
}

detector_weights() {
  local best="${PROJECT}/${NAME}/weights/best.pt"
  if [[ -f "${best}" ]]; then
    echo "${best}"
    return 0
  fi
  echo "Detector weights not found: ${best}" >&2
  return 1
}

run_yolo_val() {
  local weights="$1"
  local dataset_name="$2"
  local data_yaml="$3"
  local out_dir="${EVAL_ROOT}/detection_only/${dataset_name}"
  mkdir -p "${out_dir}"
  run_python - <<PY
import json
from pathlib import Path
from ultralytics import YOLO

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

model = YOLO(r"${weights}")
metrics = model.val(
    data=r"${data_yaml}",
    imgsz=int("${IMGSZ}"),
    batch=int("${EVAL_BATCH}"),
    device=r"${EVAL_DEVICE}",
    project=r"${out_dir}",
    name="val",
    exist_ok=True,
)
Path(r"${out_dir}/metrics.txt").write_text(str(metrics) + "\\n", encoding="utf-8")
summary = {
    "dataset": r"${dataset_name}",
    "weights": r"${weights}",
    "data": r"${data_yaml}",
    "imgsz": int("${IMGSZ}"),
    "results_dict": clean(getattr(metrics, "results_dict", {})),
    "speed": clean(getattr(metrics, "speed", {})),
    "fitness": clean(getattr(metrics, "fitness", None)),
}
Path(r"${out_dir}/metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
PY
}

run_detection_evals() {
  local weights="$1"
  local eval_yaml_root="${MERGED_YOLO_ROOT}/eval_yamls"
  run_yolo_val "${weights}" "anti_uav" "${eval_yaml_root}/anti_uav_rgb.yaml"
  run_yolo_val "${weights}" "hanlue_old" "${eval_yaml_root}/hanlue_old_rgb.yaml"
  run_yolo_val "${weights}" "hanlue_new" "${eval_yaml_root}/hanlue_new_rgb.yaml"
}

tracker_common_args=()
build_tracker_args() {
  tracker_common_args=(
    --tracker "${TRACKER}"
    --imgsz "${IMGSZ}"
    --device "${EVAL_DEVICE}"
    --conf 0.45
    --min-confidence 0.45
    --detector-assist-policy edtc_like
    --auto-confirm
    --presence-verifier pair_head_edl
    --presence-model "${PRESENCE_MODEL}"
  )
  if [[ "${TRACKER}" == "nanotrack" ]]; then
    tracker_common_args+=(
      --nanotrack-root "${NANOTRACK_ROOT}"
      --nanotrack-snapshot "${NANOTRACK_SNAPSHOT}"
      --nanotrack-device "${EVAL_DEVICE}"
    )
  fi
}

run_tracking_evals() {
  local weights="$1"
  build_tracker_args
  local limit_args=()
  if [[ "${TRACK_LIMIT}" != "0" ]]; then
    limit_args=(--limit "${TRACK_LIMIT}")
  fi

  run_python "${BATCH_REPLAY}" \
    --model "${weights}" \
    --dataset-root "${ANTIUAV_DATASET_ROOT}" \
    --split "${ANTI_TRACK_SPLIT}" \
    --modality rgb \
    --output-root "${EVAL_ROOT}/tracking/anti_uav" \
    "${limit_args[@]}" \
    "${tracker_common_args[@]}"

  run_python "${BATCH_TRACKER}" \
    --model "${weights}" \
    --tracker-root "${OLD_HANLUE_ROOT}/tracker_sequences" \
    --image-root "${OLD_HANLUE_ROOT}/images" \
    --split val \
    --output-root "${EVAL_ROOT}/tracking/hanlue_old" \
    "${limit_args[@]}" \
    "${tracker_common_args[@]}"

  local sequence_args=()
  while IFS= read -r seq_root; do
    sequence_args+=(--sequence-root "${seq_root}")
  done < <(find "${NEW_HANLUE_ROOT}" -mindepth 2 -maxdepth 2 -type d -name sequences | sort)
  run_python "${BATCH_TRACKER}" \
    --model "${weights}" \
    --output-root "${EVAL_ROOT}/tracking/hanlue_new" \
    "${sequence_args[@]}" \
    "${limit_args[@]}" \
    "${tracker_common_args[@]}"
}

if [[ "${MODE}" != "all" && "${MODE}" != "train" && "${MODE}" != "eval" ]]; then
  echo "Unsupported MODE=${MODE}; expected all, train, or eval." >&2
  exit 2
fi

if [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; then
  write_eval_yamls
  train_detector
fi

if [[ "${MODE}" == "all" || "${MODE}" == "eval" ]]; then
  write_eval_yamls
  weights="$(detector_weights)"
  if [[ "${RUN_DETECTION_EVAL}" == "1" ]]; then
    run_detection_evals "${weights}"
  fi
  if [[ "${RUN_TRACKING_EVAL}" == "1" ]]; then
    run_tracking_evals "${weights}"
  fi
fi
