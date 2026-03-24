#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-${REPO_ROOT}/runs/visdrone/yolov8n_visdrone/weights/best.pt}"
DATA="${DATA:-${REPO_ROOT}/ultralytics/cfg/datasets/VisDrone.yaml}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-1}"
DEVICE="${DEVICE:-0}"
RKNN_DEVICE="${RKNN_DEVICE:-cpu}"
HALF="${HALF:-0}"
DYNAMIC="${DYNAMIC:-0}"
SIMPLIFY="${SIMPLIFY:-1}"
INT8="${INT8:-0}"
WORKSPACE="${WORKSPACE:-4}"
OPSET="${OPSET:-12}"

if [[ ! -f "${MODEL}" ]]; then
  echo "Model checkpoint not found: ${MODEL}" >&2
  exit 1
fi

if [[ ! -f "${DATA}" ]]; then
  echo "Dataset yaml not found: ${DATA}" >&2
  exit 1
fi

formats=("$@")
if [[ ${#formats[@]} -eq 0 ]]; then
  formats=("onnx")
fi

run_export() {
  local fmt="$1"
  local export_device="${DEVICE}"
  local half_bool="False"
  local dynamic_bool="False"
  local simplify_bool="False"
  local batch_value="${BATCH}"
  local imgsz_value="${IMGSZ}"
  local int8_value="${INT8}"
  local workspace_value="${WORKSPACE}"
  local preset_note=""
  local model_stem=""
  local model_dir=""
  local model_onnx=""
  local rknn_onnx=""

  if [[ "${HALF}" == "1" ]]; then
    half_bool="True"
  fi
  if [[ "${DYNAMIC}" == "1" ]]; then
    dynamic_bool="True"
  fi
  if [[ "${SIMPLIFY}" == "1" ]]; then
    simplify_bool="True"
  fi

  case "${fmt}" in
    onnx)
      echo "Exporting ONNX from ${MODEL}"
      ;;
    engine|trt|tensorrt)
      fmt="engine"
      echo "Exporting TensorRT engine from ${MODEL}"
      ;;
    trt_orin_nx|engine_orin_nx|orin_nx_trt)
      fmt="engine"
      half_bool="True"
      dynamic_bool="True"
      simplify_bool="True"
      int8_value="0"
      batch_value="8"
      workspace_value="2"
      preset_note="Applied Orin NX 16G TensorRT FP16 preset: dynamic=True, batch=8, workspace=2."
      echo "Exporting TensorRT engine for Jetson Orin NX 16G from ${MODEL}"
      ;;
    trt_orin_nx_int8|engine_orin_nx_int8|orin_nx_trt_int8)
      fmt="engine"
      half_bool="False"
      dynamic_bool="True"
      simplify_bool="True"
      int8_value="1"
      batch_value="8"
      workspace_value="2"
      preset_note="Applied Orin NX 16G TensorRT INT8 preset: dynamic=True, batch=8, workspace=2."
      echo "Exporting INT8 TensorRT engine for Jetson Orin NX 16G from ${MODEL}"
      ;;
    rknn)
      export_device="${RKNN_DEVICE}"
      echo "Exporting RKNN-optimized ONNX from ${MODEL}"
      ;;
    *)
      echo "Unsupported export format: ${fmt}" >&2
      echo "Supported formats: onnx, engine, trt, tensorrt, trt_orin_nx, trt_orin_nx_int8, rknn" >&2
      exit 1
      ;;
  esac

  echo "  IMGSZ=${imgsz_value}"
  echo "  BATCH=${batch_value}"
  echo "  DEVICE=${export_device}"
  echo "  HALF=$([[ "${half_bool}" == "True" ]] && echo 1 || echo 0)"
  echo "  DYNAMIC=$([[ "${dynamic_bool}" == "True" ]] && echo 1 || echo 0)"
  echo "  SIMPLIFY=$([[ "${simplify_bool}" == "True" ]] && echo 1 || echo 0)"
  echo "  INT8=${int8_value}"
  echo "  WORKSPACE=${workspace_value}"
  echo "  OPSET=${OPSET}"
  if [[ -n "${preset_note}" ]]; then
    echo "  NOTE=${preset_note}"
  fi
  if [[ "${fmt}" == "engine" && "${dynamic_bool}" == "True" && "${workspace_value}" -gt 2 ]]; then
    echo "  WARNING=In this repo, dynamic TensorRT max shape scales with workspace. Large workspace may make engine build much heavier."
  fi

  if command -v yolo >/dev/null 2>&1; then
    export_args=(
      export
      model="${MODEL}"
      format="${fmt}"
      imgsz="${imgsz_value}"
      batch="${batch_value}"
      device="${export_device}"
      opset="${OPSET}"
      half="${half_bool}"
      dynamic="${dynamic_bool}"
      simplify="${simplify_bool}"
    )

    if [[ "${fmt}" == "engine" ]]; then
      export_args+=(workspace="${workspace_value}")
      if [[ "${int8_value}" == "1" ]]; then
        export_args+=(int8=True data="${DATA}")
      fi
    fi

    yolo "${export_args[@]}"
  else
    echo "'yolo' command not found, falling back to Python API."
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<PY
from ultralytics import YOLO

kwargs = {
    "format": "${fmt}",
    "imgsz": int("${imgsz_value}"),
    "batch": int("${batch_value}"),
    "device": "${export_device}",
    "opset": int("${OPSET}"),
    "half": ${half_bool},
    "dynamic": ${dynamic_bool},
    "simplify": ${simplify_bool},
}
if "${fmt}" == "engine":
    kwargs["workspace"] = int("${workspace_value}")
    if "${int8_value}" == "1":
        kwargs["int8"] = True
        kwargs["data"] = r"${DATA}"

model = YOLO(r"${MODEL}")
path = model.export(**kwargs)
print(path)
PY
  fi

  if [[ "${fmt}" == "rknn" ]]; then
    model_dir="$(dirname "${MODEL}")"
    model_stem="$(basename "${MODEL}")"
    model_stem="${model_stem%.*}"
    model_onnx="${model_dir}/${model_stem}.onnx"
    rknn_onnx="${model_dir}/${model_stem}_rknnopt.onnx"
    if [[ -f "${model_onnx}" ]]; then
      mv -f "${model_onnx}" "${rknn_onnx}"
      echo "  Renamed RKNN export to: ${rknn_onnx}"
    fi

    echo
    echo "RKNN note:"
    echo "  This repo exports an RKNN-optimized .onnx for RKNN-Toolkit / RKNN-Toolkit2."
    echo "  Saved as ${rknn_onnx}"
    echo "  Follow ${REPO_ROOT}/RKOPT_README.zh-CN.md and RKNN_Model_Zoo to generate the final .rknn."
  fi

  if [[ "${fmt}" == "engine" ]]; then
    echo
    echo "TensorRT note:"
    echo "  Export TensorRT on the same JetPack / TensorRT major version as the target device when possible."
    echo "  For Jetson Orin NX 16G, FP16 is usually the safest first choice."
  fi
}

for fmt in "${formats[@]}"; do
  run_export "${fmt}"
done
