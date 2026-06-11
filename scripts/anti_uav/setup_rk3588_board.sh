#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${VENV_DIR:-/data/venvs/anti_uav_rk3588}"
NANOTRACK_ROOT="${NANOTRACK_ROOT:-/data/codes/NanoTrack_RK3588_python}"
MODEL_DIR="${MODEL_DIR:-/data/models/anti_uav}"
YOLO_ONNX_PATH="${YOLO_ONNX_PATH:-${MODEL_DIR}/yolov8n_rkopt.onnx}"
YOLO_ONNX_URL="${YOLO_ONNX_URL:-https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8n.onnx}"
RKNN_LITE_WHL_URL="${RKNN_LITE_WHL_URL:-https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl}"
RKNN_LITE_WHL="${RKNN_LITE_WHL:-${MODEL_DIR}/wheels/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.whl}"
DOWNLOAD_YOLO_ONNX="${DOWNLOAD_YOLO_ONNX:-1}"

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

OWNER_USER="${SUDO_USER:-${USER:-$(id -un)}}"
OWNER_GROUP="$(id -gn "${OWNER_USER}")"

${SUDO} mkdir -p /data /data/codes /data/venvs /data/models /data/tmp

if ! command -v pip3 >/dev/null 2>&1; then
  ${SUDO} apt-get update
  ${SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-venv python3-opencv python3-numpy git
fi

python3 -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy<2" yacs onnxruntime

mkdir -p "$(dirname "${RKNN_LITE_WHL}")"
if ! python -c 'import rknnlite.api' >/dev/null 2>&1; then
  curl -L --fail "${RKNN_LITE_WHL_URL}" -o "${RKNN_LITE_WHL}"
  python -m pip install "${RKNN_LITE_WHL}"
fi

if [[ ! -d "${NANOTRACK_ROOT}/models" ]]; then
  git clone --depth=1 https://github.com/Try2ChangeX/NanoTrack_RK3588_python "${NANOTRACK_ROOT}"
fi

mkdir -p "${MODEL_DIR}"
if [[ "${DOWNLOAD_YOLO_ONNX}" == "1" && ! -f "${YOLO_ONNX_PATH}" ]]; then
  curl -L --fail "${YOLO_ONNX_URL}" -o "${YOLO_ONNX_PATH}"
fi

chown -R "${OWNER_USER}:${OWNER_GROUP}" "${VENV_DIR}" "${MODEL_DIR}" "${NANOTRACK_ROOT}"

cat <<EOF

RK3588 board environment is ready.

Activate:
  source "${VENV_DIR}/bin/activate"

Default detector artifact:
  ${YOLO_ONNX_PATH}

NanoTrack RKNN runtime:
  ${NANOTRACK_ROOT}

Example run:
  python "${REPO_ROOT}/scripts/anti_uav/anti_uav_rk3588.py" \\
    --model "${YOLO_ONNX_PATH}" \\
    --source /path/to/video.mp4 \\
    --tracker nanotrack_rknn \\
    --nanotrack-root "${NANOTRACK_ROOT}" \\
    --class-names "${REPO_ROOT}/rknn_model_zoo/examples/yolov8/model/coco_80_labels_list.txt" \\
    --target-class-names airplane,bird \\
    --save-output "${MODEL_DIR}/preview.mp4"

Notes:
  - The board-side script can run detector .onnx or .rknn files.
  - Per the RKNN stack, final .rknn conversion belongs to the PC-side RKNN-Toolkit2 workflow; the board uses RKNN-Toolkit-Lite2 for inference.

EOF
