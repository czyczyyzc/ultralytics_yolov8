#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXPORT_PYTHON="${EXPORT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONDA_BIN="${CONDA_BIN:-conda}"
RKNN_ENV="${RKNN_ENV:-rknn140}"
RKNN_WHEEL="${RKNN_WHEEL:-}"
INSTALL_MINIMAL_DEPS="${INSTALL_MINIMAL_DEPS:-1}"

MODEL="${MODEL:-${REPO_ROOT}/runs/anti_uav/yolov8n_anti_uav300_rgb_8gpu_b128_e50_nbs128/weights/best.pt}"
IMGSZ="${IMGSZ:-960}"
TARGET="${TARGET:-rk3588}"
QUANTIZE="${QUANTIZE:-0}"

MODEL_DIR="$(cd "$(dirname "${MODEL}")" && pwd)"
MODEL_STEM="$(basename "${MODEL}")"
MODEL_STEM="${MODEL_STEM%.*}"

ONNX_OUT="${ONNX_OUT:-${MODEL_DIR}/${MODEL_STEM}_rknnopt.onnx}"
LEGACY_ONNX_OUT="${LEGACY_ONNX_OUT:-${MODEL_DIR}/${MODEL_STEM}_rknnopt_v140.onnx}"
RKNN_OUT="${RKNN_OUT:-${MODEL_DIR}/${MODEL_STEM}_v140_fp.rknn}"

CALIB_SOURCE="${CALIB_SOURCE:-}"
CALIB_DIR="${CALIB_DIR:-${MODEL_DIR}/calibration_${MODEL_STEM}}"
DATASET_TXT="${DATASET_TXT:-${CALIB_DIR}/dataset.txt}"
CALIB_MAX_FRAMES="${CALIB_MAX_FRAMES:-64}"
CALIB_FRAME_STEP="${CALIB_FRAME_STEP:-30}"

if [[ ! -f "${MODEL}" ]]; then
  echo "Model checkpoint not found: ${MODEL}" >&2
  exit 1
fi

if [[ ! -x "${EXPORT_PYTHON}" ]]; then
  echo "Export python not found or not executable: ${EXPORT_PYTHON}" >&2
  exit 1
fi

echo "==> Export RKNN-compatible ONNX"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${EXPORT_PYTHON}" - <<PY
from ultralytics import YOLO
model = YOLO(r"${MODEL}")
path = model.export(format="rknn", imgsz=int("${IMGSZ}"), batch=1, device="cpu", opset=12, simplify=False, dynamic=False)
print(path)
PY

if [[ -f "${MODEL_DIR}/${MODEL_STEM}.onnx" ]]; then
  mv -f "${MODEL_DIR}/${MODEL_STEM}.onnx" "${ONNX_OUT}"
fi

if [[ ! -f "${ONNX_OUT}" ]]; then
  echo "Expected ONNX output not found: ${ONNX_OUT}" >&2
  exit 1
fi

echo "==> Sanitize ONNX for RKNN v1.4.0"
"${EXPORT_PYTHON}" "${REPO_ROOT}/scripts/anti_uav/sanitize_onnx_for_rknn_legacy.py" \
  --input "${ONNX_OUT}" \
  --output "${LEGACY_ONNX_OUT}"

if [[ ! -f "${LEGACY_ONNX_OUT}" ]]; then
  echo "Expected sanitized ONNX output not found: ${LEGACY_ONNX_OUT}" >&2
  exit 1
fi

echo "==> Ensure RKNN legacy environment"
CONDA_EXE="$(command -v "${CONDA_BIN}" || true)"
if [[ -z "${CONDA_EXE}" ]]; then
  echo "Unable to locate conda executable: ${CONDA_BIN}" >&2
  exit 1
fi

eval "$("${CONDA_EXE}" shell.bash hook)"
if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${RKNN_ENV}"; then
  "${CONDA_EXE}" create --override-channels -c defaults -y -n "${RKNN_ENV}" python=3.8
fi
conda activate "${RKNN_ENV}"

if ! python - <<'PY' >/dev/null 2>&1
import rknn
from rknn.api import RKNN
print(getattr(rknn, "__version__", "unknown"))
PY
then
  if [[ -z "${RKNN_WHEEL}" || ! -f "${RKNN_WHEEL}" ]]; then
    echo "RKNN_WHEEL must point to the v1.4.0 cp38 wheel when rknn is missing" >&2
    exit 1
  fi

  if [[ "${INSTALL_MINIMAL_DEPS}" == "1" ]]; then
    python -m pip install --progress-bar off \
      "numpy==1.19.5" \
      "protobuf==3.12.2" \
      "flatbuffers==1.12" \
      "requests==2.27.1" \
      "psutil==5.9.0" \
      "ruamel.yaml==0.17.4" \
      "scipy==1.5.4" \
      "tqdm==4.64.0" \
      "bfloat16==1.1" \
      "opencv-python==4.5.5.64" \
      "onnx==1.9.0" \
      "onnxoptimizer==0.2.7" \
      "onnxruntime==1.10.0"
  fi

  # Old RKNN wheels ship metadata that newer pip/setuptools refuse to parse, so unpack manually.
  python - <<PY
from pathlib import Path
import shutil
import sysconfig
import zipfile

wheel = Path(r"${RKNN_WHEEL}").resolve()
site = Path(sysconfig.get_paths()["purelib"]).resolve()
old_dist = site / "rknn_toolkit2-1.4.0_22dcfef4.dist-info"
new_dist = site / "rknn_toolkit2-1.4.0.post0.dist-info"

shutil.rmtree(site / "rknn", ignore_errors=True)
shutil.rmtree(old_dist, ignore_errors=True)
shutil.rmtree(new_dist, ignore_errors=True)

with zipfile.ZipFile(wheel) as archive:
    archive.extractall(site)

if old_dist.exists():
    old_dist.rename(new_dist)

meta = new_dist / "METADATA"
text = meta.read_text(encoding="utf-8")
text = text.replace("Version: 1.4.0-22dcfef4", "Version: 1.4.0.post0")
meta.write_text(text, encoding="utf-8")

record = new_dist / "RECORD"
if record.exists():
    record_text = record.read_text(encoding="utf-8")
    record_text = record_text.replace(
        "rknn_toolkit2-1.4.0_22dcfef4.dist-info/",
        "rknn_toolkit2-1.4.0.post0.dist-info/",
    )
    record.write_text(record_text, encoding="utf-8")
PY
fi

echo "==> RKNN environment"
python - <<'PY'
import rknn
from rknn.api import RKNN
print("rknn_version:", getattr(rknn, "__version__", "unknown"))
print("RKNN API:", RKNN)
PY

BUILD_ARGS=(
  "${REPO_ROOT}/scripts/anti_uav/build_rknn.py"
  --onnx "${LEGACY_ONNX_OUT}"
  --output "${RKNN_OUT}"
  --target "${TARGET}"
)

if [[ "${QUANTIZE}" == "1" ]]; then
  if [[ -z "${CALIB_SOURCE}" ]]; then
    echo "CALIB_SOURCE is required when QUANTIZE=1" >&2
    exit 1
  fi
  echo "==> Prepare calibration dataset"
  python "${REPO_ROOT}/scripts/anti_uav/prepare_rknn_calibration.py" \
    --source "${CALIB_SOURCE}" \
    --output-dir "${CALIB_DIR}" \
    --dataset-txt "${DATASET_TXT}" \
    --max-frames "${CALIB_MAX_FRAMES}" \
    --frame-step "${CALIB_FRAME_STEP}" \
    --width "${IMGSZ}" \
    --height "${IMGSZ}"
  BUILD_ARGS+=(--quantize --dataset "${DATASET_TXT}")
fi

echo "==> Build RKNN"
python "${BUILD_ARGS[@]}"

echo
echo "Generated artifacts:"
echo "  ONNX : ${ONNX_OUT}"
echo "  V140 : ${LEGACY_ONNX_OUT}"
echo "  RKNN : ${RKNN_OUT}"
if [[ "${QUANTIZE}" == "1" ]]; then
  echo "  CALIB: ${DATASET_TXT}"
fi
