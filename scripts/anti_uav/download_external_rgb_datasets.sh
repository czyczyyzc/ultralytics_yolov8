#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/mnt/chenziye/datasets/anti_uav/external_rgb/raw}"
DOWNLOAD_DUT="${DOWNLOAD_DUT:-1}"
DOWNLOAD_HALMSTAD="${DOWNLOAD_HALMSTAD:-1}"

mkdir -p "${RAW_ROOT}"

download_gdrive() {
  local file_id="$1"
  local output_dir="$2"
  local marker="${output_dir}/.${file_id}.done"
  mkdir -p "${output_dir}"
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${file_id}"
    return
  fi
  if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is required for Google Drive downloads." >&2
    return 1
  fi
  echo "[gdown] ${file_id} -> ${output_dir}"
  gdown "${file_id}" -O "${output_dir}/"
  touch "${marker}"
}

extract_archive() {
  local archive="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  case "${archive,,}" in
    *.zip)
      unzip -q -n "${archive}" -d "${out_dir}"
      ;;
    *.tar|*.tar.gz|*.tgz)
      tar -xf "${archive}" -C "${out_dir}"
      ;;
    *.rar)
      if command -v unrar >/dev/null 2>&1; then
        unrar x -o+ "${archive}" "${out_dir}/"
      elif command -v 7z >/dev/null 2>&1; then
        7z x -y "-o${out_dir}" "${archive}"
      else
        echo "No unrar/7z found for ${archive}; leaving archive unextracted." >&2
      fi
      ;;
    *)
      echo "[skip extract] Unknown archive suffix: ${archive}"
      ;;
  esac
}

if [[ "${DOWNLOAD_DUT}" == "1" ]]; then
  DUT_ROOT="${RAW_ROOT}/dut_anti_uav"
  mkdir -p "${DUT_ROOT}/archives"
  # Source: https://github.com/wangdongdut/DUT-Anti-UAV
  download_gdrive "1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj" "${DUT_ROOT}/archives"
  download_gdrive "1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n" "${DUT_ROOT}/archives"
  download_gdrive "1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0" "${DUT_ROOT}/archives"
  download_gdrive "1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9" "${DUT_ROOT}/archives"
  download_gdrive "16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq" "${DUT_ROOT}/archives"
  for archive in "${DUT_ROOT}"/archives/*; do
    [[ -f "${archive}" ]] || continue
    extract_archive "${archive}" "${DUT_ROOT}/extracted"
  done
fi

if [[ "${DOWNLOAD_HALMSTAD}" == "1" ]]; then
  HALMSTAD_ROOT="${RAW_ROOT}/halmstad_drone_detection"
  mkdir -p "${HALMSTAD_ROOT}"
  if [[ ! -d "${HALMSTAD_ROOT}/repo/.git" ]]; then
    git clone --depth=1 https://github.com/DroneDetectionThesis/Drone-detection-dataset "${HALMSTAD_ROOT}/repo"
  else
    git -C "${HALMSTAD_ROOT}/repo" pull --ff-only
  fi
  echo "[info] Halmstad repo cloned. If release assets are not stored by Git, download them from:"
  echo "       https://github.com/DroneDetectionThesis/Drone-detection-dataset/releases/tag/v1.0.0"
  echo "       and place/extract them under ${HALMSTAD_ROOT}/extracted"
fi

echo "[done] RAW_ROOT=${RAW_ROOT}"
