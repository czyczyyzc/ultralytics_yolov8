#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${REPO_ROOT}/third_party/SiamTrackers}"
NANOTRACK_ROOT="${NANOTRACK_ROOT:-${THIRD_PARTY_ROOT}/NanoTrack}"
BRANCH="${BRANCH:-master}"
BUILD_EXT="${BUILD_EXT:-0}"
DOWNLOAD_PRETRAINED="${DOWNLOAD_PRETRAINED:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
VARIANT="${VARIANT:-v2}"
ARCHIVE_URL="${ARCHIVE_URL:-https://github.com/HonglinChu/SiamTrackers/archive/refs/heads/${BRANCH}.zip}"
GIT_CLONE_TIMEOUT="${GIT_CLONE_TIMEOUT:-30s}"

case "${VARIANT}" in
  v1) PRETRAINED_NAME="nanotrackv1.pth" ;;
  v2) PRETRAINED_NAME="nanotrackv2.pth" ;;
  v3) PRETRAINED_NAME="nanotrackv3.pth" ;;
  *)
    echo "Unsupported VARIANT=${VARIANT}. Use v1, v2, or v3." >&2
    exit 1
    ;;
esac

PRETRAINED_URL="${PRETRAINED_URL:-https://github.com/HonglinChu/SiamTrackers/raw/${BRANCH}/NanoTrack/models/pretrained/${PRETRAINED_NAME}}"

mkdir -p "$(dirname "${THIRD_PARTY_ROOT}")"

if [[ ! -d "${THIRD_PARTY_ROOT}/.git" && ! -d "${NANOTRACK_ROOT}/nanotrack" ]]; then
  clone_cmd=(git clone --depth=1 --filter=blob:none --sparse --branch "${BRANCH}" https://github.com/HonglinChu/SiamTrackers "${THIRD_PARTY_ROOT}")
  if command -v timeout >/dev/null 2>&1; then
    clone_cmd=(timeout "${GIT_CLONE_TIMEOUT}" "${clone_cmd[@]}")
  fi
  if ! "${clone_cmd[@]}"; then
    rm -rf "${THIRD_PARTY_ROOT}"
    tmp_zip="$(mktemp /tmp/siamtrackers.XXXXXX.zip)"
    tmp_dir="$(mktemp -d /tmp/siamtrackers.XXXXXX)"
    curl -L "${ARCHIVE_URL}" -o "${tmp_zip}"
    unzip -q "${tmp_zip}" -d "${tmp_dir}"
    extracted_dir="$(find "${tmp_dir}" -maxdepth 1 -mindepth 1 -type d | head -n 1)"
    if [[ -z "${extracted_dir}" ]]; then
      echo "Unable to unpack SiamTrackers archive from ${ARCHIVE_URL}" >&2
      exit 1
    fi
    mv "${extracted_dir}" "${THIRD_PARTY_ROOT}"
    rm -f "${tmp_zip}"
    rm -rf "${tmp_dir}"
  fi
fi

if [[ -d "${THIRD_PARTY_ROOT}/.git" ]]; then
  git -C "${THIRD_PARTY_ROOT}" sparse-checkout init --cone
  git -C "${THIRD_PARTY_ROOT}" sparse-checkout set NanoTrack
  git -C "${THIRD_PARTY_ROOT}" checkout "${BRANCH}"
fi

if [[ ! -d "${NANOTRACK_ROOT}/nanotrack" ]]; then
  echo "NanoTrack workspace missing under ${NANOTRACK_ROOT}" >&2
  exit 1
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install yacs tensorboard tqdm
fi

if [[ "${DOWNLOAD_PRETRAINED}" == "1" ]]; then
  mkdir -p "${NANOTRACK_ROOT}/models/pretrained"
  if [[ ! -f "${NANOTRACK_ROOT}/models/pretrained/${PRETRAINED_NAME}" ]]; then
    curl -L "${PRETRAINED_URL}" -o "${NANOTRACK_ROOT}/models/pretrained/${PRETRAINED_NAME}"
  fi
fi

if [[ "${BUILD_EXT}" == "1" ]]; then
  (
    cd "${NANOTRACK_ROOT}"
    "${PYTHON_BIN}" -m pip install cython
    "${PYTHON_BIN}" setup.py build_ext --inplace
  )
fi

echo "Prepared NanoTrack workspace at ${NANOTRACK_ROOT}"
