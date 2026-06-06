#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/imagenet-c [corruption] [level]" >&2
  exit 2
fi

DATA_CORRUPTION="$1"
CORRUPTION="${2:-gaussian_noise}"
LEVEL="${3:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

if [[ ! -d "${DATA_CORRUPTION}/${CORRUPTION}/${LEVEL}" ]]; then
  echo "ImageNet-C split not found: ${DATA_CORRUPTION}/${CORRUPTION}/${LEVEL}" >&2
  exit 1
fi

mkdir -p Results/TTA/logs
LOG0="Results/TTA/logs/gpu0_${CORRUPTION}_l${LEVEL}.log"
LOG1="Results/TTA/logs/gpu1_${CORRUPTION}_l${LEVEL}.log"

echo "Launching GPU0: ResNet50 -> MobileViT-XXS"
bash TTA/run_imagenet_c_gpu0.sh "${DATA_CORRUPTION}" "${CORRUPTION}" "${LEVEL}" >"${LOG0}" 2>&1 &
PID0=$!

echo "Launching GPU1: MobileNetV2 -> ViT-B"
bash TTA/run_imagenet_c_gpu1.sh "${DATA_CORRUPTION}" "${CORRUPTION}" "${LEVEL}" >"${LOG1}" 2>&1 &
PID1=$!

echo "GPU0 PID: ${PID0}, log: ${LOG0}"
echo "GPU1 PID: ${PID1}, log: ${LOG1}"
echo "Monitor with: bash TTA/monitor_tta_jobs.sh"

wait "${PID0}"
STATUS0=$?
wait "${PID1}"
STATUS1=$?

echo "GPU0 exit status: ${STATUS0}"
echo "GPU1 exit status: ${STATUS1}"

if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 ]]; then
  exit 1
fi
