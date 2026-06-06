#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

source TTA/configs/nvme_env.sh
export LD_LIBRARY_PATH=""
mkdir -p "${RESULTS_DIR}"

MODELS="${MODELS:-resnet50,vit_base_patch16_224,mobilevit_xxs}" \
OUT="${RESULTS_DIR}/zoa_fp16_${CORRUPTION}_l${LEVEL}_gpu${GPU}.csv" \
FORWARD_EQUIV_FACTOR="${ZOA_FORWARD_EQUIV_FACTOR}" \
bash TTA/run_zoa_fp16_imagenet_c_sweep.sh
