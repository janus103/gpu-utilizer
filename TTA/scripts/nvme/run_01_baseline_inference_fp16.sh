#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

source TTA/configs/nvme_env.sh
export LD_LIBRARY_PATH=""
mkdir -p "${RESULTS_DIR}"
COMMON_OPTIONAL=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  COMMON_OPTIONAL+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${NO_IDLE_CHECK}" == "1" ]]; then
  COMMON_OPTIONAL+=(--no-idle-check)
fi
wait_before_process() {
  local label="$1"
  echo "[nvme] waiting ${NVME_PROCESS_GAP_SECONDS}s before ${label}"
  sleep "${NVME_PROCESS_GAP_SECONDS}"
}

run_model() {
  local model="$1"
  local image_size="$2"
  local gops="$3"
  wait_before_process "baseline ${model}"
  python3 TTA/profile_tta_stream.py \
    --model "${model}" \
    --model-source timm \
    --pretrained \
    --algorithms "" \
    --batch-sizes "${BATCH_SIZES}" \
    --image-size "${image_size}" \
    --input-mode imagenet-c \
    --data-corruption "${DATA_CORRUPTION}" \
    --corruption "${CORRUPTION}" \
    --level "${LEVEL}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" \
    --gpu-index "${GPU}" \
    --dtype "${PRECISION}" \
    --forward-gops-per-sample "${gops}" \
    --output "${RESULTS_DIR}/${model}_baseline_fp16_${CORRUPTION}_l${LEVEL}_gpu${GPU}.csv" \
    --keep-going \
    "${COMMON_OPTIONAL[@]}"
}

run_model resnet50 224 8.178
run_model mobilevit_xxs 256 0.800
run_model vit_base_patch16_224 224 35.200
