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

run_tent_eata() {
  local model="$1"
  local image_size="$2"
  local adapt_types="$3"
  local gops="$4"
  python3 TTA/profile_tta_stream.py \
    --model "${model}" \
    --model-source timm \
    --pretrained \
    --no-baseline \
    --algorithms tent,eata \
    --batch-sizes "${BATCH_SIZES}" \
    --image-size "${image_size}" \
    --adapt-param-types "${adapt_types}" \
    --input-mode imagenet-c \
    --data-corruption "${DATA_CORRUPTION}" \
    --corruption "${CORRUPTION}" \
    --level "${LEVEL}" \
    --workers "${WORKERS}" \
    --fisher-samples "${FISHER_SAMPLES}" \
    --device "${DEVICE}" \
    --gpu-index "${GPU}" \
    --dtype "${PRECISION}" \
    --forward-gops-per-sample "${gops}" \
    --tta-forward-equiv-factor "${TTA_FORWARD_EQUIV_FACTOR}" \
    --output "${RESULTS_DIR}/${model}_tent_eata_fp16_${CORRUPTION}_l${LEVEL}_gpu${GPU}.csv" \
    --keep-going \
    "${COMMON_OPTIONAL[@]}"
}

run_sar() {
  local model_list="$1"
  python3 TTA/profile_sar_fp16_stream.py \
    --models "${model_list}" \
    --batch-sizes "${BATCH_SIZES}" \
    --data-corruption "${DATA_CORRUPTION}" \
    --corruption "${CORRUPTION}" \
    --level "${LEVEL}" \
    --workers "${WORKERS}" \
    --gpu-index "${GPU}" \
    --device "${DEVICE}" \
    --pretrained \
    --forward-equiv-factor "${TTA_FORWARD_EQUIV_FACTOR}" \
    --output "${RESULTS_DIR}/sar_fp16_${CORRUPTION}_l${LEVEL}_gpu${GPU}.csv" \
    --keep-going \
    "${COMMON_OPTIONAL[@]}"
}

run_tent_eata resnet50 224 bn 8.178
run_tent_eata mobilevit_xxs 256 bn 0.800
run_tent_eata vit_base_patch16_224 224 ln 35.200
run_sar "resnet50,mobilevit_xxs,vit_base_patch16_224"
