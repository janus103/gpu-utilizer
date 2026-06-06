#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

source TTA/configs/nvme_env.sh
export LD_LIBRARY_PATH=""
mkdir -p "${RESULTS_DIR}"

ARGS=(
  --models "${MODELS:-resnet50,mobilevit_xxs,vit_base_patch16_224}"
  --batch-sizes "${BATCH_SIZES}"
  --data-corruption "${DATA_CORRUPTION}"
  --corruption "${CORRUPTION}"
  --level "${LEVEL}"
  --gpu-index "${GPU}"
  --device "${DEVICE}"
  --workers "${WORKERS}"
  --pretrained
  --keep-going
  --output "${RESULTS_DIR}/a100_fp32_training_vs_inference_${CORRUPTION}_l${LEVEL}_gpu${GPU}.csv"
)

if [[ "${MAX_SAMPLES}" != "0" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${NO_IDLE_CHECK}" == "1" ]]; then
  ARGS+=(--no-idle-check)
fi

python3 TTA/profile_fp32_train_vs_infer.py "${ARGS[@]}"
