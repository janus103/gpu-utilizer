#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

export LD_LIBRARY_PATH=""

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:${GPU}}"
DATA_CORRUPTION="${DATA_CORRUPTION:-/home/oem/servers/imagenet-c}"
CORRUPTION="${CORRUPTION:-gaussian_noise}"
LEVEL="${LEVEL:-5}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64,128}"
MODELS="${MODELS:-resnet50,vit_base_patch16_224,mobilevit_xxs}"
WORKERS="${WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
FORWARD_EQUIV_FACTOR="${FORWARD_EQUIV_FACTOR:-${ZOA_FORWARD_EQUIV_FACTOR:-2.0}}"
OUT="${OUT:-Results/TTA/zoa_fp16_imagenet_c_${CORRUPTION}_l${LEVEL}_gpu${GPU}.csv}"

ARGS=(
  --models "${MODELS}"
  --batch-sizes "${BATCH_SIZES}"
  --data-corruption "${DATA_CORRUPTION}"
  --corruption "${CORRUPTION}"
  --level "${LEVEL}"
  --gpu-index "${GPU}"
  --device "${DEVICE}"
  --workers "${WORKERS}"
  --forward-equiv-factor "${FORWARD_EQUIV_FACTOR}"
  --keep-going
  --output "${OUT}"
)

if [[ "${MAX_SAMPLES}" != "0" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${NO_IDLE_CHECK:-0}" == "1" ]]; then
  ARGS+=(--no-idle-check)
fi

python3 TTA/profile_zoa_fp16_stream.py "${ARGS[@]}"
