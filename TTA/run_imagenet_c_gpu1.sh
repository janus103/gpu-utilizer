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

export LD_LIBRARY_PATH=""
mkdir -p Results/TTA/logs

COMMON_ARGS=(
  --pretrained
  --algorithms tent,eata
  --batch-sizes 1,2,4,8,16,32,64,128
  --input-mode imagenet-c
  --data-corruption "${DATA_CORRUPTION}"
  --corruption "${CORRUPTION}"
  --level "${LEVEL}"
  --fisher-samples "${FISHER_SAMPLES:-128}"
  --tta-forward-equiv-factor "${TTA_FORWARD_EQUIV_FACTOR:-4.0}"
  --keep-going
  --device cuda:1
  --gpu-index 1
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  COMMON_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

python3 TTA/profile_tta_stream.py \
  --model mobilenetv2_100 \
  --model-source timm \
  --image-size 224 \
  --adapt-param-types bn \
  --forward-gops-per-sample "${MOBILENETV2_100_GOPS:-0.600}" \
  --output "Results/TTA/mobilenetv2_100_imagenet_c_${CORRUPTION}_l${LEVEL}_gpu1.csv" \
  "${COMMON_ARGS[@]}"

python3 TTA/profile_tta_stream.py \
  --model vit_base_patch16_224 \
  --model-source timm \
  --image-size 224 \
  --adapt-param-types ln \
  --forward-gops-per-sample "${VIT_B_16_GOPS:-35.200}" \
  --output "Results/TTA/vit_base_patch16_224_imagenet_c_${CORRUPTION}_l${LEVEL}_gpu1.csv" \
  "${COMMON_ARGS[@]}"
