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
  --device cuda:0
  --gpu-index 0
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  COMMON_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

python3 TTA/profile_tta_stream.py \
  --model resnet50 \
  --model-source auto \
  --image-size 224 \
  --adapt-param-types bn \
  --forward-gops-per-sample "${RESNET50_GOPS:-8.178}" \
  --output "Results/TTA/resnet50_imagenet_c_${CORRUPTION}_l${LEVEL}_gpu0.csv" \
  "${COMMON_ARGS[@]}"

python3 TTA/profile_tta_stream.py \
  --model mobilevit_xxs \
  --model-source timm \
  --image-size 256 \
  --adapt-param-types bn \
  --forward-gops-per-sample "${MOBILEVIT_XXS_GOPS:-0.800}" \
  --output "Results/TTA/mobilevit_xxs_imagenet_c_${CORRUPTION}_l${LEVEL}_gpu0.csv" \
  "${COMMON_ARGS[@]}"
