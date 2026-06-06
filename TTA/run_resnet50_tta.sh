#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${GPU_UTILIZER_DIR}"

# Prefer PyTorch's bundled CUDA/CuDNN libraries over older system CuDNN entries.
export LD_LIBRARY_PATH=""

python3 TTA/profile_tta.py \
  --model resnet50 \
  --model-source auto \
  --pretrained \
  --algorithms tent,eata \
  --batch-sizes 1,2,4,8,16,32,64,128 \
  --image-size 224 \
  --adapt-steps 1 \
  --fisher-samples 128 \
  --warmup 5 \
  --repeat 20 \
  --device cuda:0 \
  --gpu-index 0 \
  --input-mode synthetic \
  --output Results/TTA/resnet50_tent_eata_bs1_128.csv
