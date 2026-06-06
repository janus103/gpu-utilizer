#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
MODEL_KEY="${2:-resnet50}"
CORRUPTION="${3:-gaussian_noise}"
SEVERITY="${4:-5}"
BATCH_SIZE="${5:-128}"
LBATCH_RESET="${6:-10}"
DATA_ROOT="${DATA_ROOT:-/home/oem/servers/imagenet-c}"
EXPERIMENT="${EXPERIMENT:-online_ltta_${MODEL_KEY}_dwt2_lb${LBATCH_RESET}}"
WORKERS="${WORKERS:-5}"
LTTA_AMP="${LTTA_AMP:-1}"

case "${MODEL_KEY}" in
  resnet50)
    MODEL="resnet50_dwt_se"
    INPUT_SIZE="3 224 224"
    LR="0.05"
    ;;
  mobilevit_xxs|mobilevit-xxs)
    MODEL="mobilevit_xxs_dwt"
    INPUT_SIZE="3 256 256"
    LR="0.05"
    ;;
  vit_b|vit-b|vit_base)
    echo "ViT-B is not runnable in the current L_TTA_TRAIN tree: PatchEmbed.init_weights is missing." >&2
    exit 2
    ;;
  *)
    echo "Unknown MODEL_KEY=${MODEL_KEY}. Use resnet50 or mobilevit_xxs." >&2
    exit 2
    ;;
esac

run_one() {
  local corruption="$1"
  local data_dir="${DATA_ROOT}/${corruption}/${SEVERITY}"
  local amp_args=()
  if [[ "${LTTA_AMP}" == "1" ]]; then
    amp_args+=(--amp --amp-dtype float16)
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" python train_dwt_tta_se_online.py "${data_dir}" \
    --model "${MODEL}" \
    --num-classes 1000 \
    --input-size ${INPUT_SIZE} \
    --dwt-kernel-size 3 3 3 \
    --dwt_level 2 2 2 \
    --dwt_bn 1 81 0 \
    --lr "${LR}" \
    --epochs 1 \
    -b "${BATCH_SIZE}" \
    -j "${WORKERS}" \
    --val-split val \
    --no-prefetcher \
    --ada 1 \
    --lbatch "${LBATCH_RESET}" \
    --experiment "${EXPERIMENT}_${corruption}" \
    "${amp_args[@]}"
}

if [[ "${CORRUPTION}" == "all" ]]; then
  for corruption in \
    gaussian_noise shot_noise impulse_noise \
    defocus_blur glass_blur motion_blur zoom_blur \
    snow frost fog brightness \
    contrast elastic_transform pixelate jpeg_compression
  do
    run_one "${corruption}"
  done
else
  run_one "${CORRUPTION}"
fi
