#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage: TTA/run_external_tta_command.sh CASE

CASE:
  ltta_resnet50
  ltta_mobilevit_xxs
  ltta_vit_b              # currently unsupported by L_TTA_TRAIN script
  sfo_resnet50
  sfo_mobilevit_xxs
  sfo_vit_b

Common env:
  GPU=0 BATCH_SIZE=128 WORKERS=4 DATA_ROOT=/home/oem/servers/imagenet-c
  CORRUPTION=gaussian_noise SEVERITY=5
USAGE
  exit 2
fi

CASE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_METHODS_ROOT="${SCRIPT_DIR}/external_methods"
LTTA_ROOT="${LTTA_ROOT:-${EXTERNAL_METHODS_ROOT}/ltta}"
SFO_ROOT="${SFO_ROOT:-${EXTERNAL_METHODS_ROOT}/sfo}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-4}"
DATA_ROOT="${DATA_ROOT:-/home/oem/servers/imagenet-c}"
CORRUPTION="${CORRUPTION:-gaussian_noise}"
SEVERITY="${SEVERITY:-5}"
DEVICE="${DEVICE:-cuda}"

run_ltta() {
  local model="$1"
  cd "${LTTA_ROOT}"
  DATA_ROOT="${DATA_ROOT}" WORKERS="${WORKERS}" \
    ./run_online_ltta_dwt2_imagenet_c.sh "${GPU}" "${model}" "${CORRUPTION}" "${SEVERITY}" "${BATCH_SIZE}" "${LBATCH_RESET:-10}"
}

run_sfo_resnet() {
  cd "${SFO_ROOT}"
  local resume="${RESUME:-}"
  local resume_round="${RESUME_ROUND:-}"
  local resume_aug="${RESUME_AUG:-}"
  local extra_args=()
  [[ -n "${resume}" ]] && extra_args+=(--resume "${resume}")
  [[ -n "${resume_round}" ]] && extra_args+=(--resume-round "${resume_round}")
  [[ -n "${resume_aug}" ]] && extra_args+=(--resume-aug "${resume_aug}")
  [[ "${SFO_ALLOW_RANDOM_INIT:-1}" == "1" ]] && extra_args+=(--allow-random-init)
  CUDA_VISIBLE_DEVICES="${GPU}" python3 eval_bn_adapt.py \
    --model resnet50 \
    --data-root "${DATA_ROOT}" \
    --severity "${SEVERITY}" \
    --val-split validation \
    --num-classes 1000 \
    --input-size 3 224 224 \
    --vit-kernel-size 7 \
    --spatial-group-size 1 \
    --var-feature \
    --moe-channel \
    --lbatch 0 \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --amp \
    "${extra_args[@]}"
}

run_sfo_script() {
  local script="$1"
  cd "${SFO_ROOT}"
  GPU="${GPU}" DEVICE="${DEVICE}" BATCH_SIZE="${BATCH_SIZE}" WORKERS="${WORKERS}" DATA_ROOT="${DATA_ROOT}" SEVERITY="${SEVERITY}" \
    "./${script}"
}

case "${CASE}" in
  ltta_resnet50)
    run_ltta resnet50
    ;;
  ltta_mobilevit_xxs)
    run_ltta mobilevit_xxs
    ;;
  ltta_vit_b)
    echo "[UNSUPPORTED] L_TTA_TRAIN/run_online_ltta_dwt2_imagenet_c.sh reports ViT-B is not runnable in the current tree." >&2
    exit 2
    ;;
  sfo_resnet50)
    run_sfo_resnet
    ;;
  sfo_mobilevit_xxs)
    run_sfo_script run_eval_mobilevit_xxs_random_init_k2_tm1_aa10.sh
    ;;
  sfo_vit_b)
    run_sfo_script run_eval_vit_base_random_init_k2_tm1_aa10.sh
    ;;
  *)
    echo "[ERROR] Unknown CASE=${CASE}" >&2
    exit 2
    ;;
esac
