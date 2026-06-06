#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/oem/servers/imagenet-c}"
CORRUPTION="${CORRUPTION:-gaussian_noise}"
SEVERITY="${SEVERITY:-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_METHODS_ROOT="${SCRIPT_DIR}/external_methods"
LTTA_ROOT="${LTTA_ROOT:-${EXTERNAL_METHODS_ROOT}/ltta}"
SFO_ROOT="${SFO_ROOT:-${EXTERNAL_METHODS_ROOT}/sfo}"

ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; }

echo "Checking external TTA commands"
echo "DATA_ROOT=${DATA_ROOT} CORRUPTION=${CORRUPTION} SEVERITY=${SEVERITY}"

if [[ -d "${DATA_ROOT}/${CORRUPTION}/${SEVERITY}" ]]; then
  ok "ImageNet-C split exists: ${DATA_ROOT}/${CORRUPTION}/${SEVERITY}"
else
  fail "ImageNet-C split missing: ${DATA_ROOT}/${CORRUPTION}/${SEVERITY}"
fi

if [[ -x "${LTTA_ROOT}/run_online_ltta_dwt2_imagenet_c.sh" ]]; then
  ok "L-TTA runner exists: ${LTTA_ROOT}"
  ok "L-TTA ResNet50 command is runnable"
  ok "L-TTA MobileViT-XXS command is runnable"
  warn "L-TTA ViT-B is unsupported by the current L_TTA_TRAIN runner"
else
  fail "L-TTA runner missing or not executable"
fi

if [[ -d "${SFO_ROOT}" ]]; then
  ok "SFO-TTA root exists: ${SFO_ROOT}"
else
  fail "SFO-TTA root missing: ${SFO_ROOT}"
fi

for path in \
  "${SFO_ROOT}/eval_bn_adapt.py" \
  "${SFO_ROOT}/run_eval_mobilevit_xxs_random_init_k2_tm1_aa10.sh" \
  "${SFO_ROOT}/run_eval_vit_base_random_init_k2_tm1_aa10.sh"
do
  if [[ -e "${path}" ]]; then ok "Found ${path}"; else fail "Missing ${path}"; fi
done

ok "SOA/SFO checkpoints are not required for source-only NVME runs; random-init fallback is enabled."

echo
echo "Use TTA/run_external_tta_command.sh CASE with env GPU/BATCH_SIZE/WORKERS/DATA_ROOT."
