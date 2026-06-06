#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

source TTA/configs/nvme_env.sh
export LD_LIBRARY_PATH=""
mkdir -p "${RESULTS_DIR}"

export SFO_ALLOW_RANDOM_INIT=1
export DATA_ROOT="${DATA_CORRUPTION}"
export SEVERITY="${LEVEL}"

# L-TTA scope intentionally excludes ViT-B.
BATCH_SIZE="${LTTA_BATCH_SIZE:-128}" bash TTA/run_external_tta_command.sh ltta_resnet50
BATCH_SIZE="${LTTA_BATCH_SIZE:-128}" bash TTA/run_external_tta_command.sh ltta_mobilevit_xxs

# SOA/SFO source-only random-init runs.
BATCH_SIZE="${SFO_BATCH_SIZE:-256}" bash TTA/run_external_tta_command.sh sfo_resnet50
BATCH_SIZE="${SFO_BATCH_SIZE:-256}" bash TTA/run_external_tta_command.sh sfo_mobilevit_xxs
BATCH_SIZE="${SFO_BATCH_SIZE:-256}" bash TTA/run_external_tta_command.sh sfo_vit_b
