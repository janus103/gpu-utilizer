#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

source TTA/configs/nvme_env.sh
export RESULTS_DIR="${RESULTS_DIR:-Results/TTA/nvme_runs/a100_remaining_retries_$(date +%Y%m%d_%H%M%S)}"
export WORKERS="${WORKERS:-0}"
mkdir -p "${RESULTS_DIR}/logs"

echo "[launch] RESULTS_DIR=${RESULTS_DIR}"
echo "[launch] WORKERS=${WORKERS}"
echo "[launch] GPU 0 and GPU 1 groups will run in parallel."
echo "[launch] Each GPU group runs only one task at a time."

bash "${SCRIPT_DIR}/run_a100_remaining_gpu0.sh" 2>&1 | tee "${RESULTS_DIR}/logs/gpu0_driver.log" &
pid0="$!"
bash "${SCRIPT_DIR}/run_a100_remaining_gpu1.sh" 2>&1 | tee "${RESULTS_DIR}/logs/gpu1_driver.log" &
pid1="$!"

echo "[launch] gpu0 driver pid=${pid0}"
echo "[launch] gpu1 driver pid=${pid1}"
echo "[launch] monitor: bash ${SCRIPT_DIR}/monitor_a100_remaining_retries.sh ${RESULTS_DIR}"

set +e
wait "${pid0}"
rc0="$?"
wait "${pid1}"
rc1="$?"
set -e

echo "[done] gpu0 rc=${rc0} gpu1 rc=${rc1}"
if [[ "${rc0}" != "0" || "${rc1}" != "0" ]]; then
  exit 1
fi
