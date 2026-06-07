#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

RUN_ID="${NVME_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NVME_RESULTS_BASE="${NVME_RESULTS_BASE:-Results/TTA/nvme_runs}"
REQUESTED_RESULTS_DIR="${RESULTS_DIR:-${NVME_RESULTS_BASE}/${RUN_ID}}"
if [[ "${NVME_ALLOW_EXISTING_RESULTS_DIR:-0}" != "1" ]]; then
  CANDIDATE_RESULTS_DIR="${REQUESTED_RESULTS_DIR}"
  SUFFIX=1
  while [[ -e "${CANDIDATE_RESULTS_DIR}" ]]; do
    CANDIDATE_RESULTS_DIR="${REQUESTED_RESULTS_DIR}_${SUFFIX}"
    SUFFIX="$((SUFFIX + 1))"
  done
  export RESULTS_DIR="${CANDIDATE_RESULTS_DIR}"
else
  export RESULTS_DIR="${REQUESTED_RESULTS_DIR}"
fi
source TTA/configs/nvme_env.sh

mkdir -p "${RESULTS_DIR}/logs"
STATUS_CSV="${RESULTS_DIR}/orchestrator_status_${CORRUPTION}_l${LEVEL}_gpu${GPU}.csv"
MAX_STAGE_ATTEMPTS="${NVME_MAX_STAGE_ATTEMPTS:-0}"

printf 'stage,script,status,exit_code,attempt,log_path,started_at,ended_at\n' > "${STATUS_CSV}"

csv_append() {
  python3 - "$STATUS_CSV" "$@" <<'PY'
import csv
import sys

path = sys.argv[1]
row = sys.argv[2:]
with open(path, "a", newline="") as stream:
    csv.writer(stream).writerow(row)
PY
}

wait_before_stage() {
  local stage="$1"
  echo "[orchestrator] waiting ${NVME_PROCESS_GAP_SECONDS}s before stage=${stage}"
  sleep "${NVME_PROCESS_GAP_SECONDS}"
}

run_stage_once() {
  local stage="$1"
  local script="$2"
  local attempt="$3"
  local started_at
  local ended_at
  local log_path
  local exit_code
  local status

  started_at="$(date -Iseconds)"
  log_path="${RESULTS_DIR}/logs/${stage}_attempt${attempt}.log"
  wait_before_stage "${stage}"
  echo "[orchestrator] start stage=${stage} attempt=${attempt} log=${log_path}"

  set +e
  bash "${SCRIPT_DIR}/${script}" > "${log_path}" 2>&1
  exit_code="$?"
  set -e

  ended_at="$(date -Iseconds)"
  if [[ "${exit_code}" == "0" ]]; then
    status="ok"
  else
    status="error"
  fi
  csv_append "${stage}" "${script}" "${status}" "${exit_code}" "${attempt}" "${log_path}" "${started_at}" "${ended_at}"
  echo "[orchestrator] done stage=${stage} status=${status} exit=${exit_code}"
  return "${exit_code}"
}

STAGES=(
  "baseline:run_01_baseline_inference_fp16.sh"
  "tent_eata_sar:run_02_tent_eata_sar_fp16.sh"
  "zoa:run_03_zoa_fp16.sh"
  "ltta_sfo:run_04_ltta_sfo_source_only.sh"
)

declare -A ATTEMPTS=()
PENDING=("${STAGES[@]}")

echo "[orchestrator] run_id=${RUN_ID}"
echo "[orchestrator] results_dir=${RESULTS_DIR}"
echo "[orchestrator] gpu=${GPU} device=${DEVICE} gap_seconds=${NVME_PROCESS_GAP_SECONDS}"
echo "[orchestrator] max_stage_attempts=${MAX_STAGE_ATTEMPTS} (0 means unlimited)"

while [[ "${#PENDING[@]}" -gt 0 ]]; do
  NEXT_PENDING=()
  for item in "${PENDING[@]}"; do
    stage="${item%%:*}"
    script="${item#*:}"
    ATTEMPTS["${stage}"]="$(( ${ATTEMPTS["${stage}"]:-0} + 1 ))"
    attempt="${ATTEMPTS["${stage}"]}"

    if run_stage_once "${stage}" "${script}" "${attempt}"; then
      continue
    fi

    if [[ "${MAX_STAGE_ATTEMPTS}" != "0" && "${attempt}" -ge "${MAX_STAGE_ATTEMPTS}" ]]; then
      echo "[orchestrator] stage=${stage} reached max attempts=${MAX_STAGE_ATTEMPTS}; leaving failed."
      continue
    fi
    NEXT_PENDING+=("${item}")
  done

  PENDING=("${NEXT_PENDING[@]}")
  if [[ "${#PENDING[@]}" -gt 0 ]]; then
    echo "[orchestrator] pending stages remain: ${PENDING[*]}"
  fi
done

echo "[orchestrator] all stages completed or reached max attempts."
echo "[orchestrator] status csv: ${STATUS_CSV}"
