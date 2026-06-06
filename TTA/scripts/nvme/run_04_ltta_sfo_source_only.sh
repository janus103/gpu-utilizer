#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

source TTA/configs/nvme_env.sh
export LD_LIBRARY_PATH=""
mkdir -p "${RESULTS_DIR}"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"
STATUS_CSV="${RESULTS_DIR}/external_ltta_sfo_status_${CORRUPTION}_l${LEVEL}_gpu${GPU}.csv"

export SFO_ALLOW_RANDOM_INIT=1
export DATA_ROOT="${DATA_CORRUPTION}"
export SEVERITY="${LEVEL}"

if [[ ! -f "${STATUS_CSV}" ]]; then
  printf 'case_name,method,model,precision,batch_size,gpu,workers,data_root,corruption,severity,status,failure_reason,exit_code,log_path,started_at,ended_at\n' > "${STATUS_CSV}"
fi

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

classify_failure() {
  local exit_code="$1"
  local log_path="$2"
  if [[ "${exit_code}" == "2" ]]; then
    echo "unsupported"
    return
  fi
  if grep -Eiq 'out of memory|cuda.*memory|cublas_status_alloc_failed|cudnn_status_alloc_failed|memoryerror' "${log_path}"; then
    echo "oom"
    return
  fi
  if grep -Eiq 'no such file|does not exist|not found|filenotfounderror|assertionerror|reader out ->' "${log_path}"; then
    echo "missing_data_or_file"
    return
  fi
  if [[ "${exit_code}" == "0" ]]; then
    echo ""
    return
  fi
  echo "error"
}

run_external_case() {
  local case_name="$1"
  local method="$2"
  local model="$3"
  local batch_size="$4"
  local started_at
  local ended_at
  local log_path
  local output_csv
  local exit_code
  local failure_reason
  local status

  started_at="$(date -Iseconds)"
  log_path="${LOG_DIR}/${case_name}_${CORRUPTION}_l${LEVEL}_bs${batch_size}_gpu${GPU}.log"
  output_csv="${RESULTS_DIR}/${case_name}_${CORRUPTION}_l${LEVEL}_bs${batch_size}_gpu${GPU}.csv"
  echo "[external] start case=${case_name} batch=${batch_size} log=${log_path}"

  set +e
  BATCH_SIZE="${batch_size}" OUTPUT_CSV="${output_csv}" bash TTA/run_external_tta_command.sh "${case_name}" > "${log_path}" 2>&1
  exit_code="$?"
  set -e

  ended_at="$(date -Iseconds)"
  failure_reason="$(classify_failure "${exit_code}" "${log_path}")"
  if [[ "${exit_code}" == "0" && -z "${failure_reason}" ]]; then
    status="ok"
  else
    status="error"
  fi

  csv_append \
    "${case_name}" "${method}" "${model}" "amp_fp16" "${batch_size}" "${GPU}" "${WORKERS}" \
    "${DATA_ROOT}" "${CORRUPTION}" "${SEVERITY}" "${status}" "${failure_reason}" "${exit_code}" \
    "${log_path}" "${started_at}" "${ended_at}"

  echo "[external] done case=${case_name} status=${status} reason=${failure_reason:-none} exit=${exit_code}"
}

# L-TTA scope intentionally excludes ViT-B.
run_external_case ltta_resnet50 ltta resnet50 "${LTTA_BATCH_SIZE:-128}"
run_external_case ltta_mobilevit_xxs ltta mobilevit_xxs "${LTTA_BATCH_SIZE:-128}"

# SOA/SFO source-only random-init runs.
run_external_case sfo_resnet50 sfo resnet50 "${SFO_BATCH_SIZE:-256}"
run_external_case sfo_mobilevit_xxs sfo mobilevit_xxs "${SFO_BATCH_SIZE:-256}"
run_external_case sfo_vit_b sfo vit_base_patch16_224 "${SFO_BATCH_SIZE:-256}"

echo "[external] status csv: ${STATUS_CSV}"
