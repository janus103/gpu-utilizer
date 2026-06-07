#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpu-id> <gpu0|gpu1>" >&2
  exit 2
fi

GPU="$1"
GROUP="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

source TTA/configs/nvme_env.sh
export LD_LIBRARY_PATH=""
export GPU
export WORKERS="${WORKERS:-1}"
export ZOA_WORKERS="${ZOA_WORKERS:-4}"
export NVME_PROCESS_GAP_SECONDS="${NVME_PROCESS_GAP_SECONDS:-10}"
export RESULTS_DIR="${RESULTS_DIR:-Results/TTA/nvme_runs/a100_remaining_retries_$(date +%Y%m%d_%H%M%S)}"
export DATA_CORRUPTION="${DATA_CORRUPTION:-/home/oem/servers/imagenet-c}"
export DATA_ROOT="${DATA_ROOT:-${DATA_CORRUPTION}}"
export CORRUPTION="${CORRUPTION:-gaussian_noise}"
export LEVEL="${LEVEL:-5}"
export SEVERITY="${SEVERITY:-${LEVEL}}"
export ZOA_FORWARD_EQUIV_FACTOR="${ZOA_FORWARD_EQUIV_FACTOR:-2.0}"

mkdir -p "${RESULTS_DIR}/logs"
STATUS_CSV="${RESULTS_DIR}/a100_remaining_${GROUP}_status_${CORRUPTION}_l${LEVEL}_gpu${GPU}.csv"
LOCK_FILE="/tmp/tetc_a100_remaining_gpu${GPU}.lock"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[error] another retry script is already running for GPU ${GPU}; lock=${LOCK_FILE}" >&2
  exit 1
fi

if [[ ! -f "${STATUS_CSV}" ]]; then
  printf 'case_name,method,model,batch_size,gpu,workers,status,failure_reason,exit_code,output_csv,log_path,started_at,ended_at,elapsed_s\n' > "${STATUS_CSV}"
fi

case "${GROUP}" in
  gpu0)
    TASKS=(
      "zoa_original|zoa_resnet50|resnet50|64"
      "external|ltta_resnet50|ltta|resnet50|64"
      "external|sfo_resnet50|sfo|resnet50|1"
    )
    ;;
  gpu1)
    TASKS=(
      "external|ltta_mobilevit_xxs|ltta|mobilevit_xxs|64"
      "external|sfo_mobilevit_xxs|sfo|mobilevit_xxs|1"
      "external|sfo_vit_b|sfo|vit_base_patch16_224|1"
    )
    ;;
  *)
    echo "Unknown group: ${GROUP}" >&2
    exit 2
    ;;
esac

append_status() {
  python3 - "$STATUS_CSV" "$@" <<'PY'
import csv
import sys

path = sys.argv[1]
row = sys.argv[2:]
with open(path, "a", newline="") as stream:
    csv.writer(stream).writerow(row)
PY
}

latest_status() {
  local case_name="$1"
  python3 - "$STATUS_CSV" "$case_name" <<'PY'
import csv
import sys

path, case_name = sys.argv[1], sys.argv[2]
try:
    with open(path, newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("case_name") == case_name]
except FileNotFoundError:
    rows = []
print(rows[-1].get("status", "") if rows else "")
PY
}

progress_bar() {
  local done="$1"
  local total="$2"
  local width=24
  local filled=$(( done * width / total ))
  local empty=$(( width - filled ))
  printf '['
  printf '%*s' "${filled}" '' | tr ' ' '#'
  printf '%*s' "${empty}" '' | tr ' ' '-'
  printf '] %d/%d' "${done}" "${total}"
}

wait_for_gpu_idle() {
  local label="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[warn] nvidia-smi not found; skipping idle wait for GPU ${GPU}"
    return
  fi

  while true; do
    local pids
    set +e
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | sed '/^$/d')"
    local rc="$?"
    set -e
    if [[ "${rc}" != "0" ]]; then
      echo "[warn] nvidia-smi query failed; skipping idle wait for GPU ${GPU}"
      return
    fi
    if [[ -z "${pids}" ]]; then
      return
    fi
    echo "[wait] GPU ${GPU} busy before ${label}; active pids: ${pids//$'\n'/, }"
    sleep 30
  done
}

classify_log_failure() {
  local exit_code="$1"
  local log_path="$2"
  python3 - "$exit_code" "$log_path" <<'PY'
import pathlib
import re
import sys

exit_code = int(sys.argv[1])
log_path = pathlib.Path(sys.argv[2])
text = log_path.read_text(errors="ignore").lower() if log_path.exists() else ""
if exit_code == 0:
    print("")
elif re.search(r"out of memory|cuda.*memory|cublas_status_alloc_failed|cudnn_status_alloc_failed|memoryerror", text):
    print("oom")
elif re.search(r"no such file|does not exist|not found|filenotfounderror|assertionerror", text):
    print("missing_data_or_file")
else:
    print("error")
PY
}

write_zoa_original_csv() {
  local output_csv="$1"
  local log_path="$2"
  local case_name="$3"
  local model="$4"
  local batch_size="$5"
  python3 - "$output_csv" "$log_path" "$case_name" "$model" "$batch_size" "$GPU" "$WORKERS" <<'PY'
import csv
import pathlib
import re
import sys

output_csv, log_path, case_name, model, batch_size, gpu, workers = sys.argv[1:]
text = pathlib.Path(log_path).read_text(errors="ignore")
matches = re.findall(
    r"Under shift type\s+(\S+)\s+After\s+(\S+)\s+Top-1 Accuracy:\s+([0-9.]+)\s+and\s+Top-5 Accuracy:\s+([0-9.]+)",
    text,
)
path = pathlib.Path(output_csv)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=["case_name", "model", "algorithm", "batch_size", "gpu", "workers", "corruption", "top1", "top5"],
    )
    writer.writeheader()
    for corruption, algorithm, top1, top5 in matches:
        writer.writerow(
            {
                "case_name": case_name,
                "model": model,
                "algorithm": algorithm,
                "batch_size": batch_size,
                "gpu": gpu,
                "workers": workers,
                "corruption": corruption,
                "top1": top1,
                "top5": top5,
            }
        )
PY
}

zoa_output_status() {
  local output_csv="$1"
  python3 - "$output_csv" <<'PY'
import csv
import sys

path = sys.argv[1]
try:
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
except FileNotFoundError:
    print("missing_output")
    raise SystemExit
if not rows:
    print("missing_output")
elif all(row.get("status") == "ok" for row in rows):
    print("ok")
else:
    reasons = sorted({row.get("failure_reason") or "error" for row in rows if row.get("status") != "ok"})
    print(",".join(reasons) if reasons else "error")
PY
}

run_zoa_case() {
  local case_name="$1"
  local model="$2"
  local batch_size="$3"
  local started_at ended_at start_s end_s elapsed_s output_csv log_path exit_code status failure_reason

  output_csv="${RESULTS_DIR}/${case_name}_${CORRUPTION}_l${LEVEL}_bs${batch_size}_gpu${GPU}.csv"
  log_path="${RESULTS_DIR}/logs/${case_name}_${CORRUPTION}_l${LEVEL}_bs${batch_size}_gpu${GPU}.log"
  started_at="$(date -Iseconds)"
  start_s="$(date +%s)"

  echo "[start] ${case_name}: ZOA model=${model} batch=${batch_size} gpu=${GPU} workers=${WORKERS}"
  wait_for_gpu_idle "${case_name}"
  sleep "${NVME_PROCESS_GAP_SECONDS}"

  set +e
  PYTHONUNBUFFERED=1 python3 TTA/profile_zoa_fp16_stream.py \
    --models "${model}" \
    --batch-sizes "${batch_size}" \
    --data-corruption "${DATA_CORRUPTION}" \
    --corruption "${CORRUPTION}" \
    --level "${LEVEL}" \
    --gpu-index "${GPU}" \
    --device "cuda:${GPU}" \
    --workers "${WORKERS}" \
    --forward-equiv-factor "${ZOA_FORWARD_EQUIV_FACTOR}" \
    --keep-going \
    --output "${output_csv}" \
    > "${log_path}" 2>&1
  exit_code="$?"
  set -e

  failure_reason="$(zoa_output_status "${output_csv}")"
  if [[ "${exit_code}" == "0" && "${failure_reason}" == "ok" ]]; then
    status="ok"
    failure_reason=""
  else
    status="error"
    [[ "${failure_reason}" == "ok" ]] && failure_reason="$(classify_log_failure "${exit_code}" "${log_path}")"
  fi

  ended_at="$(date -Iseconds)"
  end_s="$(date +%s)"
  elapsed_s="$(( end_s - start_s ))"
  append_status "${case_name}" "zoa" "${model}" "${batch_size}" "${GPU}" "${WORKERS}" "${status}" "${failure_reason}" "${exit_code}" "${output_csv}" "${log_path}" "${started_at}" "${ended_at}" "${elapsed_s}"
  echo "[done] ${case_name}: status=${status} reason=${failure_reason:-none} elapsed=${elapsed_s}s"
}

run_zoa_original_case() {
  local case_name="$1"
  local model="$2"
  local batch_size="$3"
  local started_at ended_at start_s end_s elapsed_s output_csv log_path exit_code status failure_reason

  if [[ "${model}" != "resnet50" ]]; then
    echo "[error] original ZOA path is currently wired for resnet50 only, got ${model}" >&2
    exit 2
  fi

  output_csv="${RESULTS_DIR}/${case_name}_original_${CORRUPTION}_l${LEVEL}_bs${batch_size}_gpu${GPU}.csv"
  log_path="${RESULTS_DIR}/logs/${case_name}_original_${CORRUPTION}_l${LEVEL}_bs${batch_size}_gpu${GPU}.log"
  started_at="$(date -Iseconds)"
  start_s="$(date +%s)"

  echo "[start] ${case_name}: original ZOA main.py model=${model} batch=${batch_size} gpu=${GPU} workers=${ZOA_WORKERS}"
  wait_for_gpu_idle "${case_name}"
  sleep "${NVME_PROCESS_GAP_SECONDS}"

  set +e
  (
    # Conda activation hooks in this environment reference unset MKL vars.
    # Keep the parent script strict, but disable nounset only while activating.
    set +u
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate robust
    set -u
    cd "${GPU_UTILIZER_DIR}/TTA/external_methods/zoa"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python main.py \
      --arch resnet50 \
      --algorithm zoa_resnet \
      --corruption "${CORRUPTION}" \
      --level "${LEVEL}" \
      --batch_size "${batch_size}" \
      --workers "${ZOA_WORKERS}" \
      --rounds 1 \
      --tag "${case_name}_original_${CORRUPTION}_l${LEVEL}_bs${batch_size}_gpu${GPU}" \
      --use_in1k_norm \
      --use_in1k_norm_c \
      --lr 0.0001 \
      --sc 0.01 \
      --lambda_bp 1 \
      --domain_t 0.2
  ) > "${log_path}" 2>&1
  exit_code="$?"
  set -e

  failure_reason="$(classify_log_failure "${exit_code}" "${log_path}")"
  if [[ "${exit_code}" == "0" && -z "${failure_reason}" ]]; then
    status="ok"
    write_zoa_original_csv "${output_csv}" "${log_path}" "${case_name}" "${model}" "${batch_size}"
  else
    status="error"
  fi

  ended_at="$(date -Iseconds)"
  end_s="$(date +%s)"
  elapsed_s="$(( end_s - start_s ))"
  append_status "${case_name}" "zoa_original" "${model}" "${batch_size}" "${GPU}" "${ZOA_WORKERS}" "${status}" "${failure_reason}" "${exit_code}" "${output_csv}" "${log_path}" "${started_at}" "${ended_at}" "${elapsed_s}"
  echo "[done] ${case_name}: status=${status} reason=${failure_reason:-none} elapsed=${elapsed_s}s"
}

run_external_case() {
  local case_name="$1"
  local method="$2"
  local model="$3"
  local batch_size="$4"
  local started_at ended_at start_s end_s elapsed_s output_csv log_path exit_code status failure_reason

  output_csv="${RESULTS_DIR}/${case_name}_${CORRUPTION}_l${LEVEL}_bs${batch_size}_gpu${GPU}.csv"
  log_path="${RESULTS_DIR}/logs/${case_name}_${CORRUPTION}_l${LEVEL}_bs${batch_size}_gpu${GPU}.log"
  started_at="$(date -Iseconds)"
  start_s="$(date +%s)"

  echo "[start] ${case_name}: method=${method} model=${model} batch=${batch_size} gpu=${GPU} workers=${WORKERS}"
  wait_for_gpu_idle "${case_name}"
  sleep "${NVME_PROCESS_GAP_SECONDS}"

  set +e
  GPU="${GPU}" DEVICE="cuda" BATCH_SIZE="${batch_size}" WORKERS="${WORKERS}" OUTPUT_CSV="${output_csv}" \
    DATA_ROOT="${DATA_ROOT}" DATA_CORRUPTION="${DATA_CORRUPTION}" CORRUPTION="${CORRUPTION}" SEVERITY="${SEVERITY}" LEVEL="${LEVEL}" \
    bash TTA/run_external_tta_command.sh "${case_name}" \
    > "${log_path}" 2>&1
  exit_code="$?"
  set -e

  failure_reason="$(classify_log_failure "${exit_code}" "${log_path}")"
  if [[ "${exit_code}" == "0" && -z "${failure_reason}" ]]; then
    status="ok"
  else
    status="error"
  fi

  ended_at="$(date -Iseconds)"
  end_s="$(date +%s)"
  elapsed_s="$(( end_s - start_s ))"
  append_status "${case_name}" "${method}" "${model}" "${batch_size}" "${GPU}" "${WORKERS}" "${status}" "${failure_reason}" "${exit_code}" "${output_csv}" "${log_path}" "${started_at}" "${ended_at}" "${elapsed_s}"
  echo "[done] ${case_name}: status=${status} reason=${failure_reason:-none} elapsed=${elapsed_s}s"
}

echo "[config] group=${GROUP} gpu=${GPU} workers=${WORKERS} zoa_workers=${ZOA_WORKERS} results=${RESULTS_DIR}"
echo "[config] batch policy: SFO-TTA=1, all other remaining TTA cases=64"
echo "[config] each task runs as a separate process; this script never launches two tasks on GPU ${GPU}"
echo "[config] gpu lock: ${LOCK_FILE}"

total="${#TASKS[@]}"
done_count=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r kind case_name a b c <<< "${task}"
  if [[ "${SKIP_OK:-1}" == "1" && "$(latest_status "${case_name}")" == "ok" ]]; then
    done_count=$(( done_count + 1 ))
    printf '[skip] %s ' "${case_name}"
    progress_bar "${done_count}" "${total}"
    printf '\n'
    continue
  fi

  printf '[progress] before %s ' "${case_name}"
  progress_bar "${done_count}" "${total}"
  printf '\n'

  if [[ "${kind}" == "zoa" ]]; then
    run_zoa_case "${case_name}" "${a}" "${b}"
  elif [[ "${kind}" == "zoa_original" ]]; then
    run_zoa_original_case "${case_name}" "${a}" "${b}"
  elif [[ "${kind}" == "external" ]]; then
    run_external_case "${case_name}" "${a}" "${b}" "${c}"
  else
    echo "Unknown task kind: ${kind}" >&2
    exit 2
  fi

  done_count=$(( done_count + 1 ))
  printf '[progress] after %s ' "${case_name}"
  progress_bar "${done_count}" "${total}"
  printf '\n'
done

echo "[complete] ${GROUP} finished. status csv: ${STATUS_CSV}"
