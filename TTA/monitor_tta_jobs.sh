#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

INTERVAL="${INTERVAL:-5}"

while true; do
  clear || true
  date
  echo
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv
  echo
  echo "Running TTA profile processes:"
  pgrep -af "TTA/profile_tta.py|TTA/profile_tta_stream.py|run_imagenet_c_gpu" || true
  echo
  for log in Results/TTA/logs/gpu*.log; do
    [[ -e "${log}" ]] || continue
    echo "===== ${log} ====="
    tail -n 20 "${log}"
    echo
  done
  sleep "${INTERVAL}"
done
