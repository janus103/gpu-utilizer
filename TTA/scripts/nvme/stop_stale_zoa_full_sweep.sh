#!/usr/bin/env bash
set -Eeuo pipefail

CONFIRM="${CONFIRM:-0}"

mapfile -t PIDS < <(python3 <<'PY'
import subprocess

out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
for line in out.splitlines():
    stripped = line.strip()
    if not stripped:
        continue
    pid, _, cmd = stripped.partition(" ")
    if (
        "python3 TTA/profile_zoa_fp16_stream.py" in cmd
        and "--models resnet50,vit_base_patch16_224,mobilevit_xxs" in cmd
        and "--batch-sizes 1,2,4,8,16,32,64,128" in cmd
    ):
        print(pid)
PY
)

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "[stop-stale] no stale full ZOA sweep processes found"
  exit 0
fi

echo "[stop-stale] matching stale full ZOA sweep pids: ${PIDS[*]}"
if [[ "${CONFIRM}" != "1" ]]; then
  echo "[stop-stale] dry run only. Re-run with CONFIRM=1 to terminate them:"
  echo "  CONFIRM=1 bash TTA/scripts/nvme/stop_stale_zoa_full_sweep.sh"
  exit 0
fi

kill "${PIDS[@]}" || true
sleep 5

mapfile -t STILL_RUNNING < <(python3 <<'PY'
import subprocess

out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
for line in out.splitlines():
    stripped = line.strip()
    if not stripped:
        continue
    pid, _, cmd = stripped.partition(" ")
    if (
        "python3 TTA/profile_zoa_fp16_stream.py" in cmd
        and "--models resnet50,vit_base_patch16_224,mobilevit_xxs" in cmd
        and "--batch-sizes 1,2,4,8,16,32,64,128" in cmd
    ):
        print(pid)
PY
)
if [[ "${#STILL_RUNNING[@]}" -gt 0 ]]; then
  echo "[stop-stale] still running after SIGTERM, sending SIGKILL: ${STILL_RUNNING[*]}"
  kill -9 "${STILL_RUNNING[@]}" || true
fi

echo "[stop-stale] done"
