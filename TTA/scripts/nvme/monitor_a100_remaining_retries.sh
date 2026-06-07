#!/usr/bin/env bash
set -Eeuo pipefail

RESULTS_DIR="${1:-Results/TTA/nvme_runs/a100_remaining_retries_latest}"
INTERVAL="${INTERVAL:-5}"

while true; do
  clear || true
  date -Iseconds
  echo "results: ${RESULTS_DIR}"
  echo

  python3 - "${RESULTS_DIR}" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
status_files = sorted(root.glob("a100_remaining_*_status_*.csv"))
if not status_files:
    print("No status CSVs yet.")
else:
    for path in status_files:
        print(path.name)
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        latest = {}
        for row in rows:
            latest[row["case_name"]] = row
        if not latest:
            print("  no rows")
            continue
        ok = sum(1 for row in latest.values() if row.get("status") == "ok")
        err = sum(1 for row in latest.values() if row.get("status") == "error")
        print(f"  complete={ok} error={err} total_recorded={len(latest)}")
        for name, row in latest.items():
            reason = row.get("failure_reason") or "none"
            elapsed = row.get("elapsed_s") or "?"
            print(f"  - {name}: {row.get('status')} reason={reason} elapsed={elapsed}s gpu={row.get('gpu')} bs={row.get('batch_size')}")
        print()
PY

  echo "Active GPU/TTA processes:"
  ps -eo pid,ppid,stat,etimes,cmd | rg 'profile_zoa|run_external_tta_command|eval_bn_adapt|eval_vit.py|run_online_ltta' || true
  echo
  echo "Press Ctrl-C to stop monitoring."
  sleep "${INTERVAL}"
done
