#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runs each script sequentially. Each Python profiler creates and releases one
# model at a time, so only one target model is resident on the GPU per case.
bash "${SCRIPT_DIR}/run_01_baseline_inference_fp16.sh"
bash "${SCRIPT_DIR}/run_02_tent_eata_sar_fp16.sh"
bash "${SCRIPT_DIR}/run_03_zoa_fp16.sh"
bash "${SCRIPT_DIR}/run_04_ltta_sfo_source_only.sh"
