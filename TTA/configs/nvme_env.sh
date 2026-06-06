#!/usr/bin/env bash
# Source this file before running NVME TTA sweeps.

export DATA_CORRUPTION="${DATA_CORRUPTION:-/home/oem/servers/imagenet-c}"
export DATA_ROOT="${DATA_ROOT:-${DATA_CORRUPTION}}"
export CORRUPTION="${CORRUPTION:-gaussian_noise}"
export LEVEL="${LEVEL:-5}"
export SEVERITY="${SEVERITY:-${LEVEL}}"

export GPU="${GPU:-0}"
export DEVICE="${DEVICE:-cuda:${GPU}}"
export WORKERS="${WORKERS:-4}"
export BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64,128}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"
export NO_IDLE_CHECK="${NO_IDLE_CHECK:-0}"

export PRECISION="${PRECISION:-amp_fp16}"
export FISHER_SAMPLES="${FISHER_SAMPLES:-128}"
export TTA_FORWARD_EQUIV_FACTOR="${TTA_FORWARD_EQUIV_FACTOR:-4.0}"
export ZOA_FORWARD_EQUIV_FACTOR="${ZOA_FORWARD_EQUIV_FACTOR:-2.0}"

export RESULTS_DIR="${RESULTS_DIR:-Results/TTA/nvme}"
