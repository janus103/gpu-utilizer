#!/bin/bash
###############################################################################
# ResNet50 TTA Runner — 15 corruptions → CSV
#
# Usage:
#   bash scripts/run_resnet_tta.sh <GPU> <MODE> <BIT> <PRETRAINED_PATH>
#
# Arguments:
#   GPU             GPU device number (e.g. 0, 1, 2, 3)
#   MODE            Algorithm name (e.g. no_adapt, zoa_resnet, foa_resnet, t3a, bn_adapt)
#   BIT             Quantization bit-width (32 = FP32, otherwise enables --quant)
#   PRETRAINED_PATH Path to pretrained model checkpoint
#
# Examples:
#   bash scripts/run_resnet_tta.sh 0 zoa_resnet 8 ~/NIPS/jin/SOA/pytorch-image-models/ZOA_WEIGHT/ZOA_resnet50.pth
#   bash scripts/run_resnet_tta.sh 1 no_adapt 32 ~/NIPS/jin/SOA/pytorch-image-models/ZOA_WEIGHT/ZOA_resnet50.pth
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# ── Argument parsing ────────────────────────────
if [ $# -lt 4 ]; then
    echo "Usage: $0 <GPU> <MODE> <BIT> <PRETRAINED_PATH>"
    echo "  GPU  : GPU device number (0, 1, 2, 3, ...)"
    echo "  MODE : algorithm name (no_adapt, zoa_resnet, foa_resnet, t3a, bn_adapt, ...)"
    echo "  BIT  : quantization bit-width (32 for FP32, 8/6/4 for quantized)"
    echo "  PRETRAINED_PATH : path to pretrained .pth checkpoint"
    exit 1
fi

GPU_NUM="$1"
MODE="$2"
BIT="$3"
PRETRAINED_PATH="$4"

if [ ! -f "${PRETRAINED_PATH}" ]; then
    echo "[ERROR] Pretrained model not found: ${PRETRAINED_PATH}"
    exit 1
fi

# ── Tag & output paths ──────────────────────────
if [ "${BIT}" -eq 32 ] 2>/dev/null; then
    PRECISION="fp32"
else
    PRECISION="quant_${BIT}"
fi

TAG="resnet50_${MODE}_${PRECISION}"
RESULT_DIR="${PROJECT_DIR}/TTA_result"
LOG_DIR="${RESULT_DIR}/logs"
CSV_FILE="${RESULT_DIR}/${TAG}.csv"
LOG_FILE="${LOG_DIR}/${TAG}.log"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

CORRUPTIONS=(
    gaussian_noise shot_noise impulse_noise
    defocus_blur glass_blur motion_blur zoom_blur
    snow frost fog brightness
    contrast elastic_transform pixelate jpeg_compression
)

# ── Quantization flags ──────────────────────────
QUANT_ARGS=""
if [ "${PRECISION}" != "fp32" ]; then
    QUANT_ARGS="--quant --bit ${BIT}"
fi

# ── CSV header ──────────────────────────────────
echo "corruption,top1,top5" > "${CSV_FILE}"

echo "========================================================================"
echo "  ResNet50 TTA Experiment"
echo "  GPU       : ${GPU_NUM}"
echo "  Mode      : ${MODE}"
echo "  Bit       : ${BIT} (${PRECISION})"
echo "  Pretrained: ${PRETRAINED_PATH}"
echo "  Output CSV: ${CSV_FILE}"
echo "  Log       : ${LOG_FILE}"
echo "========================================================================"

FAIL_COUNT=0
TOTAL=0

: > "${LOG_FILE}"

for CORRUPT in "${CORRUPTIONS[@]}"; do
    TOTAL=$((TOTAL + 1))
    ITER_TAG="${TAG}_${CORRUPT}"

    CMD="CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py"
    CMD+=" --arch resnet50"
    CMD+=" --algorithm ${MODE}"
    CMD+=" --corruption ${CORRUPT}"
    CMD+=" --batch_size 256"
    CMD+=" --rounds 1"
    CMD+=" --tag ${ITER_TAG}"
    CMD+=" --pretrained_path ${PRETRAINED_PATH}"
    CMD+=" --use_in1k_norm --use_in1k_norm_c"
    CMD+=" --lr 0.0001 --sc 0.01 --lambda_bp 1 --domain_t 0.2"
    CMD+=" ${QUANT_ARGS}"

    echo ""
    echo "[${TOTAL}/15] Running: ${CORRUPT}"
    echo "  CMD: ${CMD}"

    ITER_LOG=$(mktemp)
    if eval ${CMD} 2>&1 | tee -a "${LOG_FILE}" | tee "${ITER_LOG}"; then
        LINE=$(grep "Under shift type" "${ITER_LOG}" | tail -1)
        if [ -n "${LINE}" ]; then
            TOP1=$(echo "${LINE}" | sed -n 's/.*Top-1 Accuracy: \([0-9.]*\).*/\1/p')
            TOP5=$(echo "${LINE}" | sed -n 's/.*Top-5 Accuracy: \([0-9.]*\).*/\1/p')
            echo "${CORRUPT},${TOP1},${TOP5}" >> "${CSV_FILE}"
            echo "  -> ${CORRUPT}: Top1=${TOP1}, Top5=${TOP5}"
        else
            echo "${CORRUPT},PARSE_ERROR,PARSE_ERROR" >> "${CSV_FILE}"
            echo "  -> [WARN] Could not parse accuracy for ${CORRUPT}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "${CORRUPT},FAILED,FAILED" >> "${CSV_FILE}"
        echo "  -> [FAIL] ${CORRUPT} exited with error"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    rm -f "${ITER_LOG}"
done

# ── Summary row ─────────────────────────────────
AVG_TOP1=$(awk -F',' 'NR>1 && $2+0==$2 {s+=$2; n++} END {if(n>0) printf "%.3f", s/n; else print "N/A"}' "${CSV_FILE}")
AVG_TOP5=$(awk -F',' 'NR>1 && $3+0==$3 {s+=$3; n++} END {if(n>0) printf "%.3f", s/n; else print "N/A"}' "${CSV_FILE}")
echo "mean,${AVG_TOP1},${AVG_TOP5}" >> "${CSV_FILE}"

echo ""
echo "========================================================================"
echo "  Done. ${TOTAL} corruptions, ${FAIL_COUNT} failures."
echo "  Mean Top-1: ${AVG_TOP1}  |  Mean Top-5: ${AVG_TOP5}"
echo "  Results saved to: ${CSV_FILE}"
echo "========================================================================"
