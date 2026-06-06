#!/bin/bash
###############################################################################
# ZOA TTA Comprehensive Experiment Script
#
# Runs all supported TTA algorithms on both ViT-B and ResNet50,
# across batch sizes [1,4,16,64] and precisions [FP32, Quant_8, Quant_4].
#
# GPU assignment:
#   - GPU 2: ResNet50 experiments
#   - GPU 3: ViT-B experiments
#
# Usage:
#   bash scripts/run_all_tta.sh          # run both models in parallel
#   bash scripts/run_all_tta.sh vit      # run ViT-B only (GPU 3)
#   bash scripts/run_all_tta.sh resnet   # run ResNet50 only (GPU 2)
###############################################################################

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RESULT_DIR="${PROJECT_DIR}/TTA_result"
LOG_DIR="${RESULT_DIR}/logs"
STATUS_FILE="${RESULT_DIR}/status_board.txt"

GPU_RESNET=2
GPU_VIT=3

BATCH_SIZES=(1 4 16 64)

# Precision order: FP32 first, then Quant_8, then Quant_4 (last, error-prone)
PRECISIONS=("fp32" "quant_8" "quant_4")

ROUNDS=1

# 15 corruptions (ImageNet-C standard)
CORRUPTIONS=(
    gaussian_noise shot_noise impulse_noise
    defocus_blur glass_blur motion_blur zoom_blur
    snow frost fog brightness
    contrast elastic_transform pixelate jpeg_compression
)

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# ──────────────────────────────────────────────
# Status board helpers
# ──────────────────────────────────────────────
init_status_board() {
    cat > "${STATUS_FILE}" <<'HEADER'
================================================================================
                     ZOA TTA Experiment Status Board
================================================================================
Format:  [TIMESTAMP]  STATUS  |  EXPERIMENT_TAG
Status:  RUNNING / SUCCESS / FAILED / SKIPPED
================================================================================
HEADER
    echo "Experiment started at: $(date '+%Y-%m-%d %H:%M:%S')" >> "${STATUS_FILE}"
    echo "================================================================================" >> "${STATUS_FILE}"
}

log_status() {
    # Usage: log_status TAG STATUS [MESSAGE]
    local tag="$1"
    local status="$2"
    local msg="${3:-}"
    local ts
    ts="$(date '+%m-%d %H:%M:%S')"
    if [ -n "${msg}" ]; then
        printf "[%s]  %-8s |  %-60s |  %s\n" "${ts}" "${status}" "${tag}" "${msg}" >> "${STATUS_FILE}"
    else
        printf "[%s]  %-8s |  %s\n" "${ts}" "${status}" "${tag}" >> "${STATUS_FILE}"
    fi
}

# ──────────────────────────────────────────────
# Result parser
# ──────────────────────────────────────────────
parse_results() {
    # Parse the captured log and write per-corruption results to a txt file.
    # Usage: parse_results LOG_FILE RESULT_FILE
    local log_file="$1"
    local result_file="$2"

    echo "Corruption,Top1,Top5" > "${result_file}"

    # Pattern: "Under shift type <corruption> After <algo> Top-1 Accuracy: <x> and Top-5 Accuracy: <y>"
    grep "Under shift type" "${log_file}" | while IFS= read -r line; do
        local corruption top1 top5
        corruption=$(echo "${line}" | sed -n 's/.*Under shift type \([^ ]*\) After.*/\1/p')
        top1=$(echo "${line}" | sed -n 's/.*Top-1 Accuracy: \([0-9.]*\).*/\1/p')
        top5=$(echo "${line}" | sed -n 's/.*Top-5 Accuracy: \([0-9.]*\).*/\1/p')
        if [ -n "${corruption}" ] && [ -n "${top1}" ] && [ -n "${top5}" ]; then
            echo "${corruption},${top1},${top5}" >> "${result_file}"
        fi
    done

    # Count how many corruptions were recorded
    local count
    count=$(tail -n +2 "${result_file}" | wc -l)
    echo "  -> Parsed ${count}/15 corruption results into ${result_file}"
}

# ──────────────────────────────────────────────
# Core experiment runner
# ──────────────────────────────────────────────
run_single_experiment() {
    # Usage: run_single_experiment GPU ARCH ALGORITHM BATCH_SIZE PRECISION EXTRA_ARGS [DISPLAY_ALGO]
    local gpu="$1"
    local arch="$2"
    local algorithm="$3"
    local batch_size="$4"
    local precision="$5"
    local extra_args="$6"
    local display_algo="${7:-${algorithm}}"  # tag name (e.g. foa_pop2 vs foa)

    # Build tag for filenames
    local model_name
    if [ "${arch}" = "vit_base" ]; then
        model_name="ViT-B"
    else
        model_name="ResNet50"
    fi
    local tag="${model_name}_${display_algo}_bs${batch_size}_${precision}"
    local result_file="${RESULT_DIR}/${tag}.txt"
    local log_file="${LOG_DIR}/${tag}.log"

    # Skip if result already exists with all 15 corruptions
    if [ -f "${result_file}" ]; then
        local existing_count
        existing_count=$(tail -n +2 "${result_file}" | wc -l)
        if [ "${existing_count}" -ge 15 ]; then
            echo "[SKIP] ${tag} already complete (${existing_count} corruptions)"
            log_status "${tag}" "SKIPPED" "Already complete"
            return 0
        fi
    fi

    # Build quant arguments
    local quant_args=""
    if [ "${precision}" = "quant_8" ]; then
        quant_args="--quant --bit 8"
    elif [ "${precision}" = "quant_4" ]; then
        quant_args="--quant --bit 4"
    fi

    # Build the full command
    local cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py"
    cmd+=" --output ./outputs"
    cmd+=" --algorithm ${algorithm}"
    cmd+=" --arch ${arch}"
    cmd+=" --batch_size ${batch_size}"
    cmd+=" --rounds ${ROUNDS}"
    cmd+=" --tag _${tag}"
    cmd+=" ${quant_args}"
    cmd+=" ${extra_args}"

    echo ""
    echo "========================================================================"
    echo "[RUN] ${tag}"
    echo "  CMD: ${cmd}"
    echo "========================================================================"

    log_status "${tag}" "RUNNING"

    # Execute and capture output
    eval ${cmd} 2>&1 | tee "${log_file}"
    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        parse_results "${log_file}" "${result_file}"
        log_status "${tag}" "SUCCESS"
        echo "[DONE] ${tag} - SUCCESS"
    else
        log_status "${tag}" "FAILED" "exit_code=${exit_code}"
        echo "[DONE] ${tag} - FAILED (exit code: ${exit_code})"
        # Write a failure marker file so we know what happened
        echo "FAILED with exit code ${exit_code}" > "${result_file}"
        echo "See log: ${log_file}" >> "${result_file}"
    fi

    return ${exit_code}
}

# ──────────────────────────────────────────────
# Algorithm definitions with hyperparameters
# ──────────────────────────────────────────────

# --- ViT-B algorithms (GPU 3) ---
# Order: zoa_vit first, then the rest
VIT_ALGORITHMS=(
    "zoa_vit"
    "no_adapt"
    "bn_adapt"
    "t3a"
    "foa_pop2"
    "foa_pop27"
)

get_vit_args() {
    local algorithm="$1"
    case "${algorithm}" in
        zoa_vit)
            echo "--lr 0.0005 --sc 0.02 --lambda_bp 30 --domain_t 0.1"
            ;;
        no_adapt)
            echo ""
            ;;
        bn_adapt)
            echo ""
            ;;
        t3a)
            echo ""
            ;;
        foa_pop2)
            echo "--popsize 2 --fitness_lambda 0.4 --num_prompts 3"
            ;;
        foa_pop27)
            echo "--popsize 27 --fitness_lambda 0.4 --num_prompts 3"
            ;;
    esac
}

# Map display algorithm name to actual --algorithm value
get_vit_algorithm_name() {
    local algorithm="$1"
    case "${algorithm}" in
        foa_pop2)  echo "foa" ;;
        foa_pop27) echo "foa" ;;
        *)         echo "${algorithm}" ;;
    esac
}

# --- ResNet50 algorithms (GPU 2) ---
# Order: zoa_resnet first, then the rest
RESNET_ALGORITHMS=(
    "zoa_resnet"
    "no_adapt"
    "bn_adapt"
    "t3a"
    "foa_resnet_pop2"
    "foa_resnet_pop27"
)

get_resnet_args() {
    local algorithm="$1"
    # ResNet always needs in1k normalization
    local base_args="--use_in1k_norm --use_in1k_norm_c"
    case "${algorithm}" in
        zoa_resnet)
            echo "${base_args} --lr 0.0001 --sc 0.01 --lambda_bp 1 --domain_t 0.2"
            ;;
        no_adapt)
            echo "${base_args}"
            ;;
        bn_adapt)
            echo "${base_args}"
            ;;
        t3a)
            echo "${base_args}"
            ;;
        foa_resnet_pop2)
            echo "${base_args} --popsize 2 --fitness_lambda 0.4 --num_prompts 3"
            ;;
        foa_resnet_pop27)
            echo "${base_args} --popsize 27 --fitness_lambda 0.4 --num_prompts 3"
            ;;
    esac
}

get_resnet_algorithm_name() {
    local algorithm="$1"
    case "${algorithm}" in
        foa_resnet_pop2)  echo "foa_resnet" ;;
        foa_resnet_pop27) echo "foa_resnet" ;;
        *)                echo "${algorithm}" ;;
    esac
}

# ──────────────────────────────────────────────
# Model-level runners
# ──────────────────────────────────────────────

run_vit_experiments() {
    echo ""
    echo "################################################################"
    echo "#  ViT-B Experiments (GPU ${GPU_VIT})"
    echo "################################################################"

    local fail_count=0

    for precision in "${PRECISIONS[@]}"; do
        echo ""
        echo "────────────────────────────────────────"
        echo "  ViT-B | Precision: ${precision}"
        echo "────────────────────────────────────────"

        for algo in "${VIT_ALGORITHMS[@]}"; do
            local actual_algo
            actual_algo=$(get_vit_algorithm_name "${algo}")
            local extra_args
            extra_args=$(get_vit_args "${algo}")

            for bs in "${BATCH_SIZES[@]}"; do
                run_single_experiment "${GPU_VIT}" "vit_base" "${actual_algo}" "${bs}" "${precision}" "${extra_args}" "${algo}"
                if [ $? -ne 0 ]; then
                    ((fail_count++))
                fi
            done
        done
    done

    echo ""
    echo "################################################################"
    echo "#  ViT-B experiments complete. Failures: ${fail_count}"
    echo "################################################################"
}

run_resnet_experiments() {
    echo ""
    echo "################################################################"
    echo "#  ResNet50 Experiments (GPU ${GPU_RESNET})"
    echo "################################################################"

    local fail_count=0

    for precision in "${PRECISIONS[@]}"; do
        echo ""
        echo "────────────────────────────────────────"
        echo "  ResNet50 | Precision: ${precision}"
        echo "────────────────────────────────────────"

        for algo in "${RESNET_ALGORITHMS[@]}"; do
            local actual_algo
            actual_algo=$(get_resnet_algorithm_name "${algo}")
            local extra_args
            extra_args=$(get_resnet_args "${algo}")

            for bs in "${BATCH_SIZES[@]}"; do
                run_single_experiment "${GPU_RESNET}" "resnet50" "${actual_algo}" "${bs}" "${precision}" "${extra_args}" "${algo}"
                if [ $? -ne 0 ]; then
                    ((fail_count++))
                fi
            done
        done
    done

    echo ""
    echo "################################################################"
    echo "#  ResNet50 experiments complete. Failures: ${fail_count}"
    echo "################################################################"
}

# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
main() {
    init_status_board

    local mode="${1:-both}"  # both / vit / resnet

    echo "========================================================================"
    echo "  ZOA TTA Comprehensive Experiments"
    echo "  Mode       : ${mode}"
    echo "  Batch sizes: ${BATCH_SIZES[*]}"
    echo "  Precisions : ${PRECISIONS[*]}"
    echo "  Rounds     : ${ROUNDS}"
    echo "  Results    : ${RESULT_DIR}"
    echo "  Status     : ${STATUS_FILE}"
    echo "========================================================================"

    case "${mode}" in
        vit)
            run_vit_experiments
            ;;
        resnet)
            run_resnet_experiments
            ;;
        both)
            # Run ResNet (GPU 2) and ViT (GPU 3) in parallel
            run_resnet_experiments > >(tee "${LOG_DIR}/_resnet_all.log") 2>&1 &
            local resnet_pid=$!

            run_vit_experiments > >(tee "${LOG_DIR}/_vit_all.log") 2>&1 &
            local vit_pid=$!

            echo ""
            echo "Launched parallel processes:"
            echo "  ResNet50 (GPU ${GPU_RESNET}): PID ${resnet_pid}"
            echo "  ViT-B    (GPU ${GPU_VIT}): PID ${vit_pid}"
            echo ""
            echo "  Monitor status:  cat ${STATUS_FILE}"
            echo "  Monitor ResNet:  tail -f ${LOG_DIR}/_resnet_all.log"
            echo "  Monitor ViT:     tail -f ${LOG_DIR}/_vit_all.log"
            echo ""

            # Wait for both to finish
            wait ${resnet_pid}
            local resnet_exit=$?
            wait ${vit_pid}
            local vit_exit=$?

            echo ""
            echo "========================================================================"
            echo "  All experiments finished."
            echo "  ResNet50 exit code: ${resnet_exit}"
            echo "  ViT-B    exit code: ${vit_exit}"
            echo "========================================================================"
            ;;
        *)
            echo "Usage: $0 [both|vit|resnet]"
            exit 1
            ;;
    esac

    # Append summary to status board
    echo "================================================================================" >> "${STATUS_FILE}"
    echo "Experiment finished at: $(date '+%Y-%m-%d %H:%M:%S')" >> "${STATUS_FILE}"

    # Count successes/failures
    local total success failed skipped
    total=$(grep -c '|' "${STATUS_FILE}" 2>/dev/null || echo 0)
    success=$(grep -c 'SUCCESS' "${STATUS_FILE}" 2>/dev/null || echo 0)
    failed=$(grep -c 'FAILED' "${STATUS_FILE}" 2>/dev/null || echo 0)
    skipped=$(grep -c 'SKIPPED' "${STATUS_FILE}" 2>/dev/null || echo 0)
    echo "Summary: Total=${total}, Success=${success}, Failed=${failed}, Skipped=${skipped}" >> "${STATUS_FILE}"

    echo ""
    echo "Final status board:"
    cat "${STATUS_FILE}"
}

main "$@"
