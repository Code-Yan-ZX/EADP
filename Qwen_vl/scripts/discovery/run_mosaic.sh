#!/usr/bin/env bash
# MosaicPrune phase-1 evaluation driver.
#
# Sequential jobs on one GPU, ordered so the primary gate (Dispersion vs
# Uniform vs Random at K=256 on DocVQA + OCRBench) lands first, then the
# remaining benchmarks, the unpruned fixed-res baseline, and finally the
# K=128 arms.
#
# Usage:
#   MODEL_FAMILY=qwen3-8b GPU_IDS="0" \
#     DATASETS="DocVQA_VAL OCRBench TextVQA_VAL" bash scripts/discovery/run_mosaic.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

prepare_output_dirs

OUTPUT_DIR="${OUTPUT_ROOT}/mosaic"
mkdir -p "${OUTPUT_DIR}"

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a GPU_LIST <<< "${GPU_IDS}"
gpu="${GPU_LIST[0]}"

run_one() {
    local model_name=$1
    local dataset_name=$2
    local tag="mosaic_${model_name}_${dataset_name}"
    echo "[$(date)] Starting ${tag} on GPU ${gpu}"
    if CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${VLMEVALKIT_DIR}/run.py" \
            --model "${model_name}" \
            --data "${dataset_name}" \
            --work-dir "${OUTPUT_DIR}" \
            --verbose \
            > "${LOG_DIR}/${tag}.log" 2>&1
    then
        echo "[$(date)] Finished ${tag} (ok)"
    else
        echo "[$(date)] Finished ${tag} (FAILED)"
    fi
}

# ---- K=256 primary gate arms ---------------------------------------------
for dataset in "${DATASET_LIST[@]}"; do
    run_one "Qwen3-VL-8B-Mosaic-256-dispersion" "${dataset}"
done
for dataset in "${DATASET_LIST[@]}"; do
    run_one "Qwen3-VL-8B-Mosaic-256-uniform" "${dataset}"
done
for dataset in "${DATASET_LIST[@]}"; do
    run_one "Qwen3-VL-8B-Mosaic-256-random" "${dataset}"
done

# ---- unpruned fixed-res baseline (1024 tokens) ----------------------------
if [ "${RUN_BASELINE:-1}" = "1" ]; then
    for dataset in "${DATASET_LIST[@]}"; do
        run_one "${BASELINE_MODEL}" "${dataset}"
    done
fi

# ---- K=128 low-budget arms ------------------------------------------------
if [ "${RUN_K128:-1}" = "1" ]; then
    for dataset in "${DATASET_LIST[@]}"; do
        run_one "Qwen3-VL-8B-Mosaic-128-dispersion" "${dataset}"
    done
    for dataset in "${DATASET_LIST[@]}"; do
        run_one "Qwen3-VL-8B-Mosaic-128-uniform" "${dataset}"
    done
    for dataset in "${DATASET_LIST[@]}"; do
        run_one "Qwen3-VL-8B-Mosaic-128-random" "${dataset}"
    done
fi

echo "========================================="
echo "[$(date)] MosaicPrune evaluation complete."
echo "========================================="
