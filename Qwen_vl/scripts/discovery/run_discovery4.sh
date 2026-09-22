#!/usr/bin/env bash
# Follow-up 3: the one-shot stratified selector.
#   Tests whether coverage can be bought without any sequential loop, which the
#   `farthest` result suggests might be most of what the objective provides.
set -u

cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate qwen3vl_clean

LOGDIR=outputs/discovery
mkdir -p "${LOGDIR}"

step () {
    local name=$1; shift
    echo "[$(date +%H:%M:%S)] === START ${name} ==="
    python -u "$@" > "${LOGDIR}/${name}.log" 2>&1
    echo "[$(date +%H:%M:%S)] === END ${name} (exit $?) ==="
}

echo "[$(date +%H:%M:%S)] waiting for chain 3..."
until grep -q "ALL BLOCK-GREEDY DONE" "${LOGDIR}/chain3.log" 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] chain 3 done, starting grid sweep"

step diag_selectors_grid \
    scripts/discovery/diag_selectors.py \
        --datasets TextVQA_VAL DocVQA_VAL OCRBench \
        --per-dataset 150 --budgets 128 256 \
        --selectors grid

echo "[$(date +%H:%M:%S)] ALL GRID DONE"
