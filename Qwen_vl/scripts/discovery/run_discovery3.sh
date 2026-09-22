#!/usr/bin/env bash
# Follow-up 2: the block-greedy family.
#   Does the coverage objective need T *sequential* greedy steps, or do a
#   handful of block-parallel steps recover the same selection at a fraction
#   of the latency?  block=1 is provably identical to the official greedy.
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

echo "[$(date +%H:%M:%S)] waiting for chain 2..."
until grep -q "ALL FOLLOW-UPS DONE" "${LOGDIR}/chain2.log" 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] chain 2 done, starting block-greedy sweep"

step diag_selectors_block \
    scripts/discovery/diag_selectors.py \
        --datasets TextVQA_VAL DocVQA_VAL OCRBench \
        --per-dataset 150 --budgets 256 \
        --selectors block8 block32

echo "[$(date +%H:%M:%S)] ALL BLOCK-GREEDY DONE"
