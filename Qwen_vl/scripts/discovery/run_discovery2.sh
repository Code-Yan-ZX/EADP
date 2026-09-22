#!/usr/bin/env bash
# Follow-up jobs. Waits for the first chain to report ALL DONE, then runs:
#   * diag_overlap  (failed the first time on an HF network call; now offline)
#   * the similarity-kernel arm, which tests hypothesis H2
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

echo "[$(date +%H:%M:%S)] waiting for chain 1..."
until grep -q "ALL DONE" "${LOGDIR}/chain.log" 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] chain 1 done, starting follow-ups"

step diag_overlap \
    scripts/discovery/diag_overlap.py --per-dataset 200 --budget 256

step diag_selectors_clamp \
    scripts/discovery/diag_selectors.py \
        --datasets TextVQA_VAL DocVQA_VAL OCRBench \
        --per-dataset 150 --budgets 256 --sim-mode clamp \
        --selectors facility

echo "[$(date +%H:%M:%S)] ALL FOLLOW-UPS DONE"
