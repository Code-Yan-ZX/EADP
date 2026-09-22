#!/usr/bin/env bash
# Chain the remaining discovery jobs so the GPU never idles between them.
# Waits for the Part 1 profiling run to exit first (single-GPU machine, and
# timings are only meaningful when nothing else is running).
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

echo "[$(date +%H:%M:%S)] waiting for profile_efficiency.py to finish..."
while pgrep -f "profile_efficiency.py" > /dev/null; do sleep 15; done
echo "[$(date +%H:%M:%S)] GPU free, starting chain"

step bench_selectors \
    scripts/discovery/bench_selectors.py

step diag_overlap \
    scripts/discovery/diag_overlap.py --per-dataset 200 --budget 256

step diag_selectors \
    scripts/discovery/diag_selectors.py \
        --datasets TextVQA_VAL DocVQA_VAL OCRBench \
        --per-dataset 150 --budgets 256 \
        --selectors facility facility_fast lazy_greedy stochastic topk topk_nms farthest

# Tighter budget: the paper's motivation for facility location is behaviour
# under *strict* budgets, so the FL-vs-TopK gap is checked at 128 as well.
step diag_selectors_b128 \
    scripts/discovery/diag_selectors.py \
        --datasets TextVQA_VAL DocVQA_VAL OCRBench \
        --per-dataset 150 --budgets 128 \
        --selectors facility topk topk_nms stochastic

step fa_battery_baseline \
    scripts/discovery/failure_analysis.py --stage battery --model baseline --per-dataset 50

step fa_battery_eadp \
    scripts/discovery/failure_analysis.py --stage battery --model eadp --per-dataset 50

step fa_probe \
    scripts/discovery/failure_analysis.py --stage probe --per-dataset 7

step fa_classify \
    scripts/discovery/failure_analysis.py --stage classify

echo "[$(date +%H:%M:%S)] ALL DONE"
