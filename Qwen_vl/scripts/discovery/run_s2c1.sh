#!/usr/bin/env bash
# S2-C1: lightweight teacher distillation viability.
#   features -> train 4 tiny scorers -> held-out separability -> accuracy
#   translation for the 2 val-selected scorers -> retained gain + gate.
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
    local rc=$?
    echo "[$(date +%H:%M:%S)] === END ${name} (exit ${rc}) ==="
    return ${rc}
}

step s2c1_features  scripts/discovery/s2c1_features.py --latency || exit 1
step s2c1_train     scripts/discovery/s2c1_train.py --latency   || exit 1
step s2c1_eval      scripts/discovery/s2c1_eval.py              || exit 1

# pick one arm per scorer family on the val split only
PICKS=$(python scripts/discovery/s2c1_select.py | tail -1)
echo "[$(date +%H:%M:%S)] selected for accuracy translation: ${PICKS}"

step s2c1_pilot scripts/discovery/s2c0_run.py \
    --scores s2c1_scores.npz --arms ${PICKS} --tag s2c1_pilot || exit 1

step s2c1_consolidate scripts/discovery/s2c1_consolidate.py --select "${PICKS// /,}"

echo "[$(date +%H:%M:%S)] S2-C1 DONE"
