#!/usr/bin/env bash
# S2-C0: forward-only proxy viability (pilot 150 + 15 causal cases).
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

step s2c0_sanity scripts/discovery/s2c0_sanity.py
step s2c0_forward_proxy scripts/discovery/s2c0_forward_proxy.py --prefix-latency

echo "[$(date +%H:%M:%S)] S2-C0 PROXY DONE"
