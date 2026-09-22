#!/usr/bin/env bash
# Restart of the tail of chain 1 after the map-dump fix (the original run was
# writing 11 MB of unused sim_matrix / image_embeds per instance).
# Emits the "ALL DONE" marker at the end so chains 2/3/4 unblock.
set -u

cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate qwen3vl_clean

LOGDIR=outputs/discovery

step () {
    local name=$1; shift
    echo "[$(date +%H:%M:%S)] === START ${name} ==="
    python -u "$@" > "${LOGDIR}/${name}.log" 2>&1
    echo "[$(date +%H:%M:%S)] === END ${name} (exit $?) ==="
}

step fa_battery_eadp \
    scripts/discovery/failure_analysis.py --stage battery --model eadp --per-dataset 50

step fa_probe \
    scripts/discovery/failure_analysis.py --stage probe --per-dataset 7

step fa_classify \
    scripts/discovery/failure_analysis.py --stage classify

# Unblock the follow-up chains waiting on this marker.
echo "[$(date +%H:%M:%S)] ALL DONE" >> "${LOGDIR}/chain.log"
echo "[$(date +%H:%M:%S)] CHAIN 1B COMPLETE"
