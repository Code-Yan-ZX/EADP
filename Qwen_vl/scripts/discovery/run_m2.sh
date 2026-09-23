#!/usr/bin/env bash
# M2 — Gradient-Distilled Early Pruning: the full, reproducible sequence.
#
# Order is enforced by m2_run.py, not by this script: every correctness gate
# must pass before the timing or accuracy stages execute, and a failed gate
# exits non-zero without producing a performance or accuracy number
# (pre-registration section 4).
#
# One model load (~7 min on this host) serves all three GPU stages, so the whole
# run is a single Python process plus two CPU-only post-processing steps.
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
LOGDIR=outputs/discovery
export PYTHONUNBUFFERED=1

step () {
  name="$1"; shift
  echo "=== $name  $(date '+%F %T') ==="
  python -u "$@" 2>&1 | tee "$LOGDIR/$name.log"
  rc=${PIPESTATUS[0]}
  echo "=== $name exit=$rc  $(date '+%F %T') ==="
  [ "$rc" -eq 0 ] || { echo "STOPPING: $name failed"; exit "$rc"; }
}

# --- 1. correctness gates + timing grid + accuracy grid (one model load) -----
step m2_run scripts/discovery/m2_run.py \
  --stages correctness perf accuracy \
  --gates A B C D E \
  --warmup 20 --iters 100 \
  --acc-seeds 0 1 2

# --- 2. CPU-only post-processing --------------------------------------------
step m2_consolidate scripts/discovery/m2_consolidate.py
step m2_figure      scripts/discovery/m2_figure.py

echo "ALL DONE  $(date '+%F %T')"
