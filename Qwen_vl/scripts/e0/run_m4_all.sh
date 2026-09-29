#!/usr/bin/env bash
# E0 M4 driver: all arm x K x dataset generation+scoring, sequential,
# resumable (each shard skips finished questions). Run after the M3 pilot
# estimate is accepted.
set -uo pipefail
cd /media/disk2/YZX/research/EADP
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() { echo "=== $* ==="; $PY Qwen_vl/scripts/e0/e0_accuracy.py "$@" 2>&1 | tail -3; }

# B0 once
run --arm b0 --K 1024
# local + ported arms at the three budgets
for K in 256 128 64; do
  for arm in b2 b1 divprune cdpruner hiprune visionzip fastv pdrop sparsevlm_norecycle; do
    run --arm $arm --K $K
  done
done
# R-res variants
run --arm rres --K 256
run --arm rres --K 128
run --arm rres --K 64
# legacy ablations, K=256 OCR panel only
run --arm a1
run --arm a2
echo "M4_GEN_DONE"
