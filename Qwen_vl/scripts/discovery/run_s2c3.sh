#!/usr/bin/env bash
# S2-C3 end to end: retarget the S2-C1 linear scorer at the head of the teacher's
# ranking, then measure whether head retention converts into accuracy.
#
# The features, the split, the model, the loss form, the optimizer and the early
# stopping criterion are all S2-C1's. Only the teacher reading changes.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
cd /media/disk2/YZX/research/EADP/Qwen_vl

ARMS="BASE HEAD_BIN HEAD_MULTI HEAD_RANK"
SEEDS="0 1 2"

# 1. train every target with three seeds (12 runs; the LLM is not loaded --
#    the scorer only ever reads the cached S2-C1 layer-4 features)
python -u scripts/discovery/s2c3_train.py --arms $ARMS --seeds $SEEDS --tag s2c3

# 2. held-out head retention + the identity check against S2-C2's baselines
python -u scripts/discovery/s2c3_eval.py --tag s2c3

# 3. accuracy translation through the UNMODIFIED generation harness, Top-K @256,
#    every arm and every seed, on the same held-out 150
python -u scripts/discovery/s2c0_run.py --scores s2c3_scores.npz \
    --arms BASE_s0 BASE_s1 BASE_s2 \
           HEAD_BIN_s0 HEAD_BIN_s1 HEAD_BIN_s2 \
           HEAD_MULTI_s0 HEAD_MULTI_s1 HEAD_MULTI_s2 \
           HEAD_RANK_s0 HEAD_RANK_s1 HEAD_RANK_s2 \
    --tag s2c3_pilot

# 4. supplementary diagnostic: is the head linearly expressible from L4 at all?
#    (CPU only, trains nothing -- interpretive, does not gate anything)
python -u scripts/discovery/s2c3_oracle.py --tag s2c3

# 5. consolidate: head-vs-accuracy, the class-1 rescue block, the verdict
python -u scripts/discovery/s2c3_consolidate.py --tag s2c3

# 6. the figure
python -u scripts/discovery/s2c3_figure.py
