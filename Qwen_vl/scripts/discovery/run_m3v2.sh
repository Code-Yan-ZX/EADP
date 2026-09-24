#!/usr/bin/env bash
# M3-v2 -- MissGuard-v2: the query-conditioned HEAD auditor.
#
# Stages, in order.  Stage 3 is the brief's §7 early gate; stages 4 and 5 run
# ONLY if it passes (§8: "Downstream 只跑过 gate 的模型").
#
#   1  m3v2_text.py      instruction-token INPUT embeddings for the frozen 450
#                        (~2 min; row-aligned with m3_bank_v1.npz)
#   2  m3v2_train.py     the auditor grid -- 3 objectives x 3 ablations x 3 seeds
#   3  m3v2_proxy.py     THE HEAD GATE, on val, before any generation
#   4  m3v2_accuracy.py  the held-out 150 grid          (gated)
#   5  m3v2_perf.sh      the paired latency measurement (gated)
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean

echo "=== 1/3  text bank ==="
python scripts/discovery/m3v2_text.py

echo "=== 2/3  auditor grid ==="
python scripts/discovery/m3v2_train.py \
  --configs h1:A h1:B h1:C h2:A h2:B h2:C h3:A h3:B h3:C \
  --seeds 0 1 2 --out m3v2_auditor

echo "=== 3/3  HEAD GATE (val, no generation) ==="
python scripts/discovery/m3v2_proxy.py --ckpt-glob 'm3v2_auditor_[hH]*.pt'
python scripts/discovery/m3v2_analyze.py
