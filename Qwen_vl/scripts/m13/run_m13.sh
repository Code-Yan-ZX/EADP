#!/usr/bin/env bash
# M13 runner: profile -> DS ablation -> depth oracle -> (analysis separately).
# Resumable: every accuracy arm shards per sample; rerun skips finished work.
set -uo pipefail
cd "$(dirname "$0")/../.."   # Qwen_vl root

PYTHON=${PYTHON:-python}
mkdir -p outputs/m13/logs

echo "=== [1/3] DS + stage profiling ==="
$PYTHON scripts/m13/m13_profile.py --blocks 30 2>&1 | tee outputs/m13/logs/profile.log

echo "=== [2/3] DeepStack branch ablation (gen+score) ==="
$PYTHON scripts/m13/m13_ds_ablation.py --mode both 2>&1 | tee outputs/m13/logs/ds_ablation.log

echo "=== [3/3] Compression-depth oracle (gen+score) ==="
$PYTHON scripts/m13/m13_depth_oracle.py --K 512 --mode both 2>&1 | tee outputs/m13/logs/depth_oracle.log

echo "=== M13 generation runs complete ==="
