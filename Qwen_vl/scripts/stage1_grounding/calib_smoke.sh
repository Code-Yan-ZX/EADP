#!/usr/bin/env bash
set -euo pipefail
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
cd "$(dirname "$0")"
for LAM in 0.25 0.5 1.0; do
  TAG="smoke_lam${LAM}"
  $PY s1g_accuracy.py --split dev --arms A_G1,B_G1 --lam $LAM --tag $TAG \
      --smoke 10 --mode both --diag
done
for LAM in 0.25 0.5 1.0; do
  TAG="smoke_lam${LAM}"
  $PY s1g_analyze.py --split dev --arms BASE,A_G1,B_G1 --lam $LAM --tag $TAG
done
