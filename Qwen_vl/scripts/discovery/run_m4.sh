#!/usr/bin/env bash
# M4-v0 -- Residual Evidence Compression: the held-out 150 grid.
#
#   0  m4_selftest.py    the capsule math against closed forms, and the
#                        determinism guarantee the first grid lacked (seconds)
#   1  m4_offline.py     residual maps + the knobs they fix (17 s)
#   2  m4_accuracy.py    the grid (13 arms x 150 instances, ~16 min)
#   3  m4_analyze.py     tables, controls, the frozen verdict
#   4  m4_figures.py     mechanism, profile, diagnostics, case studies
#
# REC0 first: it is the identity gate, and every later arm is only readable if
# the r=0 arm reproduces the stored B2 hit-for-hit.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1

echo "=== 0/4  self-test ==="
python scripts/discovery/m4_selftest.py

echo "=== 1/4  offline diagnostics ==="
python scripts/discovery/m4_offline.py

echo "=== 2/4  held-out 150 grid ==="
python scripts/discovery/m4_accuracy.py --resume --arms \
  REC0 REC-r8 REC-r16 REC-r32 \
  MEAN-r16 IMP-r16 SHUF-r16 ANCH-r16 FPS-r16 NORM-r16 \
  EVICT-r8 EVICT-r16 EVICT-r32 \
  2>&1 | tee outputs/discovery/m4_accuracy.log

echo "=== 3/4  analysis ==="
python scripts/discovery/m4_analyze.py

echo "=== 4/4  figures ==="
python scripts/discovery/m4_figures.py
