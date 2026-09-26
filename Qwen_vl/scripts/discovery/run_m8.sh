#!/usr/bin/env bash
# M8 -- Targeted Zeroth-Order Audit.
#
#   0  m8_zop.py           the ZOO-Prune reproduction (Eq. 2-4) on the 450,
#                          gated bit-exact against the tower's own merger output
#   1  m8_phase1.py        the depth-wall grid: does the estimator rank the
#                          teacher's head inside a nominated pool?  (the gate)
#   2  m8_zop_diag.py      what the score is a function of (CPU only)
#   3  m8_zop_validate.py  fp32 reverse-mode control: faithful estimator, or a
#                          bf16 quantisation artefact?
#   4  m8_zol.py           the LLM-level estimator and its resolution floor
#   5  m8_zol_group.py     the shared-direction / Hadamard escape hatch
#   6  m8_zol_phase1.py    ZO-L through the same Phase-1 gate (the brief's ask)
#   7  m8_analyze.py       every table in the report, regenerated
#
# Phases 2-5 of the brief are gated on step 1 passing.  It does not, so the
# generation grid, the confirmation panel and the efficiency campaign are not
# run; m8_analyze.py still reports the estimator's own cost.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1

echo "=== 0/7  ZOO-Prune reproduction on the 450 ==="
python scripts/discovery/m8_zop.py 2>&1 | tee outputs/discovery/m8_zop.log

echo "=== 1/7  Phase 1: the depth-wall gate ==="
python scripts/discovery/m8_phase1.py 2>&1 | tee outputs/discovery/m8_phase1.log

echo "=== 2/7  what the score measures ==="
python scripts/discovery/m8_zop_diag.py 2>&1 | tee outputs/discovery/m8_zop_diag.log

echo "=== 3/7  fp32 reverse-mode control ==="
python scripts/discovery/m8_zop_validate.py 2>&1 | tee outputs/discovery/m8_zop_validate.log

echo "=== 4/7  the LLM-level estimator and its floor ==="
python scripts/discovery/m8_zol.py --limit 16 2>&1 | tee outputs/discovery/m8_zol.log

echo "=== 5/7  shared-direction escape hatch ==="
python scripts/discovery/m8_zol_group.py --limit 8 2>&1 | tee outputs/discovery/m8_zol_group.log

echo "=== 6/7  ZO-L through the same gate ==="
python scripts/discovery/m8_zol_phase1.py --limit 40 --dirs 2 2>&1 | tee outputs/discovery/m8_zol_phase1.log

echo "=== 7/7  tables ==="
python scripts/discovery/m8_analyze.py
