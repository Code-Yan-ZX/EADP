#!/usr/bin/env bash
# Stage-1 grounding guidance discovery — full command sequence.
# Worktree: /media/disk2/YZX/research/EADP_amp (branch codex/anchor-merge-pilot)
# Env: qwen3vl_clean. Single A40. Protocol: PROTOCOL_NOTES.md (frozen 2026-10-01).
set -euo pipefail
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# 0. correctness gates G1-G3 (must PASS before anything else)
$PY s1g_gate.py --gate-samples 3 --lam 1.0

# 1. lambda calibration: smoke 10/ds, A_G1 + B_G1, lam in {0.25,0.5,1.0};
#    freeze the winner ONCE (scripts/freeze_lam.sh records the choice)
bash calib_smoke.sh

# 2. DEV panel: 7 arms x 300 (LAM / TAG frozen from step 1)
# $PY s1g_accuracy.py --split dev --arms A_G1,A_G2,A_G3,B_G1,B_G2,B_G3,C_G1 \
#       --lam <LAM> --tag <TAG> --mode both --diag
# $PY s1g_analyze.py --split dev --arms BASE,A_G1,A_G2,A_G3,B_G1,B_G2,B_G3,C_G1 \
#       --lam <LAM> --tag <TAG>

# 3. mechanism cases (uses the --diag jsonl from step 2)
# $PY s1g_cases.py --diag ../../outputs/stage1_grounding/acc/dev/<ARM>_<TAG>/K256/<DS>_diag.jsonl --arm <ARM>

# 4. paired selector latency
# $PY s1g_perf.py --arms A_G1,...,C_G1 --samples 10

# 5. if and only if DEV shows a candidate: CONFIRM panel 600
# $PY s1g_accuracy.py --split confirm --arms BASE,<WINNER> --lam <LAM> --tag <TAG> --mode both
# $PY s1g_analyze.py --split confirm --arms BASE,<WINNER> --lam <LAM> --tag <TAG>

# 6. combination with the frozen Stage-2 winner (Anchor Completion U025),
#    ONLY after Stage-1 verdict: selector=<s1 winner> + merge arm U025.
