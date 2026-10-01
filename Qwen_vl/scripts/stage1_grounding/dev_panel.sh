#!/usr/bin/env bash
set -euo pipefail
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
cd "$(dirname "$0")"
ARMS="A_G1,A_G2,A_G3,B_G1,B_G2,B_G3,C_G1"
$PY s1g_accuracy.py --split dev --arms $ARMS --lam 1.0 --tag lam1 --mode both --diag
$PY s1g_analyze.py --split dev --arms BASE,$ARMS --lam 1.0 --tag lam1
