#!/usr/bin/env bash
# P1 driver (2026-10-04): E_GATHER official-legacy full panel for the 7
# Table-4 datasets without recovered official predictions.
# Step A: build missing ACV b1 banks (AI2D/HallB/MME/MMB-CN/InfoVQA;
#         HallB already has 3 smoke records — resumable).
# Step B: L_E_GATHER generation for 7 datasets (resumable shards).
# Step C: scoring (classic 5 via frozen _score_one, new5 via rtg_score_new5).
# GPU-serial; logs in this directory; marker file on completion.
set -u
cd "$(dirname "$0")"
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
ACV=../anchor_completion_validation
BANK_DS="AI2D_TEST,HallusionBench,MME,MMBench_DEV_CN_V11,InfoVQA_VAL"
GEN_DS="ChartQA_TEST,AI2D_TEST,HallusionBench,MME,MMBench_DEV_EN_V11,MMBench_DEV_CN_V11,InfoVQA_VAL"

echo "[p1] $(date) step A: bank build (${BANK_DS})"
$PY ${ACV}/acu_bank.py --datasets "${BANK_DS}" > egather_bank5.log 2>&1
echo "[p1] $(date) step A rc=$?"

echo "[p1] $(date) step B: E_GATHER generation (${GEN_DS})"
$PY egather_legacy.py --mode gen --datasets "${GEN_DS}" > egather_gen.log 2>&1
echo "[p1] $(date) step B rc=$?"

echo "[p1] $(date) step C: scoring (${GEN_DS})"
$PY egather_legacy.py --mode score --datasets "${GEN_DS}" > egather_score.log 2>&1
echo "[p1] $(date) step C rc=$?"

touch /tmp/p1_egather_done.marker
echo "[p1] $(date) DONE"
