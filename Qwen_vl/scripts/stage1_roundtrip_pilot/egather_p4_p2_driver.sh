#!/usr/bin/env bash
# P4+P2 driver (2026-10-04): waits for the P1 E_GATHER marker, then
#   step 1: smoke the two P4 ablation arms (2 rows each, determinism +
#           lam-0 identity already gated in rtg_legacy_checks.py C2)
#   step 2: P4 ablation arms L_E_MAIN025 / L_R_GATHER full gen+score on
#           the fixed 4-dataset panel (TextVQA/ChartQA/DocVQA/OCRBench)
#   step 3: P2 legacy online efficiency panels (dev30 then table4)
# GPU-serial after P1; logs in this directory.
set -u
cd "$(dirname "$0")"
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python

echo "[p4] $(date) waiting for P1 marker /tmp/p1_egather_done.marker"
while [ ! -f /tmp/p1_egather_done.marker ]; do sleep 60; done
echo "[p4] $(date) P1 finished — smoke ablation arms"

$PY egather_legacy.py --arm L_R_GATHER  --mode both --datasets OCRBench --limit 2 > p4_smoke_rg.log 2>&1
echo "[p4] $(date) smoke L_R_GATHER rc=$?"
$PY egather_legacy.py --arm L_E_MAIN025 --mode both --datasets OCRBench --limit 2 > p4_smoke_em.log 2>&1
echo "[p4] $(date) smoke L_E_MAIN025 rc=$?"

echo "[p4] $(date) P4 ablation arms full"
$PY egather_legacy.py --arm L_R_GATHER  --mode both > p4_full_rg.log 2>&1
echo "[p4] $(date) L_R_GATHER rc=$?"
$PY egather_legacy.py --arm L_E_MAIN025 --mode both > p4_full_em.log 2>&1
echo "[p4] $(date) L_E_MAIN025 rc=$?"

echo "[p4] $(date) P2 legacy perf panels"
$PY egather_p2_perf_legacy.py --panel dev30 --out ../outputs/stage1_roundtrip_pilot/legacy_full/p2_perf_legacy_dev30.json > p2_perf_dev30.log 2>&1
echo "[p4] $(date) P2 dev30 rc=$?"
$PY egather_p2_perf_legacy.py --panel table4 --out ../outputs/stage1_roundtrip_pilot/legacy_full/p2_perf_legacy_table4.json > p2_perf_table4.log 2>&1
echo "[p4] $(date) P2 table4 rc=$?"

touch /tmp/p4_p2_done.marker
echo "[p4] $(date) DONE"
