#!/usr/bin/env bash
# GO-triggered R_MAIN025 full panel (addendum §2), chained AFTER the core
# resume finishes (serial, no preemption, single A40).
set -u
cd "$(dirname "$0")"
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
echo "[rtg-full] $(date) waiting for the core-resume driver to exit..."
while pgrep -f "rtg_run_all.sh" >/dev/null; do sleep 120; done
echo "[rtg-full] $(date) core resume finished — R_MAIN025 full panel begins"
echo "[rtg-full] throughput basis (DEV): gen 0.5-1.3 s/q, bank ~0.35 s/q;"
echo "[rtg-full] ETA bank ~1.1h + gen ~3.2h + score ~0.3h = ~4.5-5h"
$PY rtg_bank_full.py > bank_full.log 2>&1
B=$?
echo "[rtg-full] $(date) bank rc=$B"
$PY rtg_accuracy_full.py --mode both > full_run.log 2>&1
G=$?
echo "[rtg-full] $(date) gen+score rc=$G"
$PY rtg_analyze_full.py > analyze_full.log 2>&1
A=$?
echo "[rtg-full] $(date) analyze rc=$A — R_MAIN025 FULL DONE (bank=$B gen=$G ana=$A)"
