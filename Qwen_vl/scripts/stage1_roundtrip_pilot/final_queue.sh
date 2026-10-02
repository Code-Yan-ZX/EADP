#!/usr/bin/env bash
# Final delivery queue (dispatch 2026-10-03, deadline 11:00 CST).
# Starts automatically once P0 (rtg_full_run_all.sh) exits.
# P1 fresh RTG arms + 4-arm perf; P2 nonreg pairings (per-dataset complete
# pairs first); P3 E_MAIN025 nonreg + EADP-anchor fresh ablations.
set -u
cd "$(dirname "$0")"
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
ACV=../anchor_completion_validation
echo "[queue] $(date) waiting for P0 (rtg_full_run_all.sh)..."
while pgrep -f "rtg_full_run_all.sh" >/dev/null; do sleep 120; done
echo "[queue] $(date) P0 finished — starting P1"

# ---- P1: fresh-panel RTG arms + perf --------------------------------------
$PY rtg_accuracy_fresh.py --mode both > p1_fresh.log 2>&1
echo "[queue] $(date) P1 fresh arms rc=$?"
$PY rtg_perf.py > p1_perf.log 2>&1
echo "[queue] $(date) P1 perf rc=$?"

# ---- P2: nonreg pairings, whole-pair priority order ------------------------
NONREG_ORDER="RealWorldQA,MMStar,ChartQA_TEST,POPE,MMBench_DEV_EN_V11"
for ds in ${NONREG_ORDER//,/ }; do
  echo "[queue] $(date) P2 pairing: $ds"
  $PY $ACV/acu_accuracy.py --panel nonreg --arms BASE --ds "$ds" \
    --mode both > "p2_e_$ds.log" 2>&1
  echo "[queue] $(date) P2 $ds E_GATHER rc=$?"
  $PY rtg_bank_full.py --datasets "$ds" > "p2_bank_$ds.log" 2>&1
  $PY rtg_accuracy_full.py --datasets "$ds" --mode both \
    > "p2_r_$ds.log" 2>&1
  echo "[queue] $(date) P2 $ds R_MAIN025 rc=$?"
done

# ---- P3: E_MAIN025 nonreg + EADP-anchor fresh ablations --------------------
$PY $ACV/acu_accuracy.py --panel nonreg --arms MAIN025 --mode both \
  > p3_emain025_nonreg.log 2>&1
echo "[queue] $(date) P3 E_MAIN025 nonreg rc=$?"
$PY $ACV/acu_accuracy.py --panel fresh --arms MAIN100,MAIN_SIM025 \
  --mode both > p3_fresh_eablations.log 2>&1
echo "[queue] $(date) P3 fresh EADP-anchor ablations rc=$?"
echo "[queue] $(date) FINAL QUEUE DONE"
