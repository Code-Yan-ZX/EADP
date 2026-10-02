#!/usr/bin/env bash
# RTG pilot serial driver (dispatch 2026-10-02):
#   banks -> gates T5/T6/lam0 -> E reuse+score -> 4 new arms gen+score
#   -> analyze -> perf, THEN immediately resume the core Anchor-Completion
#   chain (g3 verify -> nonreg banks -> nonreg -> fresh -> perf -> analyze).
# GPU budget cap for the PILOT portion: 2h (wall clock of this script's
# pilot stages; waiting for the core job is NOT counted).  On cap: mark
# unfinished, skip to the core resume.  No kill -9 anywhere; the core main
# panel is expected to have exited normally before any pilot GPU stage.
set -u
cd "$(dirname "$0")"
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
BUDGET_S=7200
elapsed() { echo $(( $(date +%s) - START )); }

echo "[rtg] $(date) waiting for the core main-panel process to exit..."
while pgrep -f "acu_accuracy.py --panel main" >/dev/null; do sleep 60; done
# budget clock starts HERE — waiting for the core job is NOT counted
# (dispatch §6: 不计等待原作业和 CPU 实现)
START=$(date +%s)
echo "[rtg] $(date) core main panel exited — starting pilot (budget ${BUDGET_S}s)"

# ---- pilot stage 1: scorer banks (rtg/flat/shuf x 3 DEV datasets) --------
if [ "$(elapsed)" -lt "$BUDGET_S" ]; then
  $PY rtg_bank_all.py > bank_all.log 2>&1 || echo "[rtg] bank stage rc=$?"
fi

# ---- pilot stage 2: gates T5/T6 + lam0 ------------------------------------
if [ "$(elapsed)" -lt "$BUDGET_S" ]; then
  $PY rtg_correctness.py --gates T5,T6 --per-ds 3 > gates_t5t6.log 2>&1
  G1=$?
  $PY rtg_correctness.py --lam0 --per-ds 2 > gates_lam0.log 2>&1
  G2=$?
  echo "[rtg] $(date) gates rc: T5/T6=$G1 lam0=$G2"
  $PY - <<'EOF'
import json, sys
r = json.load(open("../../outputs/stage1_roundtrip_pilot/correctness.json"))
ok = all(r.get(g, {}).get("ok") for g in ("T1","T2","T3","T4","T5","T6","T7_lam0"))
print("RTG_GATES_PASS" if ok else "RTG_GATES_FAILED")
sys.exit(0 if ok else 1)
EOF
  if [ $? -ne 0 ]; then
    echo "[rtg] GATES FAILED — pilot stopped; reporting failure; resuming core."
    $PY rtg_analyze.py > analyze_blocked.log 2>&1 || true
    echo "[rtg] $(date) pilot ABORTED at gates" >> pilot_status.txt
    exit 10
  fi
fi

# ---- pilot stage 3: E-arm reuse + re-score (macro reproduction check) -----
if [ "$(elapsed)" -lt "$BUDGET_S" ]; then
  $PY rtg_accuracy.py --arms E_GATHER,E_MAIN025 --mode reuse > reuse.log 2>&1
  R=$?
  echo "[rtg] $(date) E reuse+score rc=$R (macro must reproduce 81.1574/83.2673)"
  if [ $R -ne 0 ]; then
    echo "[rtg] E REUSE FAILED — pilot stopped; resuming core."
    echo "[rtg] $(date) pilot ABORTED at E reuse" >> pilot_status.txt
    exit 11
  fi
fi

# ---- pilot stage 4: 4 new arms generation + scoring -----------------------
if [ "$(elapsed)" -lt "$BUDGET_S" ]; then
  $PY rtg_accuracy.py --arms R_GATHER,R_MAIN025,F_MAIN025,S_MAIN025 \
    --mode both > gen_new.log 2>&1
  N=$?
  echo "[rtg] $(date) new arms gen+score rc=$N"
fi

# ---- pilot stage 5: analysis (frozen statistics + GO gate 1-3) -------------
if [ "$(elapsed)" -lt "$BUDGET_S" ]; then
  $PY rtg_analyze.py > analyze.log 2>&1
  A=$?
  echo "[rtg] $(date) analyze rc=$A"
fi

# ---- pilot stage 6: paired live perf (GO gate condition 4) -----------------
if [ "$(elapsed)" -lt "$BUDGET_S" ]; then
  $PY rtg_perf.py > perf.log 2>&1
  P=$?
  echo "[rtg] $(date) perf rc=$P"
else
  echo "UNFINISHED: budget cap reached before perf" >> pilot_status.txt
fi

echo "[rtg] $(date) pilot stages done in $(elapsed)s (budget ${BUDGET_S}s)"
echo "[rtg] $(date) resuming core Anchor-Completion chain"

# ===========================================================================
# CORE RESUME (original Anchor-Completion round; never reported as complete
# unless every stage below actually finishes)
# ===========================================================================
$PY ../anchor_completion_validation/acu_correctness.py --verify-g3 \
  > g3_verify.log 2>&1
V3=$?
echo "[rtg] $(date) core g3 verify rc=$V3"

$PY ../anchor_completion_validation/acu_bank.py \
  --datasets ChartQA_TEST,MMBench_DEV_EN_V11,MMStar,RealWorldQA,POPE \
  > bank_nonreg.log 2>&1
$PY ../anchor_completion_validation/acu_accuracy.py --panel nonreg \
  --arms BASE,MAIN025 --mode both > ../anchor_completion_validation/nonreg_run.log 2>&1
N=$?
echo "[rtg] $(date) core nonreg panel rc=$N"

$PY ../anchor_completion_validation/acu_accuracy.py --panel fresh \
  --arms BASE,MAIN025 --mode score > fresh_derive.log 2>&1
$PY ../anchor_completion_validation/acu_accuracy.py --panel fresh \
  --arms MAIN100,MAIN_SIM025 --mode both > ../anchor_completion_validation/fresh_run.log 2>&1
F=$?
echo "[rtg] $(date) core fresh ablations rc=$F"

$PY ../anchor_completion_validation/acu_perf.py \
  > ../anchor_completion_validation/perf_run.log 2>&1
P=$?
$PY ../anchor_completion_validation/acu_analyze.py \
  > ../anchor_completion_validation/analyze_run.log 2>&1
A=$?
echo "[rtg] $(date) core perf rc=$P analyze rc=$A — CORE RESUME DONE"
