#!/usr/bin/env bash
# Anchor Completion Validation — sequential GPU chain (single A40).
# Stage B waits for the already-running main-panel bank job to finish,
# runs the correctness gates, and ONLY IF ALL GATES PASS launches the
# frozen generation plan.  Every stage logs to its own file; any gate
# failure stops the chain (never bypass a gate to reach accuracy runs).
set -u
cd "$(dirname "$0")"
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python

echo "[chain] $(date) waiting for main-panel banks..."

# ---- wait for the three main banks on disk -------------------------------
for ds in TextVQA_VAL DocVQA_VAL OCRBench; do
  until [ -f "../../outputs/anchor_completion_validation/bank_full_${ds}.json.gz" ]; do
    sleep 60
  done
  echo "[chain] $(date) bank ready: ${ds}"
done
# ensure the bank builder process has exited (no partial gz writes)
while pgrep -f "acu_bank.py --datasets TextVQA_VAL" >/dev/null; do sleep 30; done
echo "[chain] $(date) bank builder exited"

# ---- gates G2 G5 G3 G4 ----------------------------------------------------
$PY acu_correctness.py --gates G2,G5 > gates_g2g5.log 2>&1
G25=$?
$PY acu_correctness.py --gates G3,G4 --per-ds 3 > gates_g3g4.log 2>&1
G34=$?
echo "[chain] $(date) gates exit codes: G2/G5=$G25 G3/G4=$G34"
$PY - <<'EOF'
import json
r = json.load(open("../../outputs/anchor_completion_validation/correctness.json"))
ok = all(r.get(g, {}).get("ok") for g in ("G1", "G2", "G3", "G4", "G5"))
print("ALL_GATES_PASS" if ok else "GATES_FAILED")
raise SystemExit(0 if ok else 1)
EOF
if [ $? -ne 0 ]; then
  echo "[chain] GATES FAILED — chain stopped, no accuracy runs."
  exit 1
fi

# ---- smoke (5 rows/task, gen+score; S0 semantics apply) -------------------
$PY acu_accuracy.py --panel main --arms BASE,MAIN025 --smoke 5 --mode both \
  > smoke_main.log 2>&1
S=$?
echo "[chain] $(date) smoke exit=$S"
if [ $S -ne 0 ]; then
  echo "[chain] SMOKE FAILED — chain stopped."
  exit 1
fi

# ---- full main panel: BASE + MAIN025 --------------------------------------
$PY acu_accuracy.py --panel main --arms BASE,MAIN025 --mode both \
  > main_run.log 2>&1
M=$?
echo "[chain] $(date) main panel exit=$M"

# ---- G3 verify: gate greedy outputs vs accuracy shards (bitwise) ----------
$PY acu_correctness.py --verify-g3 > g3_verify.log 2>&1
V3=$?
echo "[chain] $(date) g3 verify exit=$V3"
if [ $V3 -ne 0 ]; then
  echo "[chain] G3 VERIFY FAILED — chain stopped."
  exit 1
fi

# ---- nonreg banks + panel --------------------------------------------------
$PY acu_bank.py --datasets ChartQA_TEST,MMBench_DEV_EN_V11,MMStar,RealWorldQA,POPE \
  > bank_nonreg.log 2>&1
$PY acu_accuracy.py --panel nonreg --arms BASE,MAIN025 --mode both \
  > nonreg_run.log 2>&1
N=$?
echo "[chain] $(date) nonreg panel exit=$N"

# ---- fresh-panel ablations --------------------------------------------------
# formal-method fresh entries are DERIVED from the main panel (no regen);
# ablation arms generate on the fresh rows only
$PY acu_accuracy.py --panel fresh --arms BASE,MAIN025 --mode score \
  > fresh_derive.log 2>&1
$PY acu_accuracy.py --panel fresh --arms MAIN100,MAIN_SIM025 --mode both \
  > fresh_run.log 2>&1
F=$?
echo "[chain] $(date) fresh ablations exit=$F"

# ---- perf -------------------------------------------------------------------
$PY acu_perf.py > perf_run.log 2>&1
P=$?
echo "[chain] $(date) perf exit=$P"

# ---- analysis ----------------------------------------------------------------
$PY acu_analyze.py > analyze_run.log 2>&1
A=$?
echo "[chain] $(date) analyze exit=$A"
echo "[chain] $(date) CHAIN DONE main=$M nonreg=$N fresh=$F perf=$P analyze=$A"
