#!/usr/bin/env bash
# E0 M4 (amendment A7): prioritized trend-first queue.
# P0 = D1 (a1/a2), D2/D4 (rres512 + 24/25 methods), D3 (pace) -- all on the
# OCR panel at K=256.  P1 = b0 + the general-panel trend pass.  P2 = the
# K=64 curve endpoint.  Deferred arms (A7): b1, cdpruner, hiprune, K=128.
set -uo pipefail
cd /media/disk2/YZX/research/EADP
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() { echo "=== $* ==="; $PY Qwen_vl/scripts/e0/e0_accuracy.py "$@" 2>&1 | tail -3; }

# --- amendment-A6 smoke gate for sparsevlm_norecycle ------------------------
SKIP_SV=no
$PY Qwen_vl/scripts/e0/e0_smoke.py --arms sparsevlm_norecycle:256 \
    > Qwen_vl/outputs/e0/e0_smoke_norecycle.log 2>&1
TRUNC=$($PY - <<'PYEOF'
import json
try:
    d = json.load(open('Qwen_vl/outputs/e0/e0_smoke.json'))
    print(d['n4']['sparsevlm_norecycle|K=256']['smoke']['truncation_rate'])
except Exception:
    print('1.0')   # fail closed
PYEOF
)
SKIP_SV=$($PY -c "print('yes' if float('${TRUNC:-1.0}') > 0.02 else 'no')")
echo "sparsevlm_norecycle smoke truncation=${TRUNC} skip=${SKIP_SV}"
[ "$SKIP_SV" = "yes" ] && \
  echo "SPARSEVLM_NORECYCLE_SKIPPED_SMOKE_FAIL" > Qwen_vl/outputs/e0/SPARSEVLM_NORECYCLE_SKIPPED

OCR_DS="--ds TextVQA_VAL,DocVQA_VAL,OCRBench,ChartQA_TEST"
GEN_DS="--ds MMBench_DEV_EN_V11,MMStar,RealWorldQA,POPE"

# --- P0: the D1/D2/D4 arms (OCR panel, K=256) -------------------------------
run --arm a1 $OCR_DS
run --arm a2 $OCR_DS
run --arm b2 --K 256 $OCR_DS
run --arm rres --K 256 $OCR_DS
run --arm fastv --K 256 $OCR_DS
run --arm pdrop --K 256 $OCR_DS
run --arm visionzip --K 256 $OCR_DS
[ "$SKIP_SV" = "no" ] && run --arm sparsevlm_norecycle --K 256 $OCR_DS
run --arm divprune --K 256 $OCR_DS
[ -f /tmp/e0_pace_ok ] && run --arm pace --K 256 $OCR_DS

# --- P1: b0 + full-arm general-panel pass (K=256; A7 extension after the
#     user asked to keep more than the OCR panel) ------------------------------
run --arm b0 --K 1024 $OCR_DS
run --arm b0 --K 1024 $GEN_DS
for arm in b2 rres fastv pdrop visionzip divprune; do
  run --arm $arm --K 256 $GEN_DS
done
[ "$SKIP_SV" = "no" ] && run --arm sparsevlm_norecycle --K 256 $GEN_DS
[ -f /tmp/e0_pace_ok ] && run --arm pace --K 256 $GEN_DS

# --- P2: K=64 curve endpoint (OCR panel) ------------------------------------
run --arm b2 --K 64 $OCR_DS
run --arm rres --K 64 $OCR_DS
run --arm fastv --K 64 $OCR_DS
run --arm visionzip --K 64 $OCR_DS
run --arm pdrop --K 64 $OCR_DS
[ -f /tmp/e0_pace_ok ] && run --arm pace --K 64 $OCR_DS

echo "M4_PRIORITY_DONE"
