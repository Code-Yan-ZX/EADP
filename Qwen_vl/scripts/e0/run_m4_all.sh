#!/usr/bin/env bash
# E0 M4 driver: all arm x K x dataset generation+scoring, sequential,
# resumable (each shard skips finished questions). Includes the amendment-A6
# smoke gate for the downgraded SparseVLM arm: if its 30-question truncation
# rate exceeds the prereg's 2 % bar, the arm is skipped and flagged instead
# of reporting a known-degenerate baseline.
set -uo pipefail
cd /media/disk2/YZX/research/EADP
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() { echo "=== $* ==="; $PY Qwen_vl/scripts/e0/e0_accuracy.py "$@" 2>&1 | tail -3; }

# --- amendment-A6 smoke gate for sparsevlm_norecycle -----------------------
SKIP_SV=no
$PY Qwen_vl/scripts/e0/e0_smoke.py --arms sparsevlm_norecycle:256 \
    > Qwen_vl/outputs/e0/e0_smoke_norecycle.log 2>&1
TRUNC=$($PY - <<'PYEOF'
import json
try:
    d = json.load(open('Qwen_vl/outputs/e0/e0_smoke.json'))
    print(d['n4']['sparsevlm_norecycle|K=256']['smoke']['truncation_rate'])
except Exception as e:
    print('1.0')  # fail closed: if the smoke cannot be read, skip the arm
PYEOF
)
SKIP_SV=$($PY -c "print('yes' if float('${TRUNC:-1.0}') > 0.02 else 'no')")
echo "sparsevlm_norecycle smoke truncation=${TRUNC} skip=${SKIP_SV}"
if [ "$SKIP_SV" = "yes" ]; then
  echo "SPARSEVLM_NORECYCLE_SKIPPED_SMOKE_FAIL" > Qwen_vl/outputs/e0/SPARSEVLM_NORECYCLE_SKIPPED
fi

# B0 once
run --arm b0 --K 1024
# local + ported arms at the three budgets
for K in 256 128 64; do
  for arm in b2 b1 divprune cdpruner hiprune visionzip fastv pdrop sparsevlm_norecycle; do
    if [ "$arm" = "sparsevlm_norecycle" ] && [ "$SKIP_SV" = "yes" ]; then
      echo "=== skipping $arm (smoke gate) ==="; continue
    fi
    run --arm $arm --K $K
  done
done
# R-res variants
run --arm rres --K 256
run --arm rres --K 128
run --arm rres --K 64
# legacy ablations, K=256 OCR panel only
run --arm a1
run --arm a2
echo "M4_GEN_DONE"
