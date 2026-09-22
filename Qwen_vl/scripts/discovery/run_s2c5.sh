#!/usr/bin/env bash
# S2-C5 -- token-local vs set-dependent factorization diagnosis.
#
# The LLM and the vision tower are never loaded: every input is the S2-C1
# layer-4 feature cache, the S2-B P1-G2 teacher maps and the cached score banks.
# Only the tiny scorers below are trained (0.52 M - 2.36 M parameters).
#
# No downstream generation is run in the normal path. The brief gates it on a
# deployable arm reaching held-out head_recall@8 clearly above HEAD_RANK with a
# paired bootstrap CI > 0 and an absolute level of ~0.82 (ideally >= 0.85).
# Step 4 evaluates that gate and records the outcome; step 5 is the command to
# use only if it is ever met.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
cd "$(dirname "$0")/../.."

TAG=${TAG:-s2c5}

# 1. C5-0 candidate accessibility diagnosis -- no training, pure read-out of the
#    cached frozen-baseline score banks (~1 min)
python scripts/discovery/s2c5_c0_diagnosis.py --tag "${TAG}_c0"

# 2. the scorer ladder: LOCAL-MLP / GLOBAL-CTX / SET-CTX / LOCAL-MLP-WIDE,
#    3 seeds each, early-stopped on validation Top-256 overlap exactly as
#    S2-C3 did (~30 min on one A40; the attention arm and the 4x-wide
#    diagnostic dominate)
python scripts/discovery/s2c5_train.py --tag "$TAG"

# 3. mechanism controls beyond the in-training wrong-image pass: seed-level
#    paired bootstrap against HEAD_RANK, per-benchmark decomposition
python scripts/discovery/s2c5_controls.py --tag "$TAG"

# 4. apply the pre-registered decision rule and write the main JSON
python scripts/discovery/s2c5_consolidate.py --tag "$TAG"

# 5. figure
python scripts/discovery/s2c5_figure.py --tag "$TAG"

# 6. ONLY if consolidate reports downstream_gate.downstream_run == true:
# python scripts/discovery/s2c0_run.py --scores "${TAG}_scores.npz" \
#     --arms <the arm(s) that cleared the gate> --tag "${TAG}_pilot"

echo
echo "S2-C5 complete. Main artifact: outputs/discovery/${TAG}_factorization.json"
