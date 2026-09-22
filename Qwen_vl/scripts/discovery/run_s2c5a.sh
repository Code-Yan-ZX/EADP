#!/usr/bin/env bash
# S2-C5A -- checkpoint / metric consistency audit for S2-C5.
#
# This stage trains nothing new and changes no architecture, feature or
# hyperparameter. It exists to rule out one methodological objection to S2-C5
# before S2-C5's verdict is allowed to stand:
#
#   S2-C5 supervised on HEAD_RANK and reports held-out teacher Top-8 recall@256,
#   but selected its checkpoints on validation Top-256 *overlap*. S2-C2 and S2-C3
#   both measured those two quantities in tension, so the ~0.795 landing point
#   could be a checkpoint-selection artefact rather than a property of the
#   token-local family.
#
# Step 1 re-selects every checkpoint on validation head_recall@8 instead, inside
# the same trajectory (so the comparison is exactly paired), and reports both
# selections side by side. Step 2 reproduces and decomposes the S2-C4 (0.7783)
# vs S2-C5 (0.7656) HEAD_RANK discrepancy, which is an aggregation-protocol
# difference and not a result difference. Step 3 applies the pre-registered
# S2-C5 rule to the re-selected arms. Step 4 is the audit report.
#
# The held-out 150 is never used to select anything in any step; it is only ever
# measured, after the checkpoints are frozen.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
cd "$(dirname "$0")/../.."

TAG=${TAG:-s2c5a}

# 1. the ladder re-run with dual checkpoint tracking: identical loop, identical
#    RNG order, identical patience and epoch budget to s2c5_train.py, with an OV
#    tracker and an H8 tracker over the one trajectory (~20 min on one A40)
python scripts/discovery/s2c5a_train.py --tag "$TAG"

# 2. both selections scored on the frozen held-out 150, paired bootstrap CIs,
#    reproduction check against the published S2-C5 run, verdict re-application
python scripts/discovery/s2c5a_consolidate.py --tag "$TAG"

# 3. the aggregation audit that explains 0.7783 vs 0.7656 (no training)
python scripts/discovery/s2c5a_aggregation.py --tag "${TAG}_aggregation"

echo
echo "S2-C5A complete. Artifacts:"
echo "  outputs/discovery/${TAG}_train.json          (both selections, raw)"
echo "  outputs/discovery/${TAG}_audit.json          (the audit)"
echo "  outputs/discovery/${TAG}_aggregation.json    (0.7783 vs 0.7656)"
