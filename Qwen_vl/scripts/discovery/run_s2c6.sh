#!/usr/bin/env bash
# S2-C6 -- residual signal localization.
#
# Three training ladders and one consolidation, all offline on the S2-C1 caches.
#
#   Part A  the identical LOCAL-MLP config at n = 60/120/180/240 nested,
#           benchmark-stratified fit subsets: is the curve still rising at 240,
#           i.e. is ~0.80 a data limit rather than a representation ceiling?
#   Part B  L2, DELTA = h4 - h2, L2+L4, L4+DELTA -- each parameter-matched to
#           L4 -- plus the L4+L4 architecture control, on the full fit set.
#   Part C  L4 plus a low-rank additive query term, with a shuffled-query
#           control, re-asking S2-C1's query question under the current
#           ranking-head objective.
#
# The pre-registration (docs/scoring_search_s2c6_prereg.md) was written before
# any of this ran; the decision rule and the verdicts live there and in
# s2c6_consolidate.py. The held-out 150 is never used to select anything: every
# checkpoint is selected on validation teacher Top-8 recall@256, and the
# held-out set is measured once per frozen checkpoint.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
cd "$(dirname "$0")/../.."

TAG=${TAG:-s2c6}

# 1. Part A: teacher-data scaling (~5 min)
python scripts/discovery/s2c6_train.py --part A --tag "$TAG"

# 2. Part B: representation localization (~20 min)
python scripts/discovery/s2c6_train.py --part B --tag "$TAG"

# 3. Part C: query re-check under the current objective (~6 min for two arms).
#    L4+QUERY is the pre-registered token-independent arm; L4+QRY-BILIN is the
#    post-hoc token-dependent amendment at an identical added budget (see
#    docs/scoring_search_s2c6.md 7).
python scripts/discovery/s2c6_train.py --part C --tag "$TAG"

# 4. apply the pre-registered rule; writes the stage artifact
python scripts/discovery/s2c6_consolidate.py --tag "$TAG" \
    | tee outputs/discovery/${TAG}_consolidate.txt

# 5. what the query changes at the selection level -- the measurement that
#    separates "the arm ignored the question" from "the question carries
#    nothing" (no training; reads the frozen checkpoints)
python scripts/discovery/s2c6_query_control.py --tag "$TAG"

# 6. tables and figure, rendered from the audit artifact rather than retyped
python scripts/discovery/s2c6_tables.py --tag "$TAG" \
    > outputs/discovery/${TAG}_tables.md
python scripts/discovery/s2c6_figure.py --tag "$TAG"

echo
echo "S2-C6 complete. Artifacts:"
echo "  outputs/discovery/${TAG}_audit.json        (verdicts, main artifact)"
echo "  outputs/discovery/${TAG}_train_{A,B,C}.json"
echo "  outputs/discovery/${TAG}_perimage_{A,B,C}.npz"
echo "  outputs/discovery/${TAG}_query_control.json"
echo "  outputs/discovery/${TAG}_tables.md, figures/${TAG}_localization.png"