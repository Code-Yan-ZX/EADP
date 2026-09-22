#!/usr/bin/env bash
# M1 -- is the L4 LOCAL-MLP gain data, or optimizer steps?
#
# Offline. Same arm (L4 / LOCAL-MLP), same target (HEAD_RANK), same loss, same
# optimizer, same selection rule as S2-C6. No query, no trajectory, no context,
# no wider model. The one thing that changes is the fit pool -- and, in the
# primary protocol, it is the *only* thing that changes, because the update
# budget is held fixed at 900 for every n.
#
# Stages, in order. Each one is a gate on the next:
#
#   0  m1_plan.py       freeze the 720-instance extension draw and audit the row
#                       space against S2-C6's                         (seconds)
#   1  m1_features.py   cache L4 for the extension rows; G1 reproduces the
#                       published cache on 8 sampled instances         (~6 min)
#   2  m1_teacher.py    P1-G2 teacher maps for the extension rows; G2 reproduces
#                       the published maps on 8 sampled instances      (~9 min)
#   3  m1_train.py      gate G3: fixed-step n=240 must reproduce S2-C6's A240
#                       epoch by epoch                                  (~2 min)
#   4  m1_train.py      primary  fixed-step,  n = 60/120/180/240/480/960  (~25 min)
#   5  m1_train.py      secondary fixed-epoch, n = 240/480/960             (~15 min)
#   6  m1_consolidate.py apply the pre-registered rule; write the artifact
#
# Stage 3 stops the chain on failure: if fixed-step n=240 does not reproduce the
# published run, the step/data decomposition is not interpretable and running the
# rest would produce numbers nobody should read.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
cd "$(dirname "$0")/../.."

TAG=${TAG:-m1}
D=scripts/discovery
O=outputs/discovery

# ---------------------------------------------------------------- data ------
python $D/m1_plan.py                        | tee $O/${TAG}_plan.txt
if [ ! -f "$O/m1_feats_L4_extra.npy" ]; then
    python $D/m1_features.py                | tee $O/${TAG}_features.txt
fi
if [ ! -f "$O/m1_gradient_scores_extra.npz" ]; then
    python $D/m1_teacher.py                 | tee $O/${TAG}_teacher.txt
fi
# is the extension pool comparable to the pool it extends? (CPU only)
python $D/m1_pool_check.py --tag "$TAG"     | tee $O/${TAG}_pool_check.txt

# ------------------------------------------------- gate G3, then the grid ----
# n=240 first and on its own: this is the check that the fixed-step protocol is a
# controlled re-parameterisation of S2-C6's run rather than a new configuration.
python $D/m1_train.py --protocol fixed-step --ns 240 --check-c6 \
    --tag "${TAG}_gate" | tee $O/${TAG}_gate_g3.txt

python $D/m1_train.py --protocol fixed-step --ns 60 120 180 240 480 960 \
    --tag "$TAG" | tee $O/${TAG}_train_fixed-step.txt
python $D/m1_train.py --protocol fixed-epoch --ns 240 480 960 \
    --tag "$TAG" | tee $O/${TAG}_train_fixed-epoch.txt

python $D/m1_consolidate.py --tag "$TAG" | tee $O/${TAG}_consolidate.txt

# 7. tables rendered from the audit artifact rather than retyped, and the
#    downstream comparison assembled and checked but deliberately not run
python $D/m1_tables.py --tag "$TAG" > $O/${TAG}_tables.md
python $D/m1_downstream.py --check | tee $O/${TAG}_downstream_ready.txt

# 8. recompute every claim from the raw per-image artifacts; exits non-zero if
#    the audit and the data disagree anywhere
python $D/m1_verify.py --tag "$TAG" | tee $O/${TAG}_verify.txt

echo
echo "M1 complete. Artifacts:"
echo "  $O/${TAG}_audit.json                 (verdicts, main artifact)"
echo "  $O/${TAG}_train_fixed-step.json      (primary protocol records)"
echo "  $O/${TAG}_train_fixed-epoch.json     (secondary protocol records)"
echo "  $O/${TAG}_perimage_*.npz             (per-image held-out metrics)"
echo "  $O/${TAG}_plan.json                  (the frozen extension draw)"
echo "  $O/${TAG}_feature_repro.json, ${TAG}_teacher_repro.json  (gates G1, G2)"
