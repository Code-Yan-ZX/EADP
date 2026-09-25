#!/usr/bin/env bash
# M4-v0 -- the paired latency measurement (brief §11: only if the accuracy gate
# reaches PROMISING or STRONG).
#
# **NOT RUN.** The M4-v0 gate returned REFUTED, so per §11 the paired protocol
# was skipped and the report makes no TTFT claim beyond the capsule stage's own
# window. This script is kept so that the skipped step is auditable rather than
# merely absent, and so it can be run unchanged if the line is ever revisited.
#
# Same protocol as M2's amendment and M3's run: every arm is measured inside
# every block in a randomised order, so the contrast is PAIRED within blocks and
# host drift cancels.  REC adds exactly one stage to B2 -- the capsule
# construction, reported as `miss_ms` in the stage table -- and the LLM prefill
# is unchanged because the budget is still 256 visual tokens.
set -u
cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1
python -u scripts/discovery/m2_perf_paired.py \
  --tags B2 REC REC8 REC32 MEAN \
  --blocks 70 --warmup-blocks 12 --tag m4_perf_paired \
  2>&1 | tee outputs/discovery/m4_perf.log
echo "EXIT=${PIPESTATUS[0]} $(date +%T)"
