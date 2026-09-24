#!/usr/bin/env bash
# wait for the running chain, then measure the two extension stages
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1
while ! grep -q "RESUME STAGES DONE\\|STOPPING" outputs/discovery/s3b_chain2.log; do sleep 20; done
grep -q "STOPPING" outputs/discovery/s3b_chain2.log && { echo "chain2 failed, aborting"; exit 1; }
for s in s3b_subst s3b_bands; do
  echo "=== $s $(date +%T)"
  python -u scripts/discovery/$s.py 2>&1 | tee outputs/discovery/$s.log
  rc=${PIPESTATUS[0]}
  echo "=== $s exit=$rc $(date +%T)"
  [ "$rc" -eq 0 ] || exit $rc
done
echo "EXT STAGES DONE $(date +%T)"
