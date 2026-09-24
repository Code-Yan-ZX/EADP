#!/usr/bin/env bash
# S3-B GPU stages only (structure/figure are CPU and run afterwards)
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1
for s in s3b_base s3b_sweep s3b_ablate s3b_crop; do
  echo "=== $s $(date +%T)"
  python -u scripts/discovery/$s.py 2>&1 | tee outputs/discovery/$s.log
  rc=${PIPESTATUS[0]}
  echo "=== $s exit=$rc $(date +%T)"
  [ "$rc" -eq 0 ] || { echo "STOPPING: $s failed"; exit "$rc"; }
done
echo "GPU STAGES DONE $(date +%T)"
