#!/usr/bin/env bash
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1
python -u scripts/discovery/m2_perf_paired.py \
  --tags B2 MG OR --blocks 60 --warmup-blocks 10 --tag m3_perf_paired \
  2>&1 | tee outputs/discovery/m3_perf.log
echo "EXIT=${PIPESTATUS[0]} $(date +%T)"
