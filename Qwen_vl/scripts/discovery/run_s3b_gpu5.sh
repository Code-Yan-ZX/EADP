#!/usr/bin/env bash
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1
while ! grep -q "EXT2 STAGES DONE\|STOPPING" outputs/discovery/s3b_chain4.log; do sleep 15; done
grep -q "STOPPING" outputs/discovery/s3b_chain4.log && { echo "chain4 failed"; exit 1; }
python -u scripts/discovery/s3b_poscheck.py 2>&1 | tee outputs/discovery/s3b_poscheck.log
echo "POSCHECK DONE $(date +%T)"
