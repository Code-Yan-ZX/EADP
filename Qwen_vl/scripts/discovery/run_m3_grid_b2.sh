#!/usr/bin/env bash
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1
python -u scripts/discovery/m3_accuracy.py --resume \
  --arms MG-lowimp-r8 MG-lowimp-r32 MG-maxred-r8 MG-maxred-r32 \
         OR-lowimp-r8 OR-lowimp-r32 \
  2>&1 | tee outputs/discovery/m3_grid_b2.log
echo "EXIT=${PIPESTATUS[0]} $(date +%T)"
