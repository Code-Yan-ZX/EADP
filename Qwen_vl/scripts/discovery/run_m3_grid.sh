#!/usr/bin/env bash
# M3-v0 held-out grid. Session A: r=16, all rules + controls.
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1
python -u scripts/discovery/m3_accuracy.py --resume \
  --arms MG0 \
         MG-lowimp-r16 MG-maxred-r16 MG-combo-r16 \
         OR-lowimp-r16 OR-maxred-r16 OR-combo-r16 OR-teacher-r16 \
         RND-lowimp-r16 RND-combo-r16 \
  2>&1 | tee outputs/discovery/m3_grid_a.log
echo "EXIT=${PIPESTATUS[0]} $(date +%T)"
