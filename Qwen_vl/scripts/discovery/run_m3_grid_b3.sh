#!/usr/bin/env bash
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1
python -u scripts/discovery/m3_accuracy.py --resume \
  --arms RND-maxred-r8 RND-maxred-r32 RND-combo-r8 RND-combo-r32 \
  2>&1 | tee outputs/discovery/m3_grid_b3.log
echo "EXIT=${PIPESTATUS[0]} $(date +%T)"
