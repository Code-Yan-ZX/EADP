#!/usr/bin/env bash
# M3-v0 v1: bank with the four added features, 3 student seeds, full grid.
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export PYTHONUNBUFFERED=1

echo "=== re-extract bank v1 $(date +%T)"
python -u scripts/discovery/m3_features.py --tag m3_bank_v1 2>&1 | tee outputs/discovery/m3_features_v1.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "extract failed"; exit 1; }

for s in 0 1 2; do
  echo "=== train student v1 seed $s $(date +%T)"
  python -u scripts/discovery/m3_train.py --tag m3_bank_v1 \
      --out m3_miss_v1_s$s --seed $s 2>&1 | tee outputs/discovery/m3_train_v1_s$s.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "train s$s failed"; exit 1; }
done

echo "=== merge student-independent arms $(date +%T)"
python -u scripts/discovery/m3_merge_arms.py --src m3_accuracy --dst m3_accuracy_v1

echo "=== grid v1 $(date +%T)"
python -u scripts/discovery/m3_accuracy.py --tag m3_accuracy_v1 --resume \
  --student m3_miss_v1_s0 --bank-gate m3_bank_v1 \
  --arms MG-lowimp-r16 MG-maxred-r16 MG-combo-r16 \
         MG-lowimp-r8 MG-lowimp-r32 MG-maxred-r8 MG-maxred-r32 \
  2>&1 | tee outputs/discovery/m3_grid_v1.log
echo "EXIT_V1=${PIPESTATUS[0]} $(date +%T)"

for s in 1 2; do
  echo "=== grid v1 seed $s $(date +%T)"
  python -u scripts/discovery/m3_accuracy.py --tag m3_accuracy_v1 --resume \
    --student m3_miss_v1_s$s --bank-gate m3_bank_v1 \
    --arms MG-lowimp-r16 MG-maxred-r16 \
    2>&1 | tee outputs/discovery/m3_grid_v1_s$s.log
  echo "EXIT_S${s}=${PIPESTATUS[0]} $(date +%T)"
done
