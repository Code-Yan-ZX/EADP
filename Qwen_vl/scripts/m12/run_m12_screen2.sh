#!/usr/bin/env bash
# M12 screen 2: Pareto budgets + opponent/base/single-scale ablations.
# Launch only after screen 1 shows a live direction.
set -e
cd "$(dirname "$0")/../.."
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
DS=TextVQA_VAL,DocVQA_VAL,OCRBench

# Pareto: retinagate + rres at the other budgets (512 exists from screen 1)
for K in 768 384 256; do
  echo "=== retinagate K=$K ==="
  $PY scripts/m12/m12_accuracy.py --arm retinagate --K $K --ds $DS --mode both 2>&1 | tail -4
  echo "=== rres K=$K ==="
  $PY scripts/m12/m12_accuracy.py --arm rres --K $K --ds $DS --mode both 2>&1 | tail -4
  echo "=== random K=$K ==="
  $PY scripts/m12/m12_accuracy.py --arm random --K $K --ds $DS --mode both 2>&1 | tail -4
done

# Ablations at the main operating point K=512
for arm in event_s1 retinagate_s1; do
  echo "=== $arm K=512 ==="
  $PY scripts/m12/m12_accuracy.py --arm $arm --K 512 --ds $DS --mode both 2>&1 | tail -4
done
echo "screen2 done"
