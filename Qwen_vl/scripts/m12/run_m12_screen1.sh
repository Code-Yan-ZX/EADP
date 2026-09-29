#!/usr/bin/env bash
# M12 screen 1: K=512 (50% of 1024 merged tokens), OCR panel + ChartQA.
# Sequential, single GPU. Baselines b0/b2/rres256 reused from E0 shards.
set -e
cd "$(dirname "$0")/../.."
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
DS=TextVQA_VAL,DocVQA_VAL,OCRBench
K=512

for arm in retinagate rres event graydog sobel variance uniform random; do
  echo "=== arm $arm K=$K ==="
  $PY scripts/m12/m12_accuracy.py --arm $arm --K $K --ds $DS \
      --mode both 2>&1 | tail -6
done
echo "screen1 done"
