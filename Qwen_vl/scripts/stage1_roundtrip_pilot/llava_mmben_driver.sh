#!/usr/bin/env bash
# MMBench-EN V11 two-model two-arm driver (2026-10-06, user approved).
# AnchorZip arms only: LRMAIN025 (128/crop) + LRMAIN0125 (64/crop).
# Official wrapper semantics: --single-pred-prompt --temperature 0
# --conv-mode vicuna_v1; circular scoring via mmben_circular_score.py
# (1292 groups, group hit iff all rotation rows correct).
set -u
while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5000 ] && break
  echo "[wait] GPU ${u}MiB $(date +%T)"; sleep 300
done
if ! mkdir /tmp/llava_full_driver.lock 2>/dev/null; then
  echo "[mmben] lock held, exit"; exit 1
fi
trap "rmdir /tmp/llava_full_driver.lock" EXIT

PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
cd $LL; export PYTHONPATH=$LL
TSV=$EV/mmbench/MMBench_DEV_EN_V11.tsv

run () { local mp=$1 out=$2 vtn=$3
  local nq=$($PY -c "import pandas as pd; print(len(pd.read_table('$TSV')))")
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$nq" ]; then
    echo "[skip] $out"; return 0; fi
  mkdir -p $(dirname $out)
  echo "[gen] $out vtn=$vtn $(date +%T)"
  $PY -m llava.eval.model_vqa_mmbench \
    --model-path $mp \
    --question-file $TSV \
    --answers-file $out \
    --visual_token_num $vtn --anchorzip --beta 2.0 --alpha 0.5 \
    --single-pred-prompt --temperature 0 --conv-mode vicuna_v1 \
    >> ${out%.jsonl}.log 2>&1 \
    && echo "[ok] $out" || echo "[FAIL] $out"; }

V15=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
NEXT=/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
run $V15 $EV/anchorzip_p3/mmben/LRMAIN025.jsonl 128
run $V15 $EV/anchorzip_p3/mmben/LRMAIN0125.jsonl 64
run $NEXT $EV/anchorzip_p3/next_mmben/LRMAIN025.jsonl 128
run $NEXT $EV/anchorzip_p3/next_mmben/LRMAIN0125.jsonl 64

# scoring
for out in $EV/anchorzip_p3/mmben/LRMAIN025.jsonl $EV/anchorzip_p3/mmben/LRMAIN0125.jsonl \
           $EV/anchorzip_p3/next_mmben/LRMAIN025.jsonl $EV/anchorzip_p3/next_mmben/LRMAIN0125.jsonl; do
  [ -s "$out" ] && $PY $WRAP/mmben_circular_score.py --question-file $TSV \
    --result-file $out --out ${out%.jsonl}.score.json
done
echo "[mmben] all done $(date)"
