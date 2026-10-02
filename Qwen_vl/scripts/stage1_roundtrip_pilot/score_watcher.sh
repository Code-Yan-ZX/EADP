#!/usr/bin/env bash
# Trigger per-dataset scoring (CPU) as soon as each full-panel shard is
# complete; GPU generation continues untouched in parallel.
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
declare -A N=( [TextVQA_VAL]=5000 [DocVQA_VAL]=5349 [OCRBench]=1000 )
for ds in TextVQA_VAL DocVQA_VAL OCRBench; do
  f="../../outputs/stage1_roundtrip_pilot/full/acc/R_MAIN025/${ds}.json"
  until [ -f "$f" ] && [ "$($PY -c "import json;print(len(json.load(open('$f'))['records']))" 2>/dev/null)" = "${N[$ds]}" ]; do
    sleep 120
  done
  echo "[watcher] $(date) $ds complete -> scoring (CPU)"
  $PY rtg_accuracy_full.py --mode score --datasets "$ds" \
    >> "score_${ds}.log" 2>&1
  echo "[watcher] $(date) $ds scored rc=$?"
done
echo "[watcher] all datasets scored"
