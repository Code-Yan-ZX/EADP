#!/usr/bin/env bash
set -u
while ! mkdir /tmp/llava_full_driver.lock 2>/dev/null; do sleep 120; done
trap "rmdir /tmp/llava_full_driver.lock" EXIT
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
cd $LL; export PYTHONPATH=$LL
declare -A VTN AZ
ARM_ORDER=(FULL EGATHER LRMAIN025 LRMAIN0125)
VTN[FULL]=0; AZ[FULL]=false; VTN[EGATHER]=128; AZ[EGATHER]=false
VTN[LRMAIN025]=128; AZ[LRMAIN025]=true; VTN[LRMAIN0125]=64; AZ[LRMAIN0125]=true
for arm in "${ARM_ORDER[@]}"; do
  out=$EV/anchorzip_p3/gqa/$arm.jsonl
  nq=$(wc -l < $EV/mm-vet/llava-mm-vet.jsonl)
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$nq" ]; then echo "[skip] gqa/$arm"; continue; fi
  mkdir -p $EV/anchorzip_p3/gqa
  flag=""; [ "${AZ[$arm]}" = true ] && flag="--anchorzip"
  echo "[gen] gqa/$arm $(date +%T)"
  $PY $WRAP/llava_eval_arm_loader.py --model-path $MP \
    --question-file $EV/gqa/llava_gqa_testdev_balanced.jsonl --image-folder $EV/gqa/data/images \
    --answers-file $out --temperature 0 --conv-mode vicuna_v1 \
    --visual_token_num ${VTN[$arm]} --beta 2.0 --alpha 0.5 $flag \
    >> $EV/anchorzip_p3/gqa/$arm.log 2>&1 \
    && echo "[ok] gqa/$arm" || echo "[FAIL] gqa/$arm"
done
echo "[gqa] done $(date)"
