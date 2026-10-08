#!/usr/bin/env bash
set -u
# This archived driver incorrectly sent ScienceQA questions to the GQA loader.
echo "Retired driver: use sqa_arms_official_driver.sh for CQMI SQA and llava_lane_a_v15.sh for GQA." >&2
exit 1
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
cd $LL
export PYTHONPATH=$LL
while ! mkdir /tmp/llava_full_driver.lock 2>/dev/null; do sleep 120; done
trap "rmdir /tmp/llava_full_driver.lock" EXIT
declare -A VTN AZ
ARM_ORDER=(FULL EGATHER LRMAIN025 LRMAIN0125)
VTN[FULL]=0;      AZ[FULL]=false
VTN[EGATHER]=128; AZ[EGATHER]=false
VTN[LRMAIN025]=128; AZ[LRMAIN025]=true
VTN[LRMAIN0125]=64;  AZ[LRMAIN0125]=true
run () { local task=$1 arm=$2 script=$3 qf=$4 imgf=$5 beta=${6:-2.0} extra=${7:-}
  local out=$EV/anchorzip_p3/$task/$arm.jsonl
  local nq=$(wc -l < "$qf" 2>/dev/null || echo 0)
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$nq" ]; then
    echo "[skip] $task/$arm"; return 0; fi
  [ -s "$out" ] && echo "[redone] $task/$arm partial"
  mkdir -p $EV/anchorzip_p3/$task
  local flag=""; [ "${AZ[$arm]}" = true ] && flag="--anchorzip"
  echo "[gen] $task/$arm $(date +%T)"
  $PY $WRAP/$script --model-path $MP --question-file $qf \
    --image-folder $imgf --answers-file $out --temperature 0 \
    --conv-mode vicuna_v1 --visual_token_num ${VTN[$arm]} \
    --beta $beta --alpha 0.5 $flag $extra \
    >> $EV/anchorzip_p3/$task/$arm.log 2>&1 \
    && echo "[ok] $task/$arm rc=0" || echo "[FAIL] $task/$arm"; }
for arm in "${ARM_ORDER[@]}"; do
  run gqa $arm llava_eval_arm_loader.py \
    $EV/scienceqa/llava_test_QCM-LEPA.json $EV/scienceqa 2.0 \
    " "
done
echo done
