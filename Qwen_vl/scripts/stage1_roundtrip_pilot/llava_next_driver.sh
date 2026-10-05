#!/usr/bin/env bash
# NeXT (llava-v1.6-vicuna-7b) four-arm driver. Per-crop budgets:
# E_GATHER/L_R_MAIN025 = 128/crop (total ~640), K64 arm = 64/crop.
# Lock-serialised with the v1.5 drivers.
set -u
if ! mkdir /tmp/llava_full_driver.lock 2>/dev/null; then
  echo "[next] lock held, exit"; exit 1
fi
trap "rmdir /tmp/llava_full_driver.lock" EXIT
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP=/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
cd $LL; export PYTHONPATH=$LL
declare -A VTN AZ
ARM_ORDER=(FULL EGATHER LRMAIN025 LRMAIN0125)
VTN[FULL]=0; AZ[FULL]=false; VTN[EGATHER]=128; AZ[EGATHER]=false
VTN[LRMAIN025]=128; AZ[LRMAIN025]=true; VTN[LRMAIN0125]=64; AZ[LRMAIN0125]=true

run () { local task=$1 arm=$2 script=$3 qf=$4 imgf=$5 beta=${6:-2.0} extra=${7:-}
  local out=$EV/anchorzip_p3/next_$task/$arm.jsonl
  local nq=$(wc -l < "$qf" 2>/dev/null || echo 0)
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$nq" ]; then
    echo "[skip] next_$task/$arm"; return 0; fi
  [ -s "$out" ] && echo "[redone] next_$task/$arm partial $(wc -l < "$out")/$nq"
  mkdir -p $EV/anchorzip_p3/next_$task
  local flag=""; [ "${AZ[$arm]}" = true ] && flag="--anchorzip"
  echo "[gen] next_$task/$arm $(date +%T)"
  $PY $WRAP/llava_eval_arm_$script.py --model-path $MP \
    --question-file $qf --image-folder $imgf --answers-file $out \
    --temperature 0 --conv-mode vicuna_v1 \
    --visual_token_num ${VTN[$arm]} --beta $beta --alpha 0.5 $flag $extra \
    >> $EV/anchorzip_p3/next_$task/$arm.log 2>&1 \
    && echo "[ok] next_$task/$arm rc=0" || echo "[FAIL] next_$task/$arm"; }

for arm in "${ARM_ORDER[@]}"; do
  run textvqa $arm model_vqa \
    $EV/textvqa/llava_textvqa_val_v051_ocr.jsonl $EV/textvqa/train_images
done
for arm in "${ARM_ORDER[@]}"; do
  run pope $arm loader $EV/pope/llava_pope_test.jsonl $EV/pope/val2014
done
for arm in "${ARM_ORDER[@]}"; do
  run sqa $arm science \
    $EV/scienceqa/llava_test_QCM-LEPA.json $EV/scienceqa/test 2.0 \
    "--single-pred-prompt --conv-mode llava_v1"
done
for arm in "${ARM_ORDER[@]}"; do
  run gqa $arm loader \
    $EV/gqa/llava_gqa_testdev_balanced.jsonl $EV/gqa/data/images
done
for arm in "${ARM_ORDER[@]}"; do
  run mmvet $arm model_vqa \
    $EV/mm-vet/llava-mm-vet.jsonl $EV/mmvet/mm-vet/images 1.0
done
echo "[next] all done $(date)"
