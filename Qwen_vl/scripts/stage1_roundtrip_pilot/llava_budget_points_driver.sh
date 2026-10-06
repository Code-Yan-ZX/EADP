#!/usr/bin/env bash
# Budget-point completion driver (2026-10-06, user approved):
#  * new arms LRMAIN00625: v1.5 K=32 (single image), NeXT 32/crop (nominal K160)
#    -- arm suffix = 0.25 * (K/128): 025=K128, 0125=K64, 00625=K32; lam=0.25 frozen
#  * VizWiz VAL reproduction for all AnchorZip arms (official test server
#    retired; community convention, user-approved) + val scorer
#  * MMBench-CN V11 (tsv at /media/disk2/YZX/LMUData, same 4876-row circular
#    layout as EN) for all arms both models
#  * scoring for every new prediction + perf panels for the new budgets
# Lock-serialised behind the running mmben driver.
set -u
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP15=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
MPNX=/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
cd $LL; export PYTHONPATH=$LL

while ! mkdir /tmp/llava_full_driver.lock 2>/dev/null; do
  echo "[budget] waiting for lock $(date +%T)"; sleep 300
done
trap "rmdir /tmp/llava_full_driver.lock" EXIT

gpu_wait () {
  while true; do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 5000 ]; then
      sleep 60
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$used" -lt 5000 ] && return 0
    fi
    echo "[budget] GPU busy (${used}MiB) $(date +%T)"; sleep 300
  done
}

nq_of () {
  local qf=$1
  if [[ "$qf" == *.tsv ]]; then
    $PY -c "import pandas as pd; print(len(pd.read_table('$qf')))"
  elif [[ "$qf" == *.json ]]; then
    $PY -c "import json; d=json.load(open('$qf')); print(len(d['questions']) if isinstance(d,dict) and 'questions' in d else len(d))"
  else
    wc -l < "$qf" 2>/dev/null || echo 0
  fi
}

gen2 () { local model=$1 task=$2 arm=$3 vtn=$4 qf=$5 imgf=$6 script=$7 beta=${8:-2.0} extra=${9:-}
  local out=$EV/anchorzip_p3/${task}/$arm.jsonl
  local nq=$(nq_of "$qf")
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$nq" ]; then
    echo "[skip] $task/$arm"; return 0; fi
  [ -s "$out" ] && echo "[redone] $task/$arm partial $(wc -l < "$out")/$nq"
  mkdir -p $EV/anchorzip_p3/$task
  gpu_wait
  echo "[gen] $task/$arm vtn=$vtn $(date +%T)"
  $PY $WRAP/llava_eval_arm_$script.py --model-path $model \
    --question-file $qf --image-folder $imgf --answers-file $out \
    --temperature 0 --conv-mode vicuna_v1 \
    --visual_token_num $vtn --beta $beta --alpha 0.5 --anchorzip $extra \
    >> $EV/anchorzip_p3/$task/$arm.log 2>&1 \
    && echo "[ok] $task/$arm rc=0" || echo "[FAIL] $task/$arm"; }

# MMBench tsv tasks go through the official model_vqa_mmbench entry point
gen_mmb () { local model=$1 task=$2 arm=$3 vtn=$4 tsv=$5
  local out=$EV/anchorzip_p3/${task}/$arm.jsonl
  local nq=$(nq_of "$tsv")
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$nq" ]; then
    echo "[skip] $task/$arm"; return 0; fi
  [ -s "$out" ] && echo "[redone] $task/$arm partial $(wc -l < "$out")/$nq"
  mkdir -p $EV/anchorzip_p3/$task
  gpu_wait
  echo "[gen] $task/$arm vtn=$vtn $(date +%T)"
  $PY -m llava.eval.model_vqa_mmbench --model-path $model \
    --question-file $tsv --answers-file $out \
    --visual_token_num $vtn --anchorzip --beta 2.0 --alpha 0.5 \
    --single-pred-prompt --temperature 0 --conv-mode vicuna_v1 \
    >> $EV/anchorzip_p3/$task/$arm.log 2>&1 \
    && echo "[ok] $task/$arm rc=0" || echo "[FAIL] $task/$arm"; }

# ---------------- 1) VizWiz val, existing arms --------------------------
for spec in "LRMAIN025 128 $MP15" "LRMAIN0125 64 $MP15"; do
  set -- $spec; gen2 $3 vizwiz $1 $2 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
done
for spec in "LRMAIN025 128 $MPNX" "LRMAIN0125 64 $MPNX"; do
  set -- $spec; gen2 $3 next_vizwiz $1 $2 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
done

# ---------------- 2) MMBench-CN V11, all arms both models ---------------
CN=$EV/mmbench/MMBench_DEV_CN_V11.tsv
EN=$EV/mmbench/MMBench_DEV_EN_V11.tsv
gen_mmb $MP15 mmbcn LRMAIN025 128 $CN
gen_mmb $MP15 mmbcn LRMAIN0125 64 $CN
gen_mmb $MP15 mmbcn LRMAIN00625 32 $CN
gen_mmb $MPNX next_mmbcn LRMAIN025 128 $CN
gen_mmb $MPNX next_mmbcn LRMAIN0125 64 $CN
gen_mmb $MPNX next_mmbcn LRMAIN00625 32 $CN

# ---------------- 3) v1.5 K=32 (LRMAIN00625) full task set --------------
V=$EV/anchorzip_p3
gen2 $MP15 textvqa LRMAIN00625 32 $EV/textvqa/llava_textvqa_val_v051_ocr.jsonl $EV/textvqa/train_images model_vqa
gen2 $MP15 pope LRMAIN00625 32 $EV/pope/llava_pope_test.jsonl $EV/pope/val2014 loader
gen2 $MP15 mme LRMAIN00625 32 $EV/MME/llava_mme.jsonl $EV/MME/MME_Benchmark loader
gen2 $MP15 sqa LRMAIN00625 32 $EV/scienceqa/llava_test_QCM-LEPA.json $EV/scienceqa/test science 2.0 "--single-pred-prompt --conv-mode llava_v1"
gen2 $MP15 gqa LRMAIN00625 32 $EV/gqa/llava_gqa_testdev_balanced.jsonl $EV/gqa/data/images loader
gen2 $MP15 mmvet LRMAIN00625 32 $EV/mm-vet/llava-mm-vet.jsonl $EV/mmvet/mm-vet/images model_vqa 1.0
gen2 $MP15 vizwiz LRMAIN00625 32 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
gen_mmb $MP15 mmben LRMAIN00625 32 $EN

# ---------------- 4) NeXT K=160 (LRMAIN00625) full task set -------------
gen2 $MPNX next_textvqa LRMAIN00625 32 $EV/textvqa/llava_textvqa_val_v051_ocr.jsonl $EV/textvqa/train_images model_vqa
gen2 $MPNX next_pope LRMAIN00625 32 $EV/pope/llava_pope_test.jsonl $EV/pope/val2014 loader
gen2 $MPNX next_mme LRMAIN00625 32 $EV/MME/llava_mme.jsonl $EV/MME/MME_Benchmark loader
gen2 $MPNX next_sqa LRMAIN00625 32 $EV/scienceqa/llava_test_QCM-LEPA.json $EV/scienceqa/test science 2.0 "--single-pred-prompt --conv-mode llava_v1"
gen2 $MPNX next_gqa LRMAIN00625 32 $EV/gqa/llava_gqa_testdev_balanced.jsonl $EV/gqa/data/images loader
gen2 $MPNX next_mmvet LRMAIN00625 32 $EV/mm-vet/llava-mm-vet.jsonl $EV/mmvet/mm-vet/images model_vqa 1.0
gen2 $MPNX next_vizwiz LRMAIN00625 32 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
gen_mmb $MPNX next_mmben LRMAIN00625 32 $EN

# ---------------- 5) scoring --------------------------------------------
score_all () {
  for task in vizwiz next_vizwiz; do
    for arm in LRMAIN025 LRMAIN0125 LRMAIN00625; do
      local out=$V/$task/$arm.jsonl
      [ -s "$out" ] && [ ! -s "$V/$task/${arm}.score.json" ] && \
        $PY $WRAP/vizwiz_val_score.py --gt-file $EV/vizwiz/val.json \
          --result-file $out --out $V/$task/${arm}.score.json
    done
  done
  for task in mmben next_mmben mmbcn next_mmbcn; do
    local tsv=$EN
    [[ "$task" == *cn* ]] && tsv=$CN
    for arm in LRMAIN025 LRMAIN0125 LRMAIN00625; do
      local out=$V/$task/$arm.jsonl
      [ -s "$out" ] && [ ! -s "$V/$task/${arm}.score.json" ] && \
        $PY $WRAP/mmben_circular_score.py --question-file $tsv \
          --result-file $out --out $V/$task/${arm}.score.json
    done
  done
  for pre in "" next_; do
    local arm=LRMAIN00625
    local out=$V/${pre}textvqa/$arm.jsonl
    if [ -s "$out" ] && [ ! -s "$V/${pre}textvqa/${arm}.score.txt" ]; then
      (cd $LL && $PY -m llava.eval.eval_textvqa \
        --annotation-file $EV/textvqa/TextVQA_0.5.1_val.json \
        --result-file $out > $V/${pre}textvqa/${arm}.score.txt 2>&1); fi
    local outp=$V/${pre}pope/$arm.jsonl
    if [ -s "$outp" ] && [ ! -s "$V/${pre}pope/${arm}.score.txt" ]; then
      (cd $LL && $PY -m llava.eval.eval_pope --annotation-dir $EV/pope \
        --question-file $EV/pope/llava_pope_test.jsonl \
        --result-file $outp > $V/${pre}pope/${arm}.score.txt 2>&1); fi
    local outs=$V/${pre}sqa/$arm.jsonl
    if [ -s "$outs" ] && [ ! -s "$V/${pre}sqa/${arm}_result.json" ]; then
      (cd $LL && $PY -m llava.eval.eval_science_qa --base-dir $EV/scienceqa \
        --result-file $outs --output-file $V/${pre}sqa/${arm}_output.json \
        --output-result $V/${pre}sqa/${arm}_result.json \
        > $V/${pre}sqa/${arm}.score.txt 2>&1); fi
    local outg=$V/${pre}gqa/$arm.jsonl
    if [ -s "$outg" ] && [ ! -s "$V/${pre}gqa/${arm}.score.json" ]; then
      $PY $WRAP/gqa_score.py --gt-file /tmp/gqa12/testdev_balanced_questions.json \
        --result-file $outg --out $V/${pre}gqa/${arm}.score.json; fi
    local outm=$V/${pre}mme/$arm.jsonl
    if [ -s "$outm" ] && [ ! -s "$V/${pre}mme/${arm}.score.txt" ]; then
      (cd $EV/MME && cp $outm answers/budget_${pre}${arm}.jsonl && \
       $PY convert_answer_to_mme.py --experiment budget_${pre}${arm} >/dev/null 2>&1 && \
       $PY calc_scores.py --results_dir eval_tool/answers/budget_${pre}${arm} \
         > $V/${pre}mme/${arm}.score.txt 2>&1); fi
  done
}
score_all

# ---------------- 6) perf panels for the new budgets --------------------
Q=$EV/textvqa/llava_textvqa_val_v051_ocr.jsonl
IMG=$EV/textvqa/train_images
OUT=$EV/anchorzip_p3/perf
mkdir -p $OUT
for spec in "v15_AZ32 32 $MP15" "next_AZ32 32 $MPNX"; do
  set -- $spec
  [ -s "$OUT/$1.json" ] && continue
  gpu_wait
  $PY $WRAP/llava_perf_panel.py --model-path $3 --image-folder $IMG \
    --question-file $Q --conv-mode vicuna_v1 --visual-token-num $2 \
    --anchorzip --tag $1 --json-out $OUT/$1.json \
    && echo "[ok] perf $1" || echo "[FAIL] perf $1"
done

echo "[budget] all done $(date)"
