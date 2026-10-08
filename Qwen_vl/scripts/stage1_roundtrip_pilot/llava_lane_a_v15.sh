#!/usr/bin/env bash
# Budget-point lane A: ALL v1.5 jobs (VizWiz val 128/64, MMB-CN 128/64/32,
# K32 full task set) + v1.5-side scoring.  Runs CONCURRENTLY with the
# Qwen3 K=128 driver (~22GB) -- v1.5 peaks ~16GB, combined ~38GB on the
# 46GB card.  Per-job VRAM guard instead of the mkdir lock (which now only
# gates against the still-running mmben driver).
set -u
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP15=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
cd $LL; export PYTHONPATH=$LL
V=$EV/anchorzip_p3
EN=$EV/mmbench/MMBench_DEV_EN_V11.tsv
CN=$EV/mmbench/MMBench_DEV_CN_V11.tsv

# wait for the mmben driver to release the lock (without grabbing it)
while [ -d /tmp/llava_full_driver.lock ]; do
  echo "[laneA] waiting for mmben $(date +%T)"; sleep 300
done

need_vram_mb=18000   # v1.5 peak ~15.7GB + margin
gpu_wait () {
  while true; do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    local free=$((total - used))
    if [ "$free" -ge "$need_vram_mb" ]; then return 0; fi
    echo "[laneA] GPU short (${free}MB free) $(date +%T)"; sleep 300
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

gen2 () { local task=$1 arm=$2 vtn=$3 qf=$4 imgf=$5 script=$6 beta=${7:-2.0} extra=${8:-}
  local out=$V/${task}/$arm.jsonl
  local nq=$(nq_of "$qf")
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$nq" ]; then
    echo "[skip] $task/$arm"; return 0; fi
  [ -s "$out" ] && echo "[redone] $task/$arm partial $(wc -l < "$out")/$nq"
  mkdir -p $V/$task
  gpu_wait
  echo "[gen] $task/$arm vtn=$vtn $(date +%T)"
  $PY $WRAP/llava_eval_arm_$script.py --model-path $MP15 \
    --question-file $qf --image-folder $imgf --answers-file $out \
    --temperature 0 --conv-mode vicuna_v1 \
    --visual_token_num $vtn --beta $beta --alpha 0.5 --anchorzip $extra \
    >> $V/$task/$arm.log 2>&1 \
    && echo "[ok] $task/$arm rc=0" || echo "[FAIL] $task/$arm"; }

gen_mmb () { local task=$1 arm=$2 vtn=$3 tsv=$4
  local out=$V/${task}/$arm.jsonl
  local nq=$(nq_of "$tsv")
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$nq" ]; then
    echo "[skip] $task/$arm"; return 0; fi
  [ -s "$out" ] && echo "[redone] $task/$arm partial $(wc -l < "$out")/$nq"
  mkdir -p $V/$task
  gpu_wait
  echo "[gen] $task/$arm vtn=$vtn $(date +%T)"
  $PY -m llava.eval.model_vqa_mmbench --model-path $MP15 \
    --question-file $tsv --answers-file $out \
    --visual_token_num $vtn --anchorzip --beta 2.0 --alpha 0.5 \
    --single-pred-prompt --temperature 0 --conv-mode vicuna_v1 \
    >> $V/$task/$arm.log 2>&1 \
    && echo "[ok] $task/$arm rc=0" || echo "[FAIL] $task/$arm"; }

# ---- 1) VizWiz val, existing v1.5 arms ----
gen2 vizwiz LRMAIN025 128 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
gen2 vizwiz LRMAIN0125 64 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
# ---- 2) MMBench-CN V11, v1.5 arms ----
gen_mmb mmbcn LRMAIN025 128 $CN
gen_mmb mmbcn LRMAIN0125 64 $CN
gen_mmb mmbcn LRMAIN00625 32 $CN
# ---- 3) v1.5 K=32 (LRMAIN00625) full task set ----
gen2 textvqa LRMAIN00625 32 $EV/textvqa/llava_textvqa_val_v051_ocr.jsonl $EV/textvqa/train_images model_vqa
gen2 pope LRMAIN00625 32 $EV/pope/llava_pope_test.jsonl $EV/pope/val2014 loader
gen2 mme LRMAIN00625 32 $EV/MME/llava_mme.jsonl $EV/MME/MME_Benchmark loader
echo "[retired SQA protocol] Use sqa_arms_official_driver.sh; QCM-LEPA outputs are archived."
gen2 gqa LRMAIN00625 32 $EV/gqa/llava_gqa_testdev_balanced.jsonl $EV/gqa/data/images loader
gen2 mmvet LRMAIN00625 32 $EV/mm-vet/llava-mm-vet.jsonl $EV/mmvet/mm-vet/images model_vqa 1.0
gen2 vizwiz LRMAIN00625 32 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
gen_mmb mmben LRMAIN00625 32 $EN

# ---- 4) v1.5-side scoring ----
for task in vizwiz; do
  for arm in LRMAIN025 LRMAIN0125 LRMAIN00625; do
    out=$V/$task/$arm.jsonl
    if [ -s "$out" ]; then
      $PY $WRAP/vizwiz_val_score.py --gt-file $EV/vizwiz/val.json \
        --result-file $out --out $V/$task/${arm}.official.score.json || exit 1
    fi
  done
done
for task in mmbcn mmben; do
  local_tsv=$EN; [ "$task" = "mmbcn" ] && local_tsv=$CN
  for arm in LRMAIN025 LRMAIN0125 LRMAIN00625; do
    out=$V/$task/$arm.jsonl
    [ -s "$out" ] && [ ! -s "$V/$task/${arm}.score.json" ] && \
      $PY $WRAP/mmben_circular_score.py --question-file $local_tsv \
        --result-file $out --out $V/$task/${arm}.score.json
  done
done
arm=LRMAIN00625
out=$V/textvqa/$arm.jsonl
[ -s "$out" ] && [ ! -s "$V/textvqa/${arm}.score.txt" ] && \
  $PY -m llava.eval.eval_textvqa --annotation-file $EV/textvqa/TextVQA_0.5.1_val.json \
    --result-file $out > $V/textvqa/${arm}.score.txt 2>&1
# Repaired protocols; refresh all requested budgets without reusing old scores.
for score_arm in LRMAIN025 LRMAIN0125 LRMAIN00625; do
  out=$V/pope/$score_arm.jsonl
  if [ -s "$out" ]; then
    $PY $WRAP/official_score.py --dataset pope --annotation-dir $EV/pope \
      --question-file $EV/pope/llava_pope_test.jsonl --result-file "$out" \
      --out $V/pope/${score_arm}.official.score.json || exit 1
  fi
  out=$V/gqa/$score_arm.jsonl
  if [ -s "$out" ]; then
    $PY $WRAP/gqa_score.py --gt-file /tmp/gqa12/testdev_balanced_questions.json \
      --result-file "$out" --out $V/gqa/${score_arm}.official.score.json || exit 1
  fi
done
# SQA is generated/scored only by the dedicated CQMI drivers above.
# Never reuse historical .score.txt: reconstructed release GT had 162 wrong labels.
for score_arm in LRMAIN025 LRMAIN0125 LRMAIN00625; do
  out=$V/mme/$score_arm.jsonl
  if [ -s "$out" ]; then
    $PY $WRAP/mme_canonical_score.py --result-file "$out" \
      --question-file $EV/MME/llava_mme.jsonl --gt-archive $EV/MME/eval_tool.zip \
      --out $V/mme/${score_arm}.canonical_gt.score.json || exit 1
  fi
done

touch /tmp/anchorzip_laneA_done.marker
echo "[laneA] all done $(date)"
