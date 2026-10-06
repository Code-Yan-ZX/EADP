#!/usr/bin/env bash
# Budget-point lane B: ALL NeXT jobs (next_VizWiz val 128/64, next_MMB-CN
# 128/64/32, K160=32/crop full task set) + NeXT-side scoring + perf panels.
# NeXT peaks ~23.5GB -- does NOT fit alongside the Qwen K=128 driver
# (~22GB), so this lane waits for /tmp/qwen3_k128_done.marker, then runs
# concurrently with lane A (v1.5 ~16GB; combined ~40GB on 46GB).
set -u
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MPNX=/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
cd $LL; export PYTHONPATH=$LL
V=$EV/anchorzip_p3
EN=$EV/mmbench/MMBench_DEV_EN_V11.tsv
CN=$EV/mmbench/MMBench_DEV_CN_V11.tsv

while [ ! -f /tmp/qwen3_k128_done.marker ]; do
  echo "[laneB] waiting for qwen3 K128 $(date +%T)"; sleep 300
done

need_vram_mb=26000   # NeXT peak ~23.5GB + margin
gpu_wait () {
  while true; do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local free=$(( (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) - used ))
    if [ "$free" -ge "$need_vram_mb" ]; then return 0; fi
    echo "[laneB] GPU short (${free}MB free) $(date +%T)"; sleep 300
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
  $PY $WRAP/llava_eval_arm_$script.py --model-path $MPNX \
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
  $PY -m llava.eval.model_vqa_mmbench --model-path $MPNX \
    --question-file $tsv --answers-file $out \
    --visual_token_num $vtn --anchorzip --beta 2.0 --alpha 0.5 \
    --single-pred-prompt --temperature 0 --conv-mode vicuna_v1 \
    >> $V/$task/$arm.log 2>&1 \
    && echo "[ok] $task/$arm rc=0" || echo "[FAIL] $task/$arm"; }

# ---- 1) NeXT VizWiz val, existing arms ----
gen2 next_vizwiz LRMAIN025 128 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
gen2 next_vizwiz LRMAIN0125 64 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
# ---- 2) NeXT MMBench-CN V11 ----
gen_mmb next_mmbcn LRMAIN025 128 $CN
gen_mmb next_mmbcn LRMAIN0125 64 $CN
gen_mmb next_mmbcn LRMAIN00625 32 $CN
# ---- 3) NeXT K=160 (LRMAIN00625, 32/crop) full task set ----
gen2 next_textvqa LRMAIN00625 32 $EV/textvqa/llava_textvqa_val_v051_ocr.jsonl $EV/textvqa/train_images model_vqa
gen2 next_pope LRMAIN00625 32 $EV/pope/llava_pope_test.jsonl $EV/pope/val2014 loader
gen2 next_mme LRMAIN00625 32 $EV/MME/llava_mme.jsonl $EV/MME/MME_Benchmark loader
gen2 next_sqa LRMAIN00625 32 $EV/scienceqa/llava_test_QCM-LEPA.json $EV/scienceqa/test science 2.0 "--single-pred-prompt --conv-mode llava_v1"
gen2 next_gqa LRMAIN00625 32 $EV/gqa/llava_gqa_testdev_balanced.jsonl $EV/gqa/data/images loader
gen2 next_mmvet LRMAIN00625 32 $EV/mm-vet/llava-mm-vet.jsonl $EV/mmvet/mm-vet/images model_vqa 1.0
gen2 next_vizwiz LRMAIN00625 32 $EV/vizwiz/llava_val.jsonl $EV/vizwiz/val model_vqa
gen_mmb next_mmben LRMAIN00625 32 $EN

# ---- 4) NeXT-side scoring ----
for arm in LRMAIN025 LRMAIN0125 LRMAIN00625; do
  out=$V/next_vizwiz/$arm.jsonl
  [ -s "$out" ] && [ ! -s "$V/next_vizwiz/${arm}.score.json" ] && \
    $PY $WRAP/vizwiz_val_score.py --gt-file $EV/vizwiz/val.json \
      --result-file $out --out $V/next_vizwiz/${arm}.score.json
done
for task in next_mmbcn next_mmben; do
  local_tsv=$EN; [ "$task" = "next_mmbcn" ] && local_tsv=$CN
  for arm in LRMAIN025 LRMAIN0125 LRMAIN00625; do
    out=$V/$task/$arm.jsonl
    [ -s "$out" ] && [ ! -s "$V/$task/${arm}.score.json" ] && \
      $PY $WRAP/mmben_circular_score.py --question-file $local_tsv \
        --result-file $out --out $V/$task/${arm}.score.json
  done
done
arm=LRMAIN00625
out=$V/next_textvqa/$arm.jsonl
[ -s "$out" ] && [ ! -s "$V/next_textvqa/${arm}.score.txt" ] && \
  $PY -m llava.eval.eval_textvqa --annotation-file $EV/textvqa/TextVQA_0.5.1_val.json \
    --result-file $out > $V/next_textvqa/${arm}.score.txt 2>&1
out=$V/next_pope/$arm.jsonl
[ -s "$out" ] && [ ! -s "$V/next_pope/${arm}.score.txt" ] && \
  $PY -m llava.eval.eval_pope --annotation-dir $EV/pope \
    --question-file $EV/pope/llava_pope_test.jsonl \
    --result-file $out > $V/next_pope/${arm}.score.txt 2>&1
out=$V/next_sqa/$arm.jsonl
[ -s "$out" ] && [ ! -s "$V/next_sqa/${arm}_result.json" ] && \
  $PY -m llava.eval.eval_science_qa --base-dir $EV/scienceqa \
    --result-file $out --output-file $V/next_sqa/${arm}_output.json \
    --output-result $V/next_sqa/${arm}_result.json > $V/next_sqa/${arm}.score.txt 2>&1
out=$V/next_gqa/$arm.jsonl
[ -s "$out" ] && [ ! -s "$V/next_gqa/${arm}.score.json" ] && \
  $PY $WRAP/gqa_score.py --gt-file /tmp/gqa12/testdev_balanced_questions.json \
    --result-file $out --out $V/next_gqa/${arm}.score.json
out=$V/next_mme/$arm.jsonl
if [ -s "$out" ] && [ ! -s "$V/next_mme/${arm}.score.txt" ]; then
  (cd $EV/MME && cp $out answers/budget_next_${arm}.jsonl && \
   $PY convert_answer_to_mme.py --experiment budget_next_${arm} >/dev/null 2>&1 && \
   $PY calc_scores.py --results_dir eval_tool/answers/budget_next_${arm} \
     > $V/next_mme/${arm}.score.txt 2>&1)
fi

# ---- 5) perf panels for the new budgets ----
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

touch /tmp/anchorzip_laneB_done.marker
echo "[laneB] all done $(date)"
