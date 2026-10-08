#!/usr/bin/env bash
# Round 2026-10-08 evening (user-approved):
#  Phase 1: NeXT (llava-v1.6) SQA three arms under the local RECONSTRUCTED CQM-I protocol
#           (llava_test_CQM-I + vicuna_v1 + greedy).  Frozen hyper-params
#           unchanged (beta 2.0 / alpha 0.5 / lambda 0.25; vtn = per-crop).
#  Phase 2: re-measure the LLaVA efficiency panels with the FIXED
#           llava_perf_panel.py (single model load; warmup no longer
#           invalidated by a reload).  16 panels = {v15,next} x
#           {FULL, AZ128/64/32, EOFF128/64/32}; output to perf2/ (old
#           perf/ preserved).  E* (official_replay+lam0) rows stay
#           archived as invalid_as_baseline_timing and are NOT re-run.
#           Qwen K128/K64 points need no re-run (P2 harness loads once).
# The author CQM-I input file is unavailable; these rows are local-protocol
# comparisons and do not by themselves establish paper-input equivalence.
set -u
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
NEXT=/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
V15=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
QF=$EV/scienceqa/llava_test_CQM-I.json
IMGF=$EV/scienceqa/test
VS=$EV/anchorzip_p3/next_sqa
P2=$EV/anchorzip_p3/perf2
Q=$EV/textvqa/llava_textvqa_val_v051_ocr.jsonl
IMG=$EV/textvqa/train_images
cd $LL; export PYTHONPATH=$LL
mkdir -p $P2

vram_guard() {  # >= needed MB free, stable across 60s
  local need=$1
  for i in 1 2; do
    while true; do
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ $((total-used)) -ge "$need" ] && break
      echo "[drv] waiting for VRAM: $((total-used))MB free"; sleep 30
    done
    sleep 60
  done
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ $((total-used)) -ge "$need" ] || { echo "[drv] VRAM guard FAILED"; return 1; }
}

run_next_sqa () { local arm=$1 vtn=$2
  local out=$VS/${arm}_CQMI_vicuna.jsonl
  local res=$VS/${arm}_CQMI_vicuna_result.json
  if [ ! -s "$out" ]; then
    vram_guard 18000 || return 1
    echo "[next-sqa] gen $arm vtn=$vtn $(date +%T)"
    $PY $WRAP/llava_eval_arm_science.py --model-path $NEXT \
      --question-file $QF --image-folder $IMGF --answers-file $out \
      --temperature 0 --conv-mode vicuna_v1 \
      --visual_token_num $vtn --beta 2.0 --alpha 0.5 --anchorzip \
      --single-pred-prompt \
      >> $VS/${arm}_CQMI_vicuna.log 2>&1
    local rc=$? n; n=$(wc -l < "$out" 2>/dev/null || echo 0)
    [ "$rc" -eq 0 ] && [ "$n" -eq 4241 ] || { echo "[next-sqa] GEN FAIL $arm rc=$rc n=$n"; return 1; }
  fi
  # Reuse/scoring requires the complete unique CQMI question set.
  $PY $WRAP/check_sqa_reuse.py --question-file "$QF" --result-file "$out" || return 1
  if [ ! -s "$res" ]; then
    $PY -m llava.eval.eval_science_qa --base-dir $EV/scienceqa \
      --result-file $out --output-file $VS/${arm}_CQMI_vicuna_output.json \
      --output-result $res > $VS/${arm}_CQMI_vicuna.score.txt 2>&1 \
      || { echo "[next-sqa] SCORE FAIL $arm"; return 1; }
  fi
  echo "[next-sqa] $arm done: $(grep Accuracy $VS/${arm}_CQMI_vicuna.score.txt | tail -1)"
}

run_panel () { local tag=$1 mp=$2 vtn=$3 az=$4
  echo "[perf2] $tag $(date +%T)"
  local extra=""; [ "$az" = true ] && extra="--anchorzip"
  $PY $WRAP/llava_perf_panel.py --model-path $mp --image-folder $IMG \
    --question-file $Q --conv-mode vicuna_v1 --visual-token-num $vtn \
    $extra --beta 2.0 --alpha 0.5 --tag $tag --json-out $P2/$tag.json \
    && echo "[ok] perf2 $tag" || echo "[FAIL] perf2 $tag"
}

run_next_sqa LRMAIN025 128  || exit 1
run_next_sqa LRMAIN0125 64  || exit 1
run_next_sqa LRMAIN00625 32 || exit 1

vram_guard 18000 || exit 1
run_panel v15_FULL  $V15 0 false
run_panel v15_AZ128 $V15 128 true
run_panel v15_AZ64  $V15 64 true
run_panel v15_AZ32  $V15 32 true
run_panel v15_EOFF128 $V15 128 false
run_panel v15_EOFF64  $V15 64 false
run_panel v15_EOFF32  $V15 32 false
run_panel next_FULL  $NEXT 0 false
run_panel next_AZ128 $NEXT 128 true
run_panel next_AZ64  $NEXT 64 true
run_panel next_AZ32  $NEXT 32 true
run_panel next_EOFF128 $NEXT 128 false
run_panel next_EOFF64  $NEXT 64 false
run_panel next_EOFF32  $NEXT 32 false

echo "[drv] all done $(date)"
touch /tmp/next_sqa_perf2_done.marker
