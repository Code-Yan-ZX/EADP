#!/usr/bin/env bash
# SQA AnchorZip arms (K128/K64/K32) with the local RECONSTRUCTED CQM-I protocol:
# llava_test_CQM-I + vicuna_v1, greedy (temperature 0), single-pred-prompt.
# Frozen method hyper-params unchanged from the budget-point rounds:
# beta=2.0, alpha=0.5, anchorzip rtg (lambda=0.25 baked in the port).
# Writes NEW files (*_CQMI_vicuna.jsonl); old QCM-LEPA/llava_v1 files untouched.
# The author CQM-I input file is unavailable; these rows are local-protocol
# comparisons and do not by themselves establish paper-input equivalence.
set -u
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP15=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
V=$EV/anchorzip_p3/sqa
QF=$EV/scienceqa/llava_test_CQM-I.json
IMGF=$EV/scienceqa/test
cd $LL; export PYTHONPATH=$LL

vram_guard() {
  for i in 1 2; do
    while true; do
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      free=$((total-used))
      [ "$free" -ge 18000 ] && break
      echo "[sqa-arms] waiting for VRAM: ${free}MB free"
      sleep 30
    done
    sleep 60
  done
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  free=$((total-used))
  [ "$free" -ge 18000 ] || { echo "[sqa-arms] VRAM guard FAILED: ${free}MB"; return 1; }
}

run_arm () { local arm=$1 vtn=$2
  local out=$V/${arm}_CQMI_vicuna.jsonl
  local res=$V/${arm}_CQMI_vicuna_result.json
  if [ ! -s "$out" ]; then
    vram_guard || return 1
    echo "[sqa-arms] gen $arm vtn=$vtn $(date +%T)"
    $PY $WRAP/llava_eval_arm_science.py --model-path $MP15 \
      --question-file $QF --image-folder $IMGF --answers-file $out \
      --temperature 0 --conv-mode vicuna_v1 \
      --visual_token_num $vtn --beta 2.0 --alpha 0.5 --anchorzip \
      --single-pred-prompt \
      >> $V/${arm}_CQMI_vicuna.log 2>&1
    local rc=$?
    local n; n=$(wc -l < "$out" 2>/dev/null || echo 0)
    if [ "$rc" -ne 0 ] || [ "$n" -ne 4241 ]; then
      echo "[sqa-arms] GEN FAIL $arm rc=$rc n=$n"; return 1
    fi
  fi
  # Reuse/scoring requires the complete unique CQMI question set.
  $PY $WRAP/check_sqa_reuse.py --question-file "$QF" --result-file "$out" || return 1
  if [ ! -s "$res" ]; then
    $PY -m llava.eval.eval_science_qa --base-dir $EV/scienceqa \
      --result-file $out \
      --output-file $V/${arm}_CQMI_vicuna_output.json \
      --output-result $res \
      > $V/${arm}_CQMI_vicuna.score.txt 2>&1 \
      || { echo "[sqa-arms] SCORE FAIL $arm"; return 1; }
  fi
  echo "[sqa-arms] $arm done: $(grep 'Accuracy' $V/${arm}_CQMI_vicuna.score.txt | tail -1)"
}

run_arm LRMAIN025 128   || exit 1
run_arm LRMAIN0125 64   || exit 1
run_arm LRMAIN00625 32  || exit 1

echo "[sqa-arms] all done"
touch /tmp/sqa_arms_cqmi_done.marker
