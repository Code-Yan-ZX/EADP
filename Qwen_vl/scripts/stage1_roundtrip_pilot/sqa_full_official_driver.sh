#!/usr/bin/env bash
# SQA FULL with the local RECONSTRUCTED CQM-I protocol: llava_test_CQM-I question
# file + vicuna_v1 conv, greedy (--temperature 0), single-pred-prompt.
# Audit item #1 fix step B (2026-10-08).  Writes NEW files only; the old
# FULL.jsonl (QCM-LEPA / llava_v1) is left untouched.
# The author CQM-I input file is unavailable; these rows are local-protocol
# comparisons and do not by themselves establish paper-input equivalence.
set -u
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP15=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
V=$EV/anchorzip_p3/sqa
cd $LL; export PYTHONPATH=$LL

OUT=$V/FULL_CQMI_vicuna.jsonl
RES=$V/FULL_CQMI_vicuna_result.json

vram_guard() {
  # need >= 20GB free, stable across 60s (lesson: ollama squatting caused OOMs)
  for i in 1 2; do
    while true; do
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      free=$((total-used))
      [ "$free" -ge 20000 ] && break
      echo "[sqa-cqmi] waiting for VRAM: ${free}MB free"
      sleep 30
    done
    sleep 60
  done
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  free=$((total-used))
  [ "$free" -ge 20000 ] || { echo "[sqa-cqmi] VRAM guard FAILED: ${free}MB"; return 1; }
}

vram_guard || exit 1

if [ ! -s "$OUT" ]; then
  $PY -m llava.eval.model_vqa_science \
    --model-path $MP15 \
    --question-file $EV/scienceqa/llava_test_CQM-I.json \
    --image-folder $EV/scienceqa/test \
    --answers-file $OUT \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    >> $V/FULL_CQMI_vicuna.log 2>&1
  rc=$?
  n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  if [ "$rc" -ne 0 ] || [ "$n" -ne 4241 ]; then
    echo "[sqa-cqmi] GEN FAIL rc=$rc n=$n"
    exit 1
  fi
  echo "[sqa-cqmi] gen ok: 4241 lines"
else
  echo "[sqa-cqmi] gen already done ($(wc -l < "$OUT") lines)"
fi

$PY $WRAP/check_sqa_reuse.py --question-file $EV/scienceqa/llava_test_CQM-I.json --result-file "$OUT" || exit 1

if [ ! -s "$RES" ]; then
  $PY -m llava.eval.eval_science_qa \
    --base-dir $EV/scienceqa \
    --result-file $OUT \
    --output-file $V/FULL_CQMI_vicuna_output.json \
    --output-result $RES \
    > $V/FULL_CQMI_vicuna.score.txt 2>&1 \
    || { echo "[sqa-cqmi] SCORE FAIL"; exit 1; }
  echo "[sqa-cqmi] score ok"
else
  echo "[sqa-cqmi] score already done"
fi

echo "[sqa-cqmi] all done"
touch /tmp/sqa_cqmi_full_done.marker
