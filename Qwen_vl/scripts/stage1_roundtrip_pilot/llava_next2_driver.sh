#!/usr/bin/env bash
# NeXT two-arm driver (scope 2026-10-05): AnchorZip arms only.
# LRMAIN025 = 128/crop (nominal total K640), LRMAIN0125 = 64/crop (nominal K320).
# FULL / E_GATHER are NOT regenerated (per user scope decision); existing
# next_textvqa 4-arm outputs are kept as-is and only the two AnchorZip arms
# are used for the remaining tasks. Task order: pope -> mme -> sqa -> gqa -> mmvet.
# MME generation reuses $EV/MME/llava_mme.jsonl (2374 q) like the v1.5 round;
# Scoring uses mme_canonical_score.py with verified eval_tool.zip GT; the
# reconstructed MME_Benchmark_release_version labels are invalid for scoring.
set -u

# Wait for the orphaned pope/LRMAIN0125 eval (launched by the retired
# 4-arm driver) to finish before touching the GPU.
while pgrep -f "llava_eval_arm_.*\.py --model-path /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b" >/dev/null; do
  echo "[wait] orphan next eval still running $(date +%T)"; sleep 120
done

# VRAM guard (46GB card): require < 5GB used for 60s before starting.
gpu_wait () {
  while true; do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 5000 ]; then
      sleep 60
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$used" -lt 5000 ] && return 0
    fi
    echo "[wait] GPU busy (${used}MiB in use), retry $(date +%T)"; sleep 300
  done
}
gpu_wait

if ! mkdir /tmp/llava_full_driver.lock 2>/dev/null; then
  echo "[next2] lock held, exit"; exit 1
fi
trap "rmdir /tmp/llava_full_driver.lock" EXIT

PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP=/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
cd $LL; export PYTHONPATH=$LL
declare -A VTN AZ
ARM_ORDER=(LRMAIN025 LRMAIN0125)
VTN[LRMAIN025]=128; AZ[LRMAIN025]=true
VTN[LRMAIN0125]=64; AZ[LRMAIN0125]=true

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
  run pope $arm loader $EV/pope/llava_pope_test.jsonl $EV/pope/val2014
done
for arm in "${ARM_ORDER[@]}"; do
  run mme $arm loader $EV/MME/llava_mme.jsonl $EV/MME/MME_Benchmark
done
echo "[retired SQA protocol] Use next_sqa_official_and_perf2_driver.sh; QCM-LEPA outputs are archived."
for arm in "${ARM_ORDER[@]}"; do
  run gqa $arm loader \
    $EV/gqa/llava_gqa_testdev_balanced.jsonl $EV/gqa/data/images
done
for arm in "${ARM_ORDER[@]}"; do
  run mmvet $arm model_vqa \
    $EV/mm-vet/llava-mm-vet.jsonl $EV/mmvet/mm-vet/images 1.0
done
echo "[next2] all done $(date)"
