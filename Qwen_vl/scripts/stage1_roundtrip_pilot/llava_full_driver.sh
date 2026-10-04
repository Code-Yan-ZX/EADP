#!/usr/bin/env bash
# LLaVA-1.5 four-arm x task driver (P3, 2026-10-05). Resumable: skips a
# (task,arm) whose answers file already exists.  Official eval paths
# verbatim (model_vqa.py / model_vqa_loader.py semantics) through the
# arm wrappers; scoring runs per task where local scripts exist.
set -u
# single-instance guard (atomic mkdir lock)
if ! mkdir /tmp/llava_full_driver.lock 2>/dev/null; then
  echo "[driver] another instance holds the lock, exit"; exit 1
fi
trap "rmdir /tmp/llava_full_driver.lock" EXIT
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
cd $LL
export PYTHONPATH=$LL

declare -A AK VTN AZ   # arm -> vtn, anchorzip
ARM_ORDER=(FULL EGATHER LRMAIN025 LRMAIN0125)
VTN[FULL]=0;      AZ[FULL]=false
VTN[EGATHER]=128; AZ[EGATHER]=false
VTN[LRMAIN025]=128; AZ[LRMAIN025]=true
VTN[LRMAIN0125]=64;  AZ[LRMAIN0125]=true

run () {  # run <task> <arm> <script: model_vqa|loader> <question> <images> <beta>
  local task=$1 arm=$2 script=$3 qf=$4 imgf=$5 beta=${6:-2.0}
  local out=$EV/anchorzip_p3/$task/$arm.jsonl
  local nq=$(wc -l < "$qf")
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$nq" ]; then
    echo "[skip] $task/$arm (complete)"; return 0
  fi
  [ -s "$out" ] && echo "[redone] $task/$arm partial $(wc -l < "$out")/$nq"
  mkdir -p $EV/anchorzip_p3/$task
  local flag=""; [ "${AZ[$arm]}" = true ] && flag="--anchorzip"
  echo "[gen] $task/$arm $(date +%T)"
  $PY $WRAP/llava_eval_arm_${script}.py \
    --model-path $MP --question-file $qf --image-folder $imgf \
    --answers-file $out --temperature 0 --conv-mode vicuna_v1 \
    --visual_token_num ${VTN[$arm]} --beta $beta --alpha 0.5 $flag \
    >> $EV/anchorzip_p3/$task/$arm.log 2>&1 \
    && echo "[ok] $task/$arm rc=0" || echo "[FAIL] $task/$arm"
}

for arm in "${ARM_ORDER[@]}"; do
  # TextVQA (model_vqa semantics)
  run textvqa $arm model_vqa \
    $EV/textvqa/llava_textvqa_val_v051_ocr.jsonl $EV/textvqa/train_images
done
for arm in "${ARM_ORDER[@]}"; do
  run pope $arm loader $EV/pope/llava_pope_test.jsonl $EV/pope/val2014
done
for arm in "${ARM_ORDER[@]}"; do
  run mme $arm loader $EV/MME/llava_mme.jsonl $EV/MME/MME_Benchmark
done
for arm in "${ARM_ORDER[@]}"; do
  run vqav2 $arm loader \
    $EV/vqav2/llava_vqav2_mscoco_test2015.jsonl $EV/vqav2/test2015
done
for arm in "${ARM_ORDER[@]}"; do
  run mmvet $arm model_vqa \
    $EV/mm-vet/llava-mm-vet.jsonl $EV/mmvet/images 1.0
done
echo "[driver] round1 done $(date)"
