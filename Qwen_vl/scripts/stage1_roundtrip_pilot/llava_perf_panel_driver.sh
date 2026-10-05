#!/usr/bin/env bash
set -u
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
cd $LL; export PYTHONPATH=$LL
Q=$EV/textvqa/llava_textvqa_val_v051_ocr.jsonl
IMG=$EV/textvqa/train_images
OUT=$EV/anchorzip_p3/perf
mkdir -p $OUT
gpu_wait () { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 5000 ] && return 0; echo "[wait] GPU ${u}MiB $(date +%T)"; sleep 300; done; }
run () { local m=$1 tag=$2 vtn=$3 az=$4 mp=$5
  gpu_wait
  local extra=""; [ "$az" = true ] && extra="--anchorzip"
  echo "[perf] $tag $(date +%T)"
  $PY $WRAP/llava_perf_panel.py --model-path $mp --image-folder $IMG \
    --question-file $Q --conv-mode vicuna_v1 --visual-token-num $vtn \
    $extra --tag $tag --json-out $OUT/$tag.json \
    && echo "[ok] $tag" || echo "[FAIL] $tag"; }
V15=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
NEXT=/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
run x v15_FULL 0 false $V15
run x v15_AZ128 128 true $V15
run x v15_AZ64 64 true $V15
run x next_FULL 0 false $NEXT
run x next_AZ128 128 true $NEXT
run x next_AZ64 64 true $NEXT
echo "[perf] all done $(date)"
