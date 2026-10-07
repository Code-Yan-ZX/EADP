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
# Gate on FULL after_k128 completion (E-side panels + Qwen K64) so the
# panel timings stay exclusive -- running alongside any GPU job would
# contaminate TTFT/module times.
while ! grep -q "\[after\] all done" /media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot/after_k128_queue.log 2>/dev/null; do
  echo "[repair] waiting for after_k128 queue 08:36:22"; sleep 300
done
while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5000 ] && break; echo "[repair] GPU ${u}MiB wait"; sleep 120
done
run () { local tag=$1 mp=$2
  [ -s "$OUT/$tag.json" ] && { echo "[skip] perf $tag"; return 0; }
  $PY $WRAP/llava_perf_panel.py --model-path $mp --image-folder $IMG \
    --question-file $Q --conv-mode vicuna_v1 --visual-token-num 32 \
    --anchorzip --beta 2.0 --alpha 0.5 --tag $tag --json-out $OUT/$tag.json \
    && echo "[ok] perf $tag" || echo "[FAIL] perf $tag"; }
run v15_AZ32 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run next_AZ32 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
touch /tmp/anchorzip_laneB_done.marker
echo "[laneB] all done $(date)" >> $WRAP/llava_lane_b.log
echo "[repair] done $(date)"
