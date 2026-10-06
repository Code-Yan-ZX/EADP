#!/usr/bin/env bash
set -u
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
MP=/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
cd $LL; export PYTHONPATH=$LL
# wait for free GPU
while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 5000 ] && break; echo "[wait] GPU ${u}MiB"; sleep 300; done
echo "[rerun] start $(date)"
$PY $WRAP/llava_eval_arm_model_vqa.py --model-path $MP \
  --question-file $EV/textvqa/llava_textvqa_val_v051_ocr.jsonl \
  --image-folder $EV/textvqa/train_images \
  --answers-file $EV/anchorzip_p3/next_textvqa/FULL_rerun.jsonl \
  --temperature 0 --conv-mode vicuna_v1 \
  --visual_token_num 0 --beta 2.0 --alpha 0.5 \
  >> $EV/anchorzip_p3/next_textvqa/FULL_rerun.log 2>&1 \
  && echo "[ok] rerun rc=0" || echo "[FAIL] rerun"
echo "[rerun] done $(date)"
