#!/usr/bin/env bash
# Early-window EOFF panels (2026-10-07): run the 6 LLaVA official-path E
# rows during the K64 SCORING window (scoring is CPU-only, GPU idle) --
# panels need an exclusive GPU for valid timing, and CPU scoring provides
# exactly that.  Idempotent with after_k64_gpu_queue.sh (same files,
# [ -s ] skip).  Does NOT touch the Qwen k-points (they wait for the
# normal trigger in after_k64_gpu_queue.sh).
set -u
LOGDIR=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
# wait until the k064 run is past generation (GPU free) but scoring era:
# GPU < 5GB while the k064 output root already has shards
while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  n=$(ls /media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/stage1_roundtrip_pilot_k064/legacy_full/acc/L_R_MAIN00625/*.json 2>/dev/null | wc -l)
  if [ "$u" -lt 5000 ] && [ "$n" -ge 8 ]; then break; fi
  echo "[early] GPU ${u}MiB shards=$n, wait $(date +%H:%M:%S)"; sleep 180
done
PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=$LOGDIR
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
cd $LL; export PYTHONPATH=$LL
Q=$EV/textvqa/llava_textvqa_val_v051_ocr.jsonl
IMG=$EV/textvqa/train_images
OUT=$EV/anchorzip_p3/perf
run_eoff () { local tag=$1 vtn=$2 mp=$3
  [ -s "$OUT/$tag.json" ] && { echo "[skip] perf $tag"; return 0; }
  echo "[early] $tag $(date +%T)"
  $PY $WRAP/llava_perf_panel.py --model-path $mp --image-folder $IMG \
    --question-file $Q --conv-mode vicuna_v1 --visual-token-num $vtn \
    --beta 2.0 --alpha 0.5 --tag $tag --json-out $OUT/$tag.json \
    && echo "[ok] perf $tag" || echo "[FAIL] perf $tag"; }
run_eoff v15_EOFF128 128 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run_eoff v15_EOFF64 64 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run_eoff v15_EOFF32 32 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run_eoff next_EOFF128 128 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
run_eoff next_EOFF64 64 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
run_eoff next_EOFF32 32 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
echo "[early] panels done $(date)"
