#!/usr/bin/env bash
# Closeout item 2 GPU chain (2026-10-07): waits for K64 gen+score to finish
# ("[after] all done" in after_k128_queue.log), then:
#   1) LLaVA official-path E rows: the E_GATHER generation path IS the
#      official encode_images at --visual_token_num K with NO --anchorzip;
#      tags v15_EOFF*/next_EOFF*.  The earlier official_replay+lam=0 rows
#      (v15_E*, next_E*) stay on disk but are invalid as BASELINE TIMING
#      (the port executes assignment/merge even at lam=0).
#   2) Qwen K=128 and K=64 online efficiency points (native engine).
# No AnchorZip (frozen) row is re-measured.
set -u
LOGDIR=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
while ! grep -q "\[after\] all done" $LOGDIR/after_k128_queue.log 2>/dev/null; do
  echo "[eff] waiting K64 queue $(date +%H:%M:%S)"; sleep 300
done
while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5000 ] && break
  echo "[eff] GPU ${u}MiB wait $(date +%H:%M:%S)"; sleep 300
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
  echo "[eff] $tag $(date +%T)"
  $PY $WRAP/llava_perf_panel.py --model-path $mp --image-folder $IMG \
    --question-file $Q --conv-mode vicuna_v1 --visual-token-num $vtn \
    --beta 2.0 --alpha 0.5 --tag $tag --json-out $OUT/$tag.json \
    && echo "[ok] perf $tag" || echo "[FAIL] perf $tag"; }
echo "[eff] llava official-path E rows start $(date)"
run_eoff v15_EOFF128 128 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run_eoff v15_EOFF64 64 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run_eoff v15_EOFF32 32 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run_eoff next_EOFF128 128 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
run_eoff next_EOFF64 64 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
run_eoff next_EOFF32 32 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
echo "[eff] qwen k-points start $(date)"
QP=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
cd $WRAP
$QP qwen_perf_kpoint.py --k 128 --panel table4 \
  --out /media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/stage1_roundtrip_pilot/legacy_full/p2_perf_legacy_k128.json \
  > qwen_perf_k128.log 2>&1 && echo "[ok] qwen K128 perf" || echo "[FAIL] qwen K128 perf"
while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5000 ] && break
  echo "[eff] GPU ${u}MiB wait $(date +%H:%M:%S)"; sleep 300
done
$QP qwen_perf_kpoint.py --k 64 --panel table4 \
  --out /media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/stage1_roundtrip_pilot/legacy_full/p2_perf_legacy_k64.json \
  > qwen_perf_k64.log 2>&1 && echo "[ok] qwen K64 perf" || echo "[FAIL] qwen K64 perf"
echo "[eff] all done $(date)"
