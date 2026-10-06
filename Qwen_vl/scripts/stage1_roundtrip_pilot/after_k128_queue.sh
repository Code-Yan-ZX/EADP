#!/usr/bin/env bash
# After-k128 GPU queue (2026-10-07, user: "跑完推送,再找论文需要的跑着,别让卡停").
# Stage 1: E-side efficiency rows via official_replay (G1-verified) + LAM=0
#          (= official EADP hard pruning) for v1.5 {128,64,32} and NeXT
#          {128,64,32}/crop — the method-vs-pruning-baseline overhead table
#          (mirrors Qwen P2 E_GATHER rows).
# Stage 2: Qwen3 K=64 ten-task round (arm L_R_MAIN00625, third Qwen budget
#          point for the cross-model budget curve), bank rebuild + gen+score.
# Triggered after k128v2 sets its done marker; pushes nothing (STATUS/push
# handled separately to avoid double-push with the parallel session).
set -u
LOGDIR=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
while [ ! -e /tmp/qwen3_k128_done.marker ]; do
  echo "[after] waiting k128v2 marker $(date +%H:%M:%S)"; sleep 300
done
echo "[after] k128v2 done, waiting GPU $(date)"
while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5000 ] && break
  echo "[after] GPU ${u}MiB wait $(date +%H:%M:%S)"; sleep 300
done

PY=/home/dell/miniconda3/envs/llava_pruner/bin/python
WRAP=$LOGDIR
EV=/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval
LL=/media/disk2/YZX/research/EADP_amp/LLaVA
cd $LL; export PYTHONPATH=$LL
Q=$EV/textvqa/llava_textvqa_val_v051_ocr.jsonl
IMG=$EV/textvqa/train_images
OUT=$EV/anchorzip_p3/perf

run_e () { local tag=$1 vtn=$2 mp=$3
  [ -s "$OUT/$tag.json" ] && { echo "[skip] perf $tag"; return 0; }
  $PY $WRAP/llava_perf_panel.py --model-path $mp --image-folder $IMG \
    --question-file $Q --conv-mode vicuna_v1 --visual-token-num $vtn \
    --anchorzip --az-mode official_replay --az-lam 0.0 \
    --beta 2.0 --alpha 0.5 --tag $tag --json-out $OUT/$tag.json \
    && echo "[ok] perf $tag" || echo "[FAIL] perf $tag"; }

echo "[after] E-side panels start $(date)"
run_e v15_E128 128 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run_e v15_E64 64 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run_e v15_E32 32 /media/disk2/YZX/doct/FastV/llava-v1.5-7b
run_e next_E128 128 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
run_e next_E64 64 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
run_e next_E32 32 /media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b
echo "[after] E-side panels done $(date)"

# ---- Qwen3 K=64 third budget point ----
QPY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
DS=TextVQA_VAL,ChartQA_TEST,DocVQA_VAL,OCRBench,AI2D_TEST,HallusionBench,MME,MMBench_DEV_EN_V11,MMBench_DEV_CN_V11,InfoVQA_VAL
cd $WRAP
echo "[after] qwen K64 bank starts $(date)"
$QPY qwen3_k128_launcher.py --stage bank --datasets $DS \
  --k 64 --arm L_R_MAIN00625 --out-root stage1_roundtrip_pilot_k064 \
  > k064_bank.log 2>&1
B=$?
echo "[after] qwen K64 bank rc=$B $(date)"
[ "$B" -ne 0 ] && { echo "[after] K64 bank FAILED, stop"; exit 1; }
while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5000 ] && break
  echo "[after] GPU ${u}MiB wait $(date +%H:%M:%S)"; sleep 300
done
echo "[after] qwen K64 gen+score starts $(date)"
$QPY qwen3_k128_launcher.py --stage run --datasets $DS --mode both \
  --k 64 --arm L_R_MAIN00625 --out-root stage1_roundtrip_pilot_k064 \
  > k064_run.log 2>&1
G=$?
echo "[after] qwen K64 run rc=$G $(date)"
echo "[after] all done $(date)"
