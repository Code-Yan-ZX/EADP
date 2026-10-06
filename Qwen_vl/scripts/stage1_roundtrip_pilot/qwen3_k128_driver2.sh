#!/usr/bin/env bash
# Qwen3 K=128 rerun driver v2 (2026-10-06, user approved: "车道B跑完自动接上").
# Waits for lane B (NeXT) full completion, then: rebuild missing banks
# (existing 3 skipped per-sample), rerun gen+score for the Table-4 ten
# tasks (TextVQA shard resumes per-sample), score, and ONLY THEN set the
# done marker (fixes the unconditional-marker incident of the first run).
set -u
LOGDIR=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
while ! grep -q "\[laneB\] all done" $LOGDIR/llava_lane_b.log 2>/dev/null; do
  echo "[k128v2] waiting for lane B $(date +%H:%M:%S)"; sleep 300
done
echo "[k128v2] lane B finished, acquiring lock $(date)"
while ! mkdir /tmp/llava_full_driver.lock 2>/dev/null; do
  echo "[k128v2] lock held, wait $(date +%H:%M:%S)"; sleep 300
done
trap "rmdir /tmp/llava_full_driver.lock" EXIT

cd $LOGDIR
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
DS=TextVQA_VAL,ChartQA_TEST,DocVQA_VAL,OCRBench,AI2D_TEST,HallusionBench,MME,MMBench_DEV_EN_V11,MMBench_DEV_CN_V11,InfoVQA_VAL

while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5000 ] && break
  echo "[k128v2] GPU ${u}MiB, wait $(date +%H:%M:%S)"; sleep 300
done

echo "[k128v2] bank rerun starts $(date)"
$PY qwen3_k128_launcher.py --stage bank --datasets $DS >> k128_bank2.log 2>&1
B=$?
echo "[k128v2] bank rc=$B $(date)"

if [ "$B" -ne 0 ]; then
  echo "[k128v2] bank FAILED, abort (no marker set)"; exit 1
fi

while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5000 ] && break
  echo "[k128v2] GPU ${u}MiB, wait $(date +%H:%M:%S)"; sleep 300
done

echo "[k128v2] gen+score rerun starts $(date)"
$PY qwen3_k128_launcher.py --stage run --datasets $DS --mode both \
  >> k128_run2.log 2>&1
G=$?
echo "[k128v2] run rc=$G $(date)"

if [ "$G" -eq 0 ]; then
  touch /tmp/qwen3_k128_done.marker
  echo "[k128v2] marker set (REAL completion) $(date)"
else
  echo "[k128v2] run FAILED, NO marker $(date)"
fi
echo "[k128v2] all done $(date)"
