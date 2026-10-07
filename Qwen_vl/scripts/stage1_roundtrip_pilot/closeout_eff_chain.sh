#!/usr/bin/env bash
set -u
LOGDIR=/media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot
while ! grep -q "\[early\] panels done" $LOGDIR/early_eoff_panels.log 2>/dev/null; do
  sleep 120; done
echo "[chain] panels done, qwen k-points start $(date +%T)"
QP=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
cd $LOGDIR
$QP qwen_perf_kpoint.py --k 128 --panel table4 \
  --out /media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/stage1_roundtrip_pilot/legacy_full/p2_perf_legacy_k128.json \
  > qwen_perf_k128.log 2>&1 && echo "[ok] qwen K128 perf" || echo "[FAIL] qwen K128 perf"
while true; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5000 ] && break
  echo "[chain] GPU ${u}MiB wait $(date +%T)"; sleep 300; done
$QP qwen_perf_kpoint.py --k 64 --panel table4 \
  --out /media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/stage1_roundtrip_pilot/legacy_full/p2_perf_legacy_k64.json \
  > qwen_perf_k64.log 2>&1 && echo "[ok] qwen K64 perf" || echo "[FAIL] qwen K64 perf"
touch /tmp/closeout_eff_done.marker
echo "[chain] all done $(date)"
