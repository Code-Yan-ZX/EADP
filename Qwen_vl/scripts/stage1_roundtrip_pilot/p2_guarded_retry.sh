#!/usr/bin/env bash
# Guarded P2 retry (2026-10-04 night): waits until GPU has >=30GB free
# sustained for 60s (external 26.5GB job pending), then runs both perf
# panels with ABSOLUTE output paths.  Aborts the run if free memory
# drops below 15GB mid-run is NOT handled — measurements under
# concurrent external load are invalid; the pre-check gate is the guard.
set -u
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
OUT=/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/stage1_roundtrip_pilot/legacy_full
cd /media/disk2/YZX/research/EADP_amp/Qwen_vl/scripts/stage1_roundtrip_pilot

wait_gpu () {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    free=$((46000 - used))
    if [ "$free" -ge 30000 ]; then
      sleep 60
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      free=$((46000 - used))
      [ "$free" -ge 30000 ] && return 0
    fi
    sleep 60
  done
}

echo "[p2retry] $(date) waiting for >=30GB free sustained 60s..."
wait_gpu
echo "[p2retry] $(date) GPU clean — running dev30"
$PY egather_p2_perf_legacy.py --panel dev30 \
   --out "$OUT/p2_perf_legacy_dev30.json" > p2_perf_dev30_retry.log 2>&1
echo "[p2retry] $(date) dev30 rc=$?"
wait_gpu
echo "[p2retry] $(date) GPU clean — running table4"
$PY egather_p2_perf_legacy.py --panel table4 \
   --out "$OUT/p2_perf_legacy_table4.json" > p2_perf_table4_retry.log 2>&1
echo "[p2retry] $(date) table4 rc=$?"
ls -la "$OUT"/p2_perf_legacy_*.json 2>/dev/null >> p2_retry_driver.log
touch /tmp/p4_p2_done.marker
echo "[p2retry] $(date) DONE"
