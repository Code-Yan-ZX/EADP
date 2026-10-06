#!/usr/bin/env bash
# Qwen3-VL-8B K=128 driver (user-approved 2026-10-06): arm L_R_MAIN0125,
# official_legacy, ten Table-4 datasets.  Banks rebuilt at K=128 (bank
# records hold the K=256 selection result and cannot be truncated) into
# outputs/stage1_roundtrip_pilot_k128/ -- frozen K=256 artifacts untouched.
# Chained behind the LLaVA budget-points driver via the shared mkdir lock.
set -u
cd "$(dirname "$0")"
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
DS=TextVQA_VAL,ChartQA_TEST,DocVQA_VAL,OCRBench,AI2D_TEST,HallusionBench,MME,MMBench_DEV_EN_V11,MMBench_DEV_CN_V11,InfoVQA_VAL

while ! mkdir /tmp/llava_full_driver.lock 2>/dev/null; do
  echo "[k128] waiting for lock $(date +%T)"; sleep 300
done
trap "rmdir /tmp/llava_full_driver.lock" EXIT

echo "[k128] $(date) bank build starts (10 datasets)"
$PY qwen3_k128_launcher.py --stage bank --datasets $DS \
  > k128_bank.log 2>&1
B=$?
echo "[k128] $(date) bank rc=$B"

echo "[k128] $(date) gen+score starts"
$PY qwen3_k128_launcher.py --stage run --datasets $DS --mode both \
  > k128_run.log 2>&1
G=$?
echo "[k128] $(date) run rc=$G -- QWEN3 K128 DONE (bank=$B run=$G)"
touch /tmp/qwen3_k128_done.marker
echo "[k128] $(date) marker /tmp/qwen3_k128_done.marker set"
