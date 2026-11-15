#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$TASK_SCRIPT_DIR/../.."
PYTHON="${PYTHON:-/home/dell/miniconda3/envs/qwen3vl_clean/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export QWEN3_VLM_ATTN_IMPL=sdpa
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$PYTHON" -u -m scripts.student_transfer.run "$@"
