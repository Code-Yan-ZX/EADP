#!/usr/bin/env bash
# S3-A — is visual-token utility conditional on the retained set? Full
# reproducible sequence. Each GPU stage resumes from its own output file, so
# re-running after an interruption is safe.
set -u
cd /media/disk2/YZX/research/EADP/Qwen_vl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
LOGDIR=outputs/discovery
export PYTHONUNBUFFERED=1

step () {
  name="$1"; shift
  echo "=== $name  $(date '+%F %T') ==="
  python -u "$@" 2>&1 | tee "$LOGDIR/$name.log"
  rc=${PIPESTATUS[0]}
  echo "=== $name exit=$rc  $(date '+%F %T') ==="
  [ "$rc" -eq 0 ] || { echo "STOPPING: $name failed"; exit "$rc"; }
}

# 1. freeze the analytic cases (CPU): subset, contexts, candidates, golds
step s3a_cases scripts/discovery/s3a_cases.py

# 2. gates + the conditional-utility measurement grid (GPU)
step s3a_nll scripts/discovery/s3a_nll.py

# 3. the rescue control (GPU)
step s3a_rescue scripts/discovery/s3a_rescue.py

# 4. statistics (CPU)
step s3a_analysis scripts/discovery/s3a_analysis.py

echo "ALL DONE  $(date '+%F %T')"
