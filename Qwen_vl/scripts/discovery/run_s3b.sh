#!/usr/bin/env bash
# S3-B — are the accuracy-critical teacher tokens small bundles? Full
# reproducible sequence. Every stage resumes from its own output file, so
# re-running after an interruption is safe; every stage also refuses to report
# success if any planned instance was skipped by an exception.
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

# 0. freeze the case bank: both bases, the F(k) machinery, the bank-L rescue
#    inventory recomputed on S2-C2's fine grid (CPU)
step s3b_cases scripts/discovery/s3b_cases.py

# 1. gates B1-B5 + the F(0) base points, 150 instances x 2 banks (GPU)
step s3b_base scripts/discovery/s3b_base.py

# 2. Part A: minimal rescue groups by prefix sweep (GPU)
step s3b_sweep scripts/discovery/s3b_sweep.py

# 3. Part B: the indivisibility arms on every bundle (GPU)
step s3b_ablate scripts/discovery/s3b_ablate.py

# 3b. the matched null class on instances the queue never rescues (GPU)
step s3b_nulls scripts/discovery/s3b_nulls.py

# 3c. substitution arms: membership changed at constant rank depth (GPU)
step s3b_subst scripts/discovery/s3b_subst.py

# 3d. rank-band ladder: P(hit) vs teacher-rank band at fixed set size (GPU)
step s3b_bands scripts/discovery/s3b_bands.py

# 4. Part C: the crop probe -- is a bundle a legible evidence unit? (GPU), plus
#    the check that calibrates it (positional convention / PNG re-encode; the
#    probe's absolute levels are only comparable inside one input path)
step s3b_crop scripts/discovery/s3b_crop.py
step s3b_poscheck scripts/discovery/s3b_poscheck.py

# 5. Part C: image-side cell statistics cache (CPU), then features, the three
#    structure tests and the verdict (CPU)
step s3b_structure scripts/discovery/s3b_structure.py

# 6. the figure
step s3b_figure scripts/discovery/s3b_figure.py

echo "ALL DONE  $(date '+%F %T')"
