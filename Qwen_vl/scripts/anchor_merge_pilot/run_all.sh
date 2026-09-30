#!/usr/bin/env bash
# Anchor-Merge Pilot — full reproducible command sequence (protocol §9).
# Worktree: /media/disk2/YZX/research/EADP_amp (branch codex/anchor-merge-pilot)
# Env: qwen3vl_clean. Single A40.
set -euo pipefail
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../../outputs/anchor_merge_pilot"

# 1. frozen manifests (DEV 100/ds prefix; CONFIRM 200/ds from e0 confirm pool,
#    OCRBench topped with 36 unused-dev-suffix rows, image-disjoint)
$PY "$HERE/amp_manifest.py"

# 2. frozen anchor bank (official facility S + cos assignment, per sample)
$PY "$HERE/amp_bank.py" --split dev
$PY "$HERE/amp_bank.py" --split confirm      # only needed if DEV is positive

# 3. correctness gates G1-G5 (must PASS before any accuracy run)
$PY "$HERE/amp_correctness.py"
#   (run twice: the second pass performs the cross-process G5 comparison)

# 4. smoke: 5 arms x 30 questions
$PY "$HERE/amp_accuracy.py" --split dev --arms BASE,U025,U050,U100,S025 --smoke 10 --mode both

# 5. DEV panel: 5 arms x 300
$PY "$HERE/amp_accuracy.py" --split dev --arms BASE,U025,U050,U100,S025 --mode both
$PY "$HERE/amp_analyze.py" --split dev --arms BASE,U025,U050,U100,S025

# 6. if (and only if) analysis_dev.json reports any_positive=true with winner W:
#    CONFIRM panel: 4 arms x 600 (BASE, W, NORM, SHUF), winner kind/lam frozen
# $PY "$HERE/amp_accuracy.py" --split confirm --arms BASE,W,NORM,SHUF \
#       --winner-lam <W.lambda> --winner-kind <W.kind> --mode both
# $PY "$HERE/amp_analyze.py" --split confirm --arms BASE,W,NORM,SHUF --winner W

# 7. paired efficiency (BASE + winner)
# $PY "$HERE/amp_perf.py" --arms BASE,W --split dev --samples 30
