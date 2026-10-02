#!/usr/bin/env bash
# Stage-1 visual-calibration discovery — full command sequence.
# Worktree: /media/disk2/YZX/research/EADP_amp (branch codex/anchor-merge-pilot)
# Env: qwen3vl_clean. Single A40. Protocol: PROTOCOL_NOTES.md (frozen 2026-10-02).
set -euo pipefail
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# 0. correctness gates G1-G4 (must PASS before anything else)
$PY s1n_gate.py --gate-samples 3

# 1. DEV panel: 4 arms x 300 (gen + official score)
$PY s1n_accuracy.py --split dev --arms A,B,C,D --mode both

# 2. analysis: table + paired bootstrap + overlap + GO/KILL readout
$PY s1n_analyze.py --split dev --arms BASE,A,B,C,D

# 3. paired selector latency
$PY s1n_perf.py --arms A,B,C,D --samples 10

# 4. GO path (ONLY if DEV shows a winner >= +0.3 macro):
#    freeze winner once -> CONFIRM panel 600
# $PY s1n_accuracy.py --split confirm --arms <WINNER> --mode both
# $PY s1n_analyze.py --split confirm --arms BASE,<WINNER>

# 5. GO+CONFIRM path: 2x2 Stage1/Stage2 combination with MAIN025
#    (A: EADP / B: EADP+MAIN025 / C: NEW-S1 / D: NEW-S1+MAIN025)

# KILL path: none of the above; freeze official EADP Stage 1 for DCC.
