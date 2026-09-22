#!/usr/bin/env bash
# S2-C2 final assembly: everything downstream of the GPU runs, in order.
# Safe to re-run; every step is deterministic and overwrites its own output.
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1. second rescue planning pass, now against the dense curve the refinement
#    produced, so the transition blocks are as narrow as the grid allows
python scripts/discovery/s2c2_rescue.py --plan

# 2. execute whatever tighter arms that implies (no-op if already covered)
python scripts/discovery/s2c2_rescue.py --run --tag s2c2_rescue --auto

# 3. token properties -> group statistics (CPU; reads the S2-C1 memmaps)
python scripts/discovery/s2c2_characterize.py --compare --spatial \
    --rescue-plan outputs/discovery/s2c2_rescue_plan.json \
    | tee outputs/discovery/s2c2_char_compare.txt

# 4. tables, diagnosis JSON, figure
python scripts/discovery/s2c2_consolidate.py | tee outputs/discovery/s2c2_consolidate.txt
python scripts/discovery/s2c2_tables.py > outputs/discovery/s2c2_tables.md
python scripts/discovery/s2c2_figure.py
echo "--- deliverables in outputs/discovery/: s2c2_diagnosis.json, s2c2_tables.md,"
echo "    figures/s2c2_swap_curves.png"
