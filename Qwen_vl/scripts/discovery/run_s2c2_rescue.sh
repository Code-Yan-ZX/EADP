#!/usr/bin/env bash
# S2-C2 pass 3: answer-rescue refinement on the 28 rescuable instances.
#
#   1. re-plan against the *final* fine teacher grid, so the transition block
#      (the tokens whose joint addition flips the answer) is as narrow as the
#      grid allows;
#   2. run the dense fine grid on those 28 instances (a few minutes -- it is
#      ~1/5 of the full benchmark set);
#   3. re-plan on that dense curve to tighten the blocks;
#   4. run the leave-one-out ablations that the tightened blocks imply, plus the
#      early-rank fragility control and the within-block shuffle.
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/discovery/s2c2_rescue.py --plan
python scripts/discovery/s2c2_rescue.py --run --tag s2c2_rescue --auto
echo "--- second planning pass on the dense curve"
python scripts/discovery/s2c2_rescue.py --plan
python scripts/discovery/s2c2_rescue.py --run --tag s2c2_rescue --auto
