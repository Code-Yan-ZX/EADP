#!/usr/bin/env bash
# S2-C2 pass 2: resolve the low-k end of the curve (where pass 1 already showed a
# large jump at k=8) and split the swap into its two halves.
#
#   teacher:k     add the teacher's best k, remove the student's worst k
#   addteacher:k  add the teacher's best k, remove k *random* student tokens
#   remworst:k    remove the student's worst k, add k tokens from neither set
#   random:k      both halves random  (the matched control)
#
# addteacher isolates the addition effect; remworst isolates the removal effect;
# teacher is their combination under the real orderings.
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/discovery/s2c2_swap.py --tag s2c2_swap --arms \
  teacher:2 teacher:4 teacher:6 teacher:12 teacher:24 teacher:40 teacher:80 \
  random:2:s0 random:4:s0 random:24:s0 random:40:s0 random:48:s0 random:80:s0 \
  addteacher:8:s0 addteacher:16:s0 addteacher:24:s0 addteacher:40:s0 \
  remworst:8:s0 remworst:16:s0 remworst:24:s0 remworst:40:s0 \
  adversarial:8 adversarial:48 \
  shuffled:8 shuffled:48
