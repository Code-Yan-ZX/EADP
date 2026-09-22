#!/usr/bin/env bash
# S2-C2 pass 2, in diagnostic-priority order (each block is independently useful,
# so an early stop costs the least important results).
#
#  block 1  add/remove split of the swap.  teacher:k adds the teacher's best k
#           AND removes the student's worst k; these two arms separate them:
#             addteacher:k   teacher's best k in, k *random* student tokens out
#             remworst:k     student's worst k out, k tokens from neither set in
#           With teacher:k (= addteacher + remworst) and random:k (= neither),
#           this is a 2x2 that decides whether the value is in the tokens the
#           teacher brings or in the budget the student was wasting.
#  block 2  fine low-k resolution of the teacher curve.
#  block 3  the random control (3 seeds) -- the baseline the teacher curve has
#           to beat at matched k.
#  block 4  adversarial (mirror ordering) and shuffled (another image's map):
#           same value distribution / same overlap trajectory, wrong content.
#  block 5  remaining random ks and the F_rev identity check.
#
# identity_S / identity_T already ran (s2c2_identity.json) and reproduce the
# cached student and teacher accuracies exactly.
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/discovery/s2c2_swap.py --tag s2c2_swap --arms \
  addteacher:8:s0 addteacher:16:s0 addteacher:24:s0 addteacher:40:s0 \
  remworst:8:s0 remworst:16:s0 remworst:24:s0 remworst:40:s0 \
  teacher:6 teacher:12 teacher:24 teacher:40 teacher:80 \
  random:8:s0 random:16:s0 random:16:s1 random:16:s2 \
  random:32:s0 random:32:s1 random:32:s2 random:64:s0 random:64:s1 random:64:s2 \
  adversarial:8 adversarial:16 adversarial:32 adversarial:48 adversarial:64 adversarial:96 \
  shuffled:8 shuffled:16 shuffled:32 shuffled:48 shuffled:64 shuffled:96 \
  random:2:s0 random:4:s0 random:24:s0 random:40:s0 random:48:s0 \
  random:80:s0 random:96:s0 \
  F_rev:16 F_rev:32 F_rev:64 F_rev:96
