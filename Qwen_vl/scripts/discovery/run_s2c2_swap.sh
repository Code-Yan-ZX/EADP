#!/usr/bin/env bash
# S2-C2 step 2: the budget-preserving swap grid on the held-out 150.
#   group A  teacher-priority   add T_only by descending teacher rank
#   group B  random             same overlap trajectory, random identity, 3 seeds
#   group C  adversarial        mirror ordering (teacher's weakest first)
#   group D  shuffled           another image's teacher map (content-free control)
#   group E  F_rev              from T, remove the teacher's weakest / add student's best
#
# Budget is 256 in every arm. identity_S / identity_T were run separately
# (s2c2_identity.json) and reproduce the cached student/teacher accuracies exactly.
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/discovery/s2c2_swap.py --tag s2c2_swap --arms \
  teacher:8 teacher:16 teacher:32 teacher:48 teacher:64 teacher:96 \
  random:8:s0 random:16:s0 random:16:s1 random:16:s2 \
  random:32:s0 random:32:s1 random:32:s2 \
  random:64:s0 random:64:s1 random:64:s2 random:96:s0 \
  adversarial:16 adversarial:32 adversarial:64 adversarial:96 \
  shuffled:16 shuffled:32 shuffled:64 shuffled:96 \
  F_rev:16 F_rev:32 F_rev:64 F_rev:96
