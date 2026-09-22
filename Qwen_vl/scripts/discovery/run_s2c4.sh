#!/usr/bin/env bash
# S2-C4 -- per-image direction transfer diagnosis.
#
# Entirely CPU: the LLM is never loaded. The only inputs are the S2-C1 layer-4
# feature cache and the P1-G2 teacher maps, both already on disk.
#
# No downstream generation is run in the normal path. The brief gates it on a
# deployable arm reaching head_recall@8 above S2-C3's 0.766 (ideally 0.82-0.85);
# consolidate.py evaluates the gate and records the outcome. If the gate is ever
# met, step 6 is the command to use.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
cd "$(dirname "$0")/../.."

TAG=${TAG:-s2c4}

# 1. oracle directions + cheap descriptors + standardisation stats
#    (~2 min, reads the 3.9 GB feature memmap twice: layer 4 and layer 2)
python scripts/discovery/s2c4_build.py --tag "$TAG"

# 2. the cross-image transfer matrix and every fit-set-derived arm
#    (~6 min, the dominant cost is 150 x 240 direction-token products)
python scripts/discovery/s2c4_transfer.py --tag "$TAG"

# 3. direction geometry: cosine structure, PCA spectrum, benchmark conditioning
python scripts/discovery/s2c4_geometry.py --tag "$TAG"

# 4. the low-dimensional test: oracle coefficient bound vs deployable ridge
python scripts/discovery/s2c4_pca.py --tag "$TAG"

# 5. identity check -- the vectorised metric path against brute force, plus the
#    two S2-C3 §6 calibration points. Exits non-zero on any disagreement.
python scripts/discovery/s2c4_verify.py --tag "$TAG"

# 6. apply the pre-registered decision rule and write the main JSON
python scripts/discovery/s2c4_consolidate.py --tag "$TAG"

# 7. figure
python scripts/discovery/s2c4_figure.py

# 8. ONLY if consolidate reports downstream_gate.downstream_run == true:
# python scripts/discovery/s2c0_run.py --scores "${TAG}_scores.npz" \
#     --arms <the arm(s) that cleared the gate> --tag "${TAG}_pilot"

echo
echo "S2-C4 complete. Main artifact: outputs/discovery/${TAG}_direction.json"
echo "Figure: outputs/discovery/figures/s2c4_direction_transfer.png"
