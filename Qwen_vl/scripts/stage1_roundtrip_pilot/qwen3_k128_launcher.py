"""Qwen3-VL-8B K=128 launcher (user-approved 2026-10-06).

Arm L_R_MAIN0125 at K=128, official_legacy scenario (DeepStack off, 1-D
positions, greedy) -- the frozen method RTG + official facility +
Completion lambda=0.25, only the budget moves 256 -> 128 (naming follows
the LLaVA-side convention: 025=K256, 0125=K128 on the Qwen side).

Isolation: RC.OUT_DIR is retargeted to outputs/stage1_roundtrip_pilot_k128
BEFORE importing the round modules, so the new banks (banks store the
K=256 selection result and CANNOT be truncated -- they must be rebuilt at
K=128) and all shards/scores land in a dedicated append-only root.  The
frozen K=256 banks and L_R_MAIN025 artifacts are untouched.

Usage (qwen3vl_clean env):
  python qwen3_k128_launcher.py --stage bank --datasets <csv>
  python qwen3_k128_launcher.py --stage run  --datasets <csv> [--mode both]
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["bank", "run"])
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    args, passthrough = ap.parse_known_args()

    import rtg_common as RC
    RC.K = 128
    RC.OUT_DIR = os.path.join(os.path.dirname(RC.OUT_DIR),
                              "stage1_roundtrip_pilot_k128")
    os.makedirs(os.path.join(RC.OUT_DIR, "full"), exist_ok=True)

    _orig_arm_cfg = RC.arm_cfg

    def arm_cfg_k128(name: str) -> dict:
        cfg = dict(_orig_arm_cfg(name))
        cfg["K"] = 128          # the ONLY change; lam=0.25 frozen
        return cfg

    RC.arm_cfg = arm_cfg_k128
    # re-inject --datasets: parse_known_args consumed it, but the child
    # parsers (rtg_bank_full / rtg_accuracy_legacy) need it too — without
    # this they silently fall back to their 3-dataset / Table-4 defaults
    # (root cause of the 2026-10-06 bank 3/10 + ChartQA crash incident).
    if args.datasets and "--datasets" not in passthrough:
        passthrough = passthrough + ["--datasets", args.datasets]
    sys_argv_backup = sys.argv
    sys.argv = [sys.argv[0]] + passthrough

    if args.stage == "bank":
        import rtg_bank_full as RBF
        RBF.main()
    else:
        import rtg_accuracy_legacy as RAL
        RAL.ARM = "L_R_MAIN0125"
        # retarget the import-time-derived path constants (egather
        # set_arm pattern): FULL/FAIL_LOG were built from the default arm
        RAL.FULL = os.path.join(RC.OUT_DIR, "legacy_full", "acc",
                                "L_R_MAIN0125")
        os.makedirs(RAL.FULL, exist_ok=True)
        RAL.FAIL_LOG = os.path.join(RC.OUT_DIR, "legacy_full",
                                    "score_failures.jsonl")
        RAL.main()
    sys.argv = sys_argv_backup


if __name__ == "__main__":
    main()
