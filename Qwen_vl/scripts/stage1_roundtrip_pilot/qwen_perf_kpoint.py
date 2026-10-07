"""Qwen3 K-budget online efficiency point (closeout item 2).

One budget per process: sets RC.K then runs the frozen P2 online perf
harness (egather_p2_perf_legacy, arms FULL/E_GATHER/L_R_MAIN025,
interleaved fixed64+natural-EOS, table4 panel = first 5 rows per task).
The ONLY change vs the K=256 P2 run is the budget; output goes to a
dedicated append-only file.

E_GATHER here is timed through the NATIVE engine's official EADP
selection path (scorer='eadp'), not the port's official_replay+lam=0
Python path (the latter executes assignment/merge even at lam=0 and
would slow the baseline).

Usage: qwen3vl_clean python qwen_perf_kpoint.py --k 64 --out <json>
"""
from __future__ import annotations

import argparse
import os

import rtg_common as RC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", default="table4")
    args = ap.parse_args()

    RC.K = args.k
    import egather_p2_perf_legacy as P2
    sys_argv = [__file__, "--panel", args.panel, "--out", args.out]
    import sys
    sys.argv = sys_argv
    P2.main()
    print(f"[kpoint] K={args.k} written {args.out}", flush=True)


if __name__ == "__main__":
    main()
