"""
M2 — one model load, three stages, in the pre-registered order.

Loading Qwen3-VL-8B from this host's disk takes ~7 minutes, so the correctness
gates, the timing grid and the accuracy grid share a single process. The order is
not a convenience: the pre-registration requires every correctness gate to pass
*before* an accuracy or timing number is produced, and this driver enforces it --
a failed gate aborts the run with a non-zero exit code and no performance or
accuracy stage is executed.

Usage
    python scripts/discovery/m2_run.py                      # all three stages
    python scripts/discovery/m2_run.py --stages perf accuracy
    python scripts/discovery/m2_run.py --gates A B C
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                       # noqa: E402
import m2_accuracy                                                  # noqa: E402
import m2_correctness                                               # noqa: E402
import m2_perf                                                      # noqa: E402
from common import eadp_model_name                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["correctness", "perf", "accuracy"])
    ap.add_argument("--gates", nargs="+", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--perf-arms", nargs="+", default=[a["tag"] for a in m2_perf.ARMS])
    ap.add_argument("--acc-arms", nargs="+", default=list(m2_accuracy.ARM_SPEC))
    ap.add_argument("--acc-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--acc-resume", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[M2] loading model ...", flush=True)
    model = common.load_model(eadp_model_name(256, 0.5, 2.0),
                              max_new_tokens=m2_accuracy.MAX_NEW)
    model.model.eval()
    print(f"[M2] model loaded in {time.time()-t0:.0f} s", flush=True)

    if "correctness" in args.stages:
        rep = m2_correctness.stage(model, args.gates, "m2_correctness")
        if not rep.get("all_passed"):
            print("\n[M2] CORRECTNESS GATE FAILED -- stopping before any "
                  "performance or accuracy measurement (prereg §4).")
            sys.exit(2)

    if "perf" in args.stages:
        pa = argparse.Namespace(iters=args.iters, warmup=args.warmup,
                                arms=args.perf_arms, decode_tokens=32,
                                tag="m2_perf")
        m2_perf.stage(model, pa)

    if "accuracy" in args.stages:
        aa = argparse.Namespace(arms=args.acc_arms, seeds=args.acc_seeds,
                                tag="m2_accuracy", resume=args.acc_resume)
        m2_accuracy.stage(model, aa)

    print(f"\n[M2] done in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
