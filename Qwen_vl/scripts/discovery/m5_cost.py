"""
M5 step 0b -- two measurements that size the teacher grid.

1. Does the bank's fp16 vision column change the teacher-forced NLL?  G-VIS
   showed it differs from the tower's bf16 output by at most 2.98e-8 (one fp16
   subnormal ulp) on 4.5e-5 of elements.  This puts a number on what that does
   to L, which is the only thing the teacher reads.
2. How does one teacher-forced NLL forward scale with batch?  The deletion
   measurements share a sequence length, so they can be stacked; whether that
   is worth it depends entirely on this curve.

Writes `m5_cost.json`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import eadp_model_name                                   # noqa: E402
from m5_common import BUDGET, bank_items, load_bank                  # noqa: E402
from m5_teacher import TeacherHarness                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="m5_cost")
    args = ap.parse_args()

    bank = load_bank()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=32)
    model.model.eval()
    torch.set_grad_enabled(False)
    h = TeacherHarness(model)

    items = bank_items(model, bank, splits=("fit",))
    rep = dict(n_instances=args.n, vis_effect=[], batch_curve=[])

    # ---- 1. fp16 bank vs live tower output, through L ----------------------
    from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine
    eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                       selector="block8", tag="M5C"))
    import s3a_common as C3
    for it in items[:args.n]:
        i = it["bank_row"]
        prep_live = eng.prepare(it["msg"], it["ds"])
        vis_live = prep_live["vis"].detach().clone()
        ans = h.answer_ids(C3.golds_of(it["row"]))
        s0 = np.sort(np.asarray(bank["s0"][i], dtype=np.int64)).tolist()

        # h.prep reads a numpy row; build the two prep dicts from the SAME
        # prompt skeleton so only the vision block differs.
        base = h.prep(it["msg"], it["ds"], bank["vis"][i])
        p_bank = base
        p_live = dict(base, vis=vis_live)
        L_bank, per_b, _ = h.nll(p_bank, s0, ans)
        L_live, per_l, _ = h.nll(p_live, s0, ans)
        mad = float((vis_live.float()
                     - base["vis"].float()).abs().max())
        rep["vis_effect"].append(dict(key=it["key"], L_bank=L_bank,
                                      L_live=L_live, dL=L_bank - L_live,
                                      vis_max_abs_diff=mad, n_golds=len(ans)))
        print(f"[vis] {it['key']:22s} dL = {L_bank - L_live:+.3e}  "
              f"(max |dvis| = {mad:.2e})", flush=True)
        del prep_live, vis_live, base, p_bank, p_live
        torch.cuda.empty_cache()

    # ---- 2. the batch curve ------------------------------------------------
    it = items[0]
    base = h.prep(it["msg"], it["ds"], bank["vis"][it["bank_row"]])
    ans = h.answer_ids(C3.golds_of(it["row"]))
    s0 = np.sort(np.asarray(bank["s0"][it["bank_row"]], dtype=np.int64)).tolist()
    gold = ans[0]
    for B in args.batches:
        rows = [[t for t in s0 if t != s0[j % len(s0)]] for j in range(B)]
        try:
            h.nll_multi(base, rows, gold)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                h.nll_multi(base, rows, gold)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3 / args.reps
            rep["batch_curve"].append(dict(batch=B, ms=ms, ms_per_row=ms / B))
            print(f"[batch] B={B:3d}  {ms:7.1f} ms   {ms/B:6.2f} ms/row",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[batch] B={B} OOM -- stopping")
            torch.cuda.empty_cache()
            break
        torch.cuda.empty_cache()

    from m2_gdep import dump_json
    dump_json(f"{args.tag}.json", rep)


if __name__ == "__main__":
    main()
