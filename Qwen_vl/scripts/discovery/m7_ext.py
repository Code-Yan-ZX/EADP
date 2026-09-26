"""
M7 step 4 -- the independent confirmation panel.

The test-150 screening result is inside the noise floor of a 150-instance panel
(M3-v0 measured MDE80 = 7.0 macro there; M4 measured 8.2).  Confirming or
killing M7 needs a panel that has never carried a single M6 or M7 number.

That panel already exists: M1's 720-row extension set, drawn 240 per benchmark
from outside the frozen bank, and asserted by `m1_plan.py` to be disjoint from
both the bank and the S2-A causal cases.  It carries no teacher scores, which
costs nothing here -- M7 selects nothing with the teacher, and the union is a
function of cheap features and the incumbent's own greedy order alone.

This script computes exactly those two things for the 720: the incumbent's full
greedy order, and the cheap feature matrix.  It is `m5_bank.py` and
`m6_eapd_order.py` fused into one pass, since both need the same forward.

Outputs
    m7_ext.npz   key, ds, idx, s0, order, X, feature_names

Usage
    python scripts/discovery/m7_ext.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                       # noqa: E402
from instrumented import SELECTORS, attach_pruner                    # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine, dump_json   # noqa: E402
from m5_common import (BASE_SELECTOR, BUDGET, FEATURES, GRID, N_VIS,  # noqa: E402
                       bank_items, safe_features)

TAG = "m7_ext"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    plan = json.load(open(os.path.join(OUTPUT_DIR, "m1_plan.json")))
    inst = plan["instances"]
    print(f"[panel] M1 extension: {len(inst)} rows "
          f"({plan['n_extra_per_ds']} per benchmark, seed {plan['extraset_seed']})")

    # a pseudo-bank: `bank_items` only needs these four columns
    bank = dict(key=[p["key"] for p in inst],
                ds=[p["ds"] for p in inst],
                idx=[p["idx"] for p in inst],
                split=["ext"] * len(inst))

    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0), max_new_tokens=64)
    model.model.eval()
    torch.set_grad_enabled(False)
    eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                       selector=BASE_SELECTOR, tag="M7EXT"))
    pruner = attach_pruner(model, selector=BASE_SELECTOR, capture=False)
    pruner.visual_token_num = BUDGET
    pruner.sim_mode = "rebound"
    pruner.keep_gpu = True
    eng.pruner = model.pruner = pruner

    items = bank_items(model, bank, splits=("ext",))
    if args.limit:
        items = items[:args.limit]
    n = len(items)

    order = np.zeros((n, N_VIS), dtype=np.int32)
    s0s = np.zeros((n, BUDGET), dtype=np.int32)
    Xs = np.zeros((n, N_VIS, len(FEATURES)), dtype=np.float32)
    bad_order, bad_s0 = [], []

    t0 = time.time()
    for j, it in enumerate(items):
        prep = eng.prepare(it["msg"], it["ds"])
        vis = prep["vis"]
        text_llm, text_seq = eng._instruction_embeds(prep)
        pruner(vis, text_llm, text_seq, prep["gthw"])
        g = pruner.last_gpu
        live = g["select_idx"][0].to(torch.long)
        full, _ = SELECTORS[BASE_SELECTOR](g["importance"], g["sim_matrix"], N_VIS)
        full = full[0].to(torch.long)
        s0 = torch.sort(live).values
        if not torch.equal(full[:BUDGET], live):
            bad_order.append(it["key"])
        if s0.numel() != BUDGET or torch.unique(s0).numel() != BUDGET:
            bad_s0.append(it["key"])
        X = safe_features(g, g["image_features"].float(), s0)
        order[j] = full.cpu().numpy().astype(np.int32)
        s0s[j] = s0.cpu().numpy().astype(np.int32)
        Xs[j] = X.detach().float().cpu().numpy()
        pruner.last_gpu = {}
        del prep, vis, X, g, full, live, s0
        if (j + 1) % 100 == 0:
            print(f"  {j+1}/{n}  {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()

    out = dict(key=np.array([it["key"] for it in items]),
               ds=np.array([it["ds"] for it in items]),
               idx=np.array([it["idx"] for it in items]),
               s0=s0s, order=order, X=Xs,
               feature_names=np.array(list(FEATURES)))
    np.savez(os.path.join(OUTPUT_DIR, f"{TAG}.npz"), **out)
    dump_json(f"{TAG}_meta.json", dict(
        n=n, budget=BUDGET, grid=GRID, base_selector=BASE_SELECTOR,
        source="m1_plan.json extension rows", features=list(FEATURES),
        G_ORDER=dict(checked=n, n_bad=len(bad_order), examples=bad_order[:5],
                     passed=bool(not bad_order),
                     what="first 256 greedy picks == live selection order"),
        G_S0=dict(checked=n, n_bad=len(bad_s0), examples=bad_s0[:5],
                  passed=bool(not bad_s0), what="live s0 is 256 unique indices"),
        wall_seconds=float(time.time() - t0)))
    print(f"[saved] {TAG}.npz  ({time.time()-t0:.0f}s, "
          f"order-bad {len(bad_order)}, s0-bad {len(bad_s0)})")


if __name__ == "__main__":
    main()
