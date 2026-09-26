"""
M6 step 1 -- the incumbent's exact continuation.

`R1 EADP-next` is the baseline the brief names first, and it has to be the real
thing rather than a re-ranking of the EADP score.  The incumbent selects with a
facility-location greedy (`_greed_select_impl`), whose step t depends only on
picks 0..t-1.  Running that same greedy to 1024 steps therefore *continues* the
256-step run the incumbent performs rather than approximating it, and picks
256..256+k-1 are exactly the tokens EADP would have taken next.

This script reads `importance` and `sim_matrix` off the real pruner (via the
same `TimedEADPPruner.last_gpu` capture `m5_bank.py` uses), re-runs the greedy
to full length, and writes the order.  Two gates:

    G-NEXT   the first 256 picks of the 1024-step run equal, in *order*, the
             selection order the live 256-step pruner produced;
    G-S0     their sorted set equals the bank's `s0`, so this order describes
             the same base selector every M3/M4/M5 number was measured against.

Outputs
    m6_eapd_order.npz   key, order (n, 1024), plus the two gate results

Usage
    python scripts/discovery/m6_eapd_order.py
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
from instrumented import SELECTORS, attach_pruner                    # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine, dump_json   # noqa: E402
from m5_common import BASE_SELECTOR, BUDGET, N_VIS, bank_items, load_bank  # noqa: E402

TAG = "m6_eapd_order"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=TAG,
                    help="output tag; use a different one for determinism checks "
                         "so the full cache is not overwritten")
    args = ap.parse_args()

    bank = load_bank()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0), max_new_tokens=64)
    model.model.eval()
    torch.set_grad_enabled(False)

    eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                       selector=BASE_SELECTOR, tag="M6ORDER"))
    pruner = attach_pruner(model, selector=BASE_SELECTOR, capture=False)
    pruner.visual_token_num = BUDGET
    pruner.sim_mode = "rebound"
    pruner.keep_gpu = True
    eng.pruner = model.pruner = pruner

    items = bank_items(model, bank, splits=("fit", "val", "test"))
    if args.limit:
        items = items[:args.limit]
    n = len(items)

    order = np.zeros((n, N_VIS), dtype=np.int32)
    bad_next, bad_s0 = [], []
    t0 = time.time()
    for j, it in enumerate(items):
        i = it["bank_row"]
        prep = eng.prepare(it["msg"], it["ds"])
        vis = prep["vis"]
        text_llm, text_seq = eng._instruction_embeds(prep)
        pruner(vis, text_llm, text_seq, prep["gthw"])
        g = pruner.last_gpu

        live = g["select_idx"][0].to(torch.long)                 # (256,) in order
        # The incumbent's own selector, run to full length.  `block8` fills the
        # budget in rounds of 8 and a round depends only on the picks before it,
        # so the first 256 picks of a 1024-long run are the 256-step run.
        full, _ = SELECTORS[BASE_SELECTOR](g["importance"], g["sim_matrix"], N_VIS)
        full = full[0].to(torch.long)                            # (1024,) in order

        if not torch.equal(full[:BUDGET], live):
            bad_next.append(it["key"])
        if not np.array_equal(np.sort(full[:BUDGET].cpu().numpy()),
                              np.sort(bank["s0"][i].astype(np.int64))):
            bad_s0.append(it["key"])
        order[j] = full.cpu().numpy().astype(np.int32)

        pruner.last_gpu = {}
        del prep, vis, g, full, live
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{n}  {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()

    keys = np.array([it["key"] for it in items])
    np.savez(os.path.join(common.OUTPUT_DIR, f"{args.tag}.npz"), key=keys, order=order)
    dump_json(f"{args.tag}_meta.json", dict(
        n=n, budget=BUDGET, selector=BASE_SELECTOR,
        G_NEXT=dict(checked=n, n_bad=len(bad_next), examples=bad_next[:5],
                    passed=bool(not bad_next),
                    what="first 256 greedy picks == live 256-step selection order"),
        G_S0=dict(checked=n, n_bad=len(bad_s0), examples=bad_s0[:5],
                  passed=bool(not bad_s0),
                  what="sorted first 256 == bank s0 (same base selector)"),
        wall_seconds=float(time.time() - t0)))
    print(f"[saved] {TAG}.npz  ({time.time()-t0:.0f}s, "
          f"G-NEXT bad {len(bad_next)}, G-S0 bad {len(bad_s0)})")


if __name__ == "__main__":
    main()
