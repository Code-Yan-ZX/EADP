"""
M5 step 0c -- the bit-exact bank.

`m5_recon.py` measured that the m3 bank's fp16 vision column is NOT the tower's
bf16 output: it differs by at most 2.98e-8 (one fp16 subnormal ulp) on 4.5e-5
of elements, and `m5_cost.py` measured what that does to the teacher -- on one
of three instances L moved by **+2.79e-2 nats**.  That is the same order as the
deletion effects this line exists to measure, so the bank cannot be used as-is.

This rebuilds the vision column as bf16-exact (stored as a uint16 view, the
S3-A convention) for all 450, together with the M5 feature matrix, and gates
the result against the m3 bank's S0:

    G-S0   every instance's live select_idx == the m3 bank's s0.  If this
           holds, the two banks describe the same base selector and every M3/M4
           number remains comparable to an M5 number.
    G-VIS  re-loading a stored row reproduces the live tower output bit-for-bit.

Outputs
    m5_bank.npz   key, ds, idx, split, s0, g2, X, feature_names, meta
    m5_vis.npy    (450, 1024, 4096) uint16 == bf16, memory-mappable

Usage
    python scripts/discovery/m5_bank.py
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
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine, dump_json   # noqa: E402
from m5_common import (BASE_SELECTOR, BUDGET, FEATURES, GRID, N_VIS,  # noqa: E402
                       bank_items, load_bank, safe_features)
from instrumented import attach_pruner                               # noqa: E402

TAG = "m5_bank"
VIS_NPY = "m5_vis.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()

    m3 = load_bank()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=64)
    model.model.eval()
    torch.set_grad_enabled(False)

    eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                       selector=BASE_SELECTOR, tag="M5BANK"))
    pruner = attach_pruner(model, selector=BASE_SELECTOR, capture=False)
    pruner.visual_token_num = BUDGET
    pruner.sim_mode = "rebound"
    pruner.keep_gpu = True
    eng.pruner = model.pruner = pruner

    items = bank_items(model, m3, splits=("fit", "val", "test"))
    if args.limit:
        items = items[:args.limit]
    n = len(items)

    vis_mm = np.lib.format.open_memmap(
        os.path.join(OUTPUT_DIR, VIS_NPY), mode="w+", dtype=np.uint16,
        shape=(n, N_VIS, 4096))
    Xs = np.zeros((n, N_VIS, len(FEATURES)), dtype=np.float32)
    s0s = np.zeros((n, BUDGET), dtype=np.int32)
    g2s = np.zeros((n, N_VIS), dtype=np.float32)

    t0 = time.time()
    n_s0_bad = []
    for j, it in enumerate(items):
        i = it["bank_row"]
        prep = eng.prepare(it["msg"], it["ds"])
        vis = prep["vis"]
        text_llm, text_seq = eng._instruction_embeds(prep)
        pruner(vis, text_llm, text_seq, prep["gthw"])
        g = pruner.last_gpu
        s0 = torch.sort(g["select_idx"][0].to(torch.long)).values
        assert s0.numel() == BUDGET
        if not np.array_equal(s0.detach().cpu().numpy(),
                              np.sort(m3["s0"][i].astype(np.int64))):
            n_s0_bad.append(it["key"])
        X = safe_features(g, g["image_features"].float(), s0)
        vis_mm[j] = vis.detach().view(torch.uint16).cpu().numpy()
        Xs[j] = X.detach().float().cpu().numpy()
        s0s[j] = s0.detach().cpu().numpy().astype(np.int32)
        g2s[j] = np.asarray(m3["g2"][i], dtype=np.float32)
        pruner.last_gpu = {}
        del prep, vis, X, g
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    vis_mm.flush()
    del vis_mm

    # ---- G-VIS: reload a stored row and compare it to a fresh tower pass ----
    vm = np.load(os.path.join(OUTPUT_DIR, VIS_NPY), mmap_mode="r")
    rows = []
    for j in [0, n // 3, 2 * n // 3, n - 1]:
        it = items[j]
        prep = eng.prepare(it["msg"], it["ds"])
        live = prep["vis"]
        stored = torch.from_numpy(np.array(vm[j])).view(torch.bfloat16) \
            .to(live.device)
        rows.append(dict(key=it["key"], bit_exact=bool(torch.equal(live, stored)),
                         max_abs_diff=float((live.float()
                                             - stored.float()).abs().max())))
        print(f"[G-VIS] {it['key']:22s} bit_exact={rows[-1]['bit_exact']}",
              flush=True)
        del prep, live, stored
        torch.cuda.empty_cache()
    del vm

    out = dict(key=np.array([it["key"] for it in items]),
               ds=np.array([it["ds"] for it in items]),
               idx=np.array([it["idx"] for it in items]),
               split=np.array([it["split"] for it in items]),
               s0=s0s, g2=g2s, X=Xs,
               feature_names=np.array(list(FEATURES)))
    np.savez(os.path.join(OUTPUT_DIR, f"{args.tag}.npz"), **out)
    dump_json(f"{args.tag}_meta.json", dict(
        n=n, vis_file=VIS_NPY, grid=GRID, budget=BUDGET,
        base_selector=BASE_SELECTOR, features=list(FEATURES),
        splits={s: int(sum(1 for it in items if it["split"] == s))
                for s in ("fit", "val", "test")},
        G_S0=dict(checked=n, n_bad=len(n_s0_bad), examples=n_s0_bad[:5],
                  passed=bool(not n_s0_bad)),
        G_VIS=dict(rows=rows, passed=all(r["bit_exact"] for r in rows)),
        wall_seconds=float(time.time() - t0)))
    print(f"[saved] {args.tag}.npz + {VIS_NPY}  "
          f"({time.time()-t0:.0f}s, s0 mismatches: {len(n_s0_bad)})")


if __name__ == "__main__":
    main()
