"""
M3-v0 step 1 -- cache the pre-LLM audit state of the frozen 450.

One vision-tower forward + one EADP scoring pass per instance.  No LLM layer,
no generation, no backward.  Cached per instance:

    X   (1024, F)  the hand-built audit features, from `m3_common.handcrafted_t`
                   -- THE SAME function the live pruner calls, so there is no
                   train/serve skew by construction;
    vis (1024, 4096) fp16  the post-merger vision feature the tower already
                   produced (the student's only non-scalar input);
    s0  (256)      B2's selection, SORTED (the selector returns greedy order);
    g2  (1024)     the P1-G2 teacher, read from the S2-B cache -- TRAINING LABEL
                   ONLY, plus the ORACLE arm's ceiling measurement.

Row order is the M1 plan's, so `split` (fit/val/test) travels with the row and
the held-out 150 can never contribute a training label.

Usage
    python scripts/discovery/m3_features.py
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
from m1_common import load_m1_plan                                   # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine              # noqa: E402
from m3_common import (BASE_SELECTOR, BUDGET, FEATURES, GRID, N_VIS,  # noqa: E402
                       handcrafted_t, load_teacher)
from instrumented import attach_pruner                               # noqa: E402

TAG = "m3_bank"
SPLITS = ("fit", "val", "test")


def bank_rows(model):
    """The frozen 450 (150/benchmark) in M1-plan order, split-labelled."""
    _, plan, keys, rows_of = load_m1_plan()
    want = [(i, "fit") for i in rows_of["fit"]] + \
           [(i, "val") for i in rows_of["val"]] + \
           [(i, "test") for i in rows_of["test"]]
    cache, out = {}, []
    for i, split in want:
        p = plan[i]
        ds, idx = p["ds"], int(p["idx"])
        assert keys[i] == f"{ds}_{idx}"
        if ds not in cache:
            cache[ds] = common.build_dataset(ds)
        dataset = cache[ds]
        model.set_dump_image(dataset.dump_image)
        row = dataset.data.iloc[idx]
        out.append(dict(key=keys[i], ds=ds, idx=idx, split=split,
                        msg=common.build_message(model, dataset, ds, row)))
    assert len(out) == 450, len(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()

    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=64)
    model.model.eval()
    torch.set_grad_enabled(False)

    eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                       selector=BASE_SELECTOR, tag="M3BASE"))
    # Build the bank pruner the SAME way m2_accuracy builds B2's: attach_pruner
    # copies alpha/beta/visual_dim/spatial_merge_size off the loaded model, so
    # no literal can drift from the reference arm.
    pruner = attach_pruner(model, selector=BASE_SELECTOR, capture=False)
    pruner.visual_token_num = BUDGET
    pruner.sim_mode = "rebound"
    pruner.keep_gpu = True
    eng.pruner = pruner
    model.pruner = pruner
    assert eng.pruner is model.pruner

    rows = bank_rows(model)
    if args.limit:
        rows = rows[: args.limit]
    teacher = load_teacher()

    rec = {k: [] for k in ("key", "ds", "idx", "split", "s0", "g2")}
    Xs, VS = [], []
    t0 = time.time()
    for n_, it in enumerate(rows):
        prep = eng.prepare(it["msg"], it["ds"])
        vis = prep["vis"]
        text_llm, text_seq = eng._instruction_embeds(prep)
        pruner(vis, text_llm, text_seq, prep["gthw"])
        g = pruner.last_gpu
        s0 = torch.sort(g["select_idx"][0].to(torch.long)).values
        assert s0.numel() == BUDGET
        X = handcrafted_t(g, g["image_features"].float(), s0)
        assert X.shape[0] == N_VIS and torch.isfinite(X).all(), X.shape
        Xs.append(X.detach().float().cpu().numpy())
        VS.append(vis.detach().to(torch.float16).cpu().numpy())
        pruner.last_gpu = {}

        rec["key"].append(it["key"])
        rec["ds"].append(it["ds"])
        rec["idx"].append(it["idx"])
        rec["split"].append(it["split"])
        rec["s0"].append(s0.detach().cpu().numpy().astype(np.int32))
        rec["g2"].append(teacher[it["key"]].astype(np.float32))
        del prep, vis
        if (n_ + 1) % 50 == 0:
            print(f"  {n_+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    out = {k: (np.array(v) if k in ("key", "ds", "split") else np.stack(v))
           for k, v in rec.items()}
    out["X"] = np.stack(Xs)
    out["vis"] = np.stack(VS)
    # The feature NAMES travel with the columns, so a later append to FEATURES
    # cannot silently reinterpret an older bank.
    out["feature_names"] = np.array(list(FEATURES))
    path = os.path.join(OUTPUT_DIR, f"{args.tag}.npz")
    np.savez(path, **out)
    meta = dict(n=len(rows),
                splits={s: int(sum(1 for r in rec["split"] if r == s)) for s in SPLITS},
                ds_order=sorted(set(rec["ds"])), grid=GRID, budget=BUDGET,
                base_selector=BASE_SELECTOR, features=len(Xs[0][0]),
                vis_dim=int(out["vis"].shape[-1]),
                wall_seconds=float(time.time() - t0))
    with open(os.path.join(OUTPUT_DIR, f"{args.tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[saved] {path}  {meta}")


if __name__ == "__main__":
    main()
