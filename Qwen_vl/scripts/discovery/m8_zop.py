"""
M8 step 1 -- the ZOO-Prune estimator, reproduced as published, on the frozen 450.

What is reproduced, and from where
----------------------------------
ZOO-Prune (Kim et al., CVPR 2026, arXiv:2509.24837, github.com/AIM-SKKU/ZOO-Prune)
scores a visual token by perturbing the *pre-projector* vision feature with a
random direction, running the MM-projector, and reading the norm of the change
in the token's *projected* embedding:

    u_j ~ N(0, I_{d_v});  u_j <- u_j / ||u_j||_2            (Eq. 2)
    delta_{i,j} = [ M(x_i + h*u_j) - M(x_i - h*u_j) ] / 2h  (Eq. 3)
    S(i)        = (1/m) * sum_j || delta_{i,j} ||_2          (Eq. 4)
    S_hat(i)    = min-max normalisation of S over the image

with h = 0.01, m = 64 by default, and -- this is in the code and not in the
paper text -- **one shared bank of m directions applied to every token at once**
(`llava_arch.py`: `u_expanded = u.unsqueeze(1).expand(-1, N_v, -1)`).

On Qwen3-VL the projector is `Qwen3VLVisionPatchMerger` (LayerNorm(1152) ->
view(-1, 4608) -> Linear(4608,4608) -> GELU -> Linear(4608,4096)), applied to
the 2x2-merged patch blocks of the vision tower's output.  The pruner this
project measures operates on exactly that merger's output -- the (1024, 4096)
`image_features` -- so the merger IS the modality-alignment bottleneck ZOO-Prune
targets, and the perturbation is applied at the same place in the pipeline.

Gate G-MERGER proves the reproduction is exact rather than approximate: running
the captured merger input through the merger reproduces the tower's own output
bit-for-bit (max |delta| = 0 on every instance).

What this script writes
-----------------------
    m8_zop.npz     S (450, 1024) for each direction budget m, plus the cost
    m8_zop_meta.json

Usage
    python scripts/discovery/m8_zop.py
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
from m5_common import BUDGET, N_VIS, bank_items, load_m5_bank        # noqa: E402
from m8_common import ZO_DIRECTIONS, ZO_H, dump_json                 # noqa: E402

TAG = "m8_zop"


def zo_scores(merger, x: torch.Tensor, m: int, h: float, seed: int,
              chunk: int = 16):
    """ZOO-Prune Eq. 2-4.  `x` is (N_patch, d_v) bf16; returns S (N_merged,).

    The direction bank is shared across tokens exactly as the official code
    does.  Directions are drawn once in fp32 and normalised there -- drawing
    them in bf16 would quantise the direction itself before it is ever applied.
    """
    dev = x.device
    n_patch, d_v = x.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.randn(m, d_v, generator=g, dtype=torch.float32)
    U = U / U.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    U = U.to(dev, torch.float32)
    acc = None
    for c0 in range(0, m, chunk):
        Uc = U[c0:c0 + chunk]
        k = Uc.shape[0]
        xp = (x.float().unsqueeze(0) + h * Uc.unsqueeze(1)).to(x.dtype)
        xm = (x.float().unsqueeze(0) - h * Uc.unsqueeze(1)).to(x.dtype)
        zp = merger(xp.reshape(-1, d_v)).float().view(k, n_patch // 4, -1)
        zm = merger(xm.reshape(-1, d_v)).float().view(k, n_patch // 4, -1)
        d = (zp - zm) / (2.0 * h)
        s = d.norm(dim=-1)                                   # (k, N_merged)
        acc = s if acc is None else acc + s
        del xp, xm, zp, zm, d, s
    return (acc / m).mean(dim=0) if m > 1 else acc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dirs", type=int, nargs="+", default=list(ZO_DIRECTIONS))
    ap.add_argument("--h", type=float, default=ZO_H)
    ap.add_argument("--tag", type=str, default=TAG)
    args = ap.parse_args()

    bank = load_m5_bank()
    n = len(bank["key"])
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0), max_new_tokens=8)
    model.model.eval()
    torch.set_grad_enabled(False)
    dev = next(model.model.parameters()).device
    visual = model.model.visual
    merger = visual.merger
    print(f"[ZO-P] merger: {type(merger).__name__}  "
          f"d_v={merger.norm.normalized_shape[0]} "
          f"d_l={merger.linear_fc2.out_features}")

    items = bank_items(model, bank, splits=("fit", "val", "test"))
    if args.limit:
        items = items[:args.limit]
    print(f"[ZO-P] {len(items)} instances, direction budgets {args.dirs}")

    S = {m: np.zeros((len(items), N_VIS), dtype=np.float32) for m in args.dirs}
    gate = dict(checked=0, max_abs_diff=0.0, bad=[])
    cost = {m: [] for m in args.dirs}
    tower_ms, wall0 = [], time.time()

    for j, it in enumerate(items):
        inputs = model._processor_inputs(
            model._build_messages(it["msg"], dataset=it["ds"]))
        pv = inputs["pixel_values"].type(visual.dtype)
        gthw = inputs["image_grid_thw"]
        cap = {}
        hk = merger.register_forward_pre_hook(
            lambda mod, a: cap.__setitem__("x", a[0].detach().clone()))
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        t0.record()
        with torch.no_grad():
            out = visual(pv, grid_thw=gthw)
        t1.record()
        hk.remove()
        torch.cuda.synchronize()
        tower_ms.append(t0.elapsed_time(t1))
        vis = out[0] if isinstance(out, (tuple, list)) else out
        x = cap["x"]
        assert x.shape[0] == 4 * vis.shape[0], (x.shape, vis.shape)

        # G-MERGER: the captured input must reproduce the tower's own output
        with torch.no_grad():
            repro = merger(x)
        dd = float((repro.float() - vis.float()).abs().max())
        gate["checked"] += 1
        gate["max_abs_diff"] = max(gate["max_abs_diff"], dd)
        if dd != 0.0:
            gate["bad"].append(it["key"])

        for m in args.dirs:
            e0 = torch.cuda.Event(True)
            e1 = torch.cuda.Event(True)
            e0.record()
            with torch.no_grad():
                s = zo_scores(merger, x, m, args.h, seed=0)
            e1.record()
            torch.cuda.synchronize()
            cost[m].append(e0.elapsed_time(e1))
            S[m][j] = s.float().cpu().numpy()
        del out, vis, x, repro, cap
        torch.cuda.empty_cache()
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(items)}  {time.time()-wall0:.0f}s  "
                  f"tower {np.mean(tower_ms):.0f} ms  "
                  f"ZO-P(m={args.dirs[-1]}) {np.mean(cost[args.dirs[-1]]):.1f} ms",
                  flush=True)

    out = dict(key=np.array([it["key"] for it in items]),
               ds=np.array([it["ds"] for it in items]),
               **{f"S_m{m}": S[m] for m in args.dirs})
    np.savez(os.path.join(OUTPUT_DIR, f"{args.tag}.npz"), **out)
    dump_json(f"{args.tag}_meta.json", dict(
        n=len(items), h=args.h, dirs=list(args.dirs), seed=0,
        source="ZOO-Prune Eq.2-4 reproduced; shared direction bank; "
               "perturbation at the pre-merger vision features; response read "
               "at the merger output",
        G_MERGER=dict(checked=gate["checked"], max_abs_diff=gate["max_abs_diff"],
                      n_bad=len(gate["bad"]), examples=gate["bad"][:5],
                      passed=bool(not gate["bad"] and gate["max_abs_diff"] == 0.0),
                      what="merger(captured input) == tower output, bit-for-bit"),
        cost=dict(tower_ms_mean=float(np.mean(tower_ms)),
                  zo_ms={str(m): dict(mean=float(np.mean(cost[m])),
                                      median=float(np.median(cost[m])))
                         for m in args.dirs}),
        wall_seconds=float(time.time() - wall0)))
    print(f"\n[saved] {args.tag}.npz  ({time.time()-wall0:.0f}s)")
    print(f"  G-MERGER: {gate['checked']} checked, max|d| = {gate['max_abs_diff']}, "
          f"bad {len(gate['bad'])}")
    print(f"  tower {np.mean(tower_ms):.0f} ms;  " +
          "  ".join(f"ZO-P m={m}: {np.mean(cost[m]):.1f} ms" for m in args.dirs))


if __name__ == "__main__":
    main()
