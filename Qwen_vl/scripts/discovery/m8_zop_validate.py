"""
M8 step 5 -- is the ZOO-Prune reproduction faithful, and is `S ~ ||v||` real?

`m8_zop_diag.py` finds that the published estimator, run on Qwen3-VL's merger,
agrees with the token's post-merger feature norm at Spearman 0.986.  Two very
different things could produce that number, and they have opposite meanings:

    (a) the estimator is faithful and the merger's Jacobian norm genuinely
        tracks the output norm -- a fact about the estimator;
    (b) the perturbation never reaches the merger, because h*u at h = 0.01 is
        ~2.9e-4 per component against a bf16 ulp of the same order, so the
        finite difference is reading the *quantisation pattern* of x rather
        than a derivative -- an artefact of the dtype.

This script separates them by computing the same quantity two ways on the SAME
merger, once in bf16 (as deployed) and once in fp32 (where the step is clean):

    EXACT  reverse-mode random projections of the merger Jacobian:
               g = d(v . M(x)) / dx   for  v ~ N(0,I)/||.||
               ||J^T v||  estimates  ||J||_F / sqrt(d_l)
           no finite difference, no step size, no dtype ambiguity.
    FD     the published central difference, at three step sizes.

If FD tracks EXACT, the reproduction is faithful.  If EXACT also tracks ||v||,
the correlation is a property of the model and not of the arithmetic.

Usage
    python scripts/discovery/m8_zop_validate.py --limit 12
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
from common import OUTPUT_DIR, eadp_model_name                       # noqa: E402
from m5_common import BUDGET, N_VIS, bank_items, load_m5_bank        # noqa: E402
from m8_common import dump_json                                      # noqa: E402
from scipy.stats import spearmanr                                    # noqa: E402

TAG = "m8_zop_validate"
H_GRID = (0.01, 0.1, 1.0)
DIRS = 8


def fd_score(merger, x, h, seed, chunk=8):
    """Published central difference, on whatever dtype `merger`/`x` are."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.randn(DIRS, x.shape[1], generator=g, dtype=torch.float32)
    U = U / U.norm(dim=-1, keepdim=True)
    U = U.to(x.device, x.dtype)
    acc = 0.0
    for c0 in range(0, DIRS, chunk):
        Uc = U[c0:c0 + chunk]
        k = Uc.shape[0]
        # cast back to the working dtype before the merger, exactly as
        # m8_zop.py does -- otherwise the two runs round in different places
        xp = (x.unsqueeze(0) + h * Uc.unsqueeze(1)).to(x.dtype)
        xm = (x.unsqueeze(0) - h * Uc.unsqueeze(1)).to(x.dtype)
        zp = merger(xp.reshape(-1, x.shape[1])).float().view(k, -1, 4096)
        zm = merger(xm.reshape(-1, x.shape[1])).float().view(k, -1, 4096)
        acc = acc + ((zp - zm) / (2.0 * h)).norm(dim=-1)
        del xp, xm, zp, zm
    return (acc / DIRS).mean(dim=0)


def exact_score(merger, x_f32, seed, dirs=DIRS):
    """||J^T v|| averaged over random output directions -- no step size."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    acc = 0.0
    for _ in range(dirs):
        xr = x_f32.detach().clone().requires_grad_(True)
        z = merger(xr)
        v = torch.randn(z.shape[-1], generator=g, dtype=torch.float32).to(z.device)
        v = v / v.norm()
        (z.float() @ v).sum().backward()
        acc = acc + xr.grad.detach().view(-1, 4, x_f32.shape[-1]).norm(dim=(1, 2))
        del xr, z, v
    return acc / dirs


def survive_frac(x, u, h):
    """How much of the published perturbation survives the bf16 cast.

    The published step is applied to a bf16 tensor, so `x + h*u` is rounded
    back to bf16 before the merger ever sees it.  The fraction of the intended
    displacement that is still there afterwards is the honest measure of
    whether the estimator probes the merger or only its rounding -- far more
    direct than comparing h to an estimated ulp.
    """
    xp = (x.float() + h * u.float()).to(x.dtype).float()
    want = h * u.float().norm(dim=-1)
    got = (xp - x.float()).norm(dim=-1)
    return float((got / want.clamp_min(1e-30)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    bank = load_m5_bank()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0), max_new_tokens=8)
    model.model.eval()
    visual = model.model.visual
    merger = visual.merger
    items = bank_items(model, bank, splits=("test",))
    if args.limit:
        items = items[:args.limit]

    merger_f32 = None
    rows = []
    t0 = time.time()
    for j, it in enumerate(items):
        inputs = model._processor_inputs(
            model._build_messages(it["msg"], dataset=it["ds"]))
        pv = inputs["pixel_values"].type(visual.dtype)
        cap = {}
        hk = merger.register_forward_pre_hook(
            lambda m, a: cap.__setitem__("x", a[0].detach().clone()))
        with torch.no_grad():
            out = visual(pv, grid_thw=inputs["image_grid_thw"])
        hk.remove()
        vis = (out[0] if isinstance(out, (tuple, list)) else out)
        x = cap["x"]                                        # (4096, 1152) bf16
        xn = x.float().norm(dim=-1)
        g0 = torch.Generator().manual_seed(0)
        u0 = torch.randn(x.shape[1], generator=g0, dtype=torch.float32)
        u0 = (u0 / u0.norm()).to(x.device)
        surv = survive_frac(x, u0, 0.01)

        with torch.no_grad():
            fd_bf16 = {h: fd_score(merger, x, h, 0).float().cpu().numpy()
                       for h in H_GRID}
            fd_bf16_2 = fd_score(merger, x, 0.01, 1).float().cpu().numpy()

        if merger_f32 is None:
            import copy
            merger_f32 = copy.deepcopy(merger).float().eval()
            for p_ in merger_f32.parameters():
                p_.requires_grad_(False)
        x32 = x.float()
        torch.set_grad_enabled(True)
        ex = exact_score(merger_f32, x32, 0).cpu().numpy()
        torch.set_grad_enabled(False)
        with torch.no_grad():
            fd_f32 = fd_score(merger_f32, x32, 0.01, 0).cpu().numpy()
            out_norm = merger_f32(x32).float().norm(dim=-1).cpu().numpy()
        bank_norm = vis.float().norm(dim=-1).cpu().numpy()

        rows.append(dict(
            key=it["key"],
            rel_fd_bf16_vs_fd_f32=float(spearmanr(fd_bf16[0.01], fd_f32).statistic),
            rel_fd_f32_vs_exact=float(spearmanr(fd_f32, ex).statistic),
            rel_exact_vs_outnorm=float(spearmanr(ex, out_norm).statistic),
            rel_fd_f32_vs_outnorm=float(spearmanr(fd_f32, out_norm).statistic),
            rel_fd_bf16_vs_banknorm=float(spearmanr(fd_bf16[0.01], bank_norm).statistic),
            rel_exact_vs_banknorm=float(spearmanr(ex, bank_norm).statistic),
            rel_split_half_bf16=float(spearmanr(fd_bf16[0.01], fd_bf16_2).statistic),
            x_median_norm=float(xn.median()),
            survived_frac=float(surv),
            scale_h01=float(np.median(fd_bf16[0.01])),
            scale_h1=float(np.median(fd_bf16[1.0])),
            scale_ratio_h1_over_h01=float(np.median(fd_bf16[1.0])
                                          / max(np.median(fd_bf16[0.01]), 1e-12))))
        print(f"  {j+1}/{len(items)} {time.time()-t0:.0f}s  surv "
              f"{surv:.4f}  "
              f"fd_f32~exact {rows[-1]['rel_fd_f32_vs_exact']:.3f}  "
              f"exact~||out|| {rows[-1]['rel_exact_vs_outnorm']:.3f}  "
              f"fd_bf16~||bank|| {rows[-1]['rel_fd_bf16_vs_banknorm']:.3f}  "
              f"fd_bf16 scale h=1/h=0.01: {rows[-1]['scale_ratio_h1_over_h01']:.2f}",
              flush=True)
        del out, vis, x, x32, cap
        torch.cuda.empty_cache()

    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in rows[0] if k != "key"}
    dump_json(f"{TAG}.json", dict(n=len(rows), h_grid=list(H_GRID), dirs=DIRS,
                                  rows=rows, mean=agg))
    print("\n[mean]")
    for k, v in agg.items():
        print(f"  {k:34s} {v:+.4f}")
    print("\n  survived_frac = fraction of the intended h=0.01 displacement "
          "still present after the bf16 cast")


if __name__ == "__main__":
    main()
