"""E0 selector registry (prereg §3.2 / §4).

A selector sees one context dict and returns ``keep_idx``: a 1-D LongTensor of
visual-token indices, raster-order ascending, duplicate-free, length = budget
(or N when N <= K).  The engine does everything else (shared shrink of the
main features, the three DeepStack streams and the 3-D positions).

Local methods (kept bit-compatible with their legacy callers):
  * identity            -- keep everything (B0 / N1/N2);
  * random              -- uniform control;
  * b1                  -- official EADP scoring + facility-location
                           (instrumented.TimedEADPPruner stages, selector
                           'facility');
  * b2                  -- official EADP scoring + block8 facility
                           (selector 'block8'), the project incumbent;
  * divprune / cdpruner -- repo pruners (Qwen_vl/model/{divpruner,cdpruner}.py),
                           index-exposing wrappers around their own cores;
  * hiprune             -- repo HiPruner on the vision-attention importance
                           from model_fixed_res.qwen3_visual_forward_with_importance.

Official ports (FastV / PDrop / SparseVLM / VisionZip / PACE) register into
SELECTORS from ``Qwen_vl/model/baselines/`` in M2.
"""

from __future__ import annotations

import os
import sys

import torch

QWEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISC_DIR = os.path.join(QWEN_ROOT, "scripts", "discovery")
for _p in (QWEN_ROOT, DISC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _identity(K, ctx):
    prep = ctx["prep"]
    return torch.arange(prep["n_vis"], device=ctx["V"].device)


def _random(K, ctx):
    g = ctx.get("gen", None) or torch.Generator(device="cpu")
    if ctx.get("seed") is not None:
        g.manual_seed(int(ctx["seed"]))
    n = ctx["prep"]["n_vis"]
    if K >= n:
        return torch.arange(n, device=ctx["V"].device)
    idx = torch.randperm(n, generator=g)[:K]
    return idx.sort().values.to(ctx["V"].device)


# -- EADP scoring + selectable final operator (b1 / b2) ---------------------
_EADP_CACHE = {}


def _eadp_parts(K, ctx, selector_name: str):
    """Official scoring stages + official selector, per image, via the
    instrumented pruner (imported verbatim; scoring code never duplicated)."""
    from common import CudaTimer          # scripts/discovery/common.py
    import instrumented                   # scripts/discovery/instrumented.py

    eng = ctx["engine"]
    prep = ctx["prep"]
    V, gthw = ctx["V"], prep["gthw"]
    sms = eng.inner.visual.spatial_merge_size
    key = id(eng)
    pruner = _EADP_CACHE.get(key)
    if pruner is None:
        dim = V.shape[-1]
        from model.pruner import VisualTokenPruner
        pruner = instrumented.TimedEADPPruner(
            visual_token_num=K, alpha=0.5, beta=2.0, visual_dim=dim,
            spatial_merge_size=sms)
        pruner.eval()
        _EADP_CACHE[key] = pruner

    split_sizes = (gthw.prod(-1) // (sms ** 2)).tolist()
    keep, offset = [], 0
    for i, n in enumerate(split_sizes):
        token_num = min(K, n)
        if token_num >= n:
            keep.append(torch.arange(offset, offset + n, device=V.device))
            offset += n
            continue
        feats = V[offset:offset + n].unsqueeze(0)
        te = ctx["text_mean"][i:i + 1]
        ts = ctx["text_seq"][i:i + 1]
        gh = int(gthw[i, 1]) // sms
        gw = int(gthw[i, 2]) // sms
        importance = pruner._score(feats, te, ts, gh, gw, CudaTimer())
        sim_matrix = pruner._similarity(feats)
        sel, _ = instrumented.SELECTORS[selector_name](
            importance, sim_matrix, token_num)
        idx = sel[0].sort().values
        keep.append(offset + idx.to(V.device))
        offset += n
    return torch.cat(keep)


def _b1(K, ctx):
    return _eadp_parts(K, ctx, "facility")


def _b2(K, ctx):
    return _eadp_parts(K, ctx, "block8")


# -- DivPrune ----------------------------------------------------------------
def _divprune(K, ctx):
    from model.divpruner import DivPruner

    eng, prep, V, gthw = ctx["engine"], ctx["prep"], ctx["V"], ctx["prep"]["gthw"]
    sms = eng.inner.visual.spatial_merge_size
    p = DivPruner(visual_token_num=K, spatial_merge_size=sms).to(V.device)
    split_sizes = (gthw.prod(-1) // (sms ** 2)).tolist()
    keep, offset = [], 0
    for n in split_sizes:
        token_num = min(K, n)
        if token_num >= n:
            keep.append(torch.arange(offset, offset + n, device=V.device))
        else:
            s, _ = p.divprune_core(V[offset:offset + n], n, num_keep=token_num)
            keep.append(offset + s.sort().values.to(V.device))
        offset += n
    return torch.cat(keep)


# -- CDPruner -----------------------------------------------------------------
def _cdpruner(K, ctx):
    from model.cdpruner import CDPruner

    eng, prep, V, gthw = ctx["engine"], ctx["prep"], ctx["V"], ctx["prep"]["gthw"]
    sms = eng.inner.visual.spatial_merge_size
    dim = V.shape[-1]
    p = CDPruner(visual_token_num=K, visual_dim=dim,
                 spatial_merge_size=sms).to(V.device)
    split_sizes = (gthw.prod(-1) // (sms ** 2)).tolist()
    keep, offset = [], 0
    for i, n in enumerate(split_sizes):
        token_num = min(K, n)
        if token_num >= n:
            keep.append(torch.arange(offset, offset + n, device=V.device))
        else:
            idx = p.select_indices(V[offset:offset + n],
                                   ctx["text_mean"][i:i + 1], token_num)
            keep.append(offset + idx.to(V.device))
        offset += n
    return torch.cat(keep)


# -- HiPrune ------------------------------------------------------------------
def _hiprune(K, ctx):
    from model.hipruner import HiPruner

    eng, prep, gthw = ctx["engine"], ctx["prep"], ctx["prep"]["gthw"]
    V = ctx["V"]
    sms = eng.inner.visual.spatial_merge_size
    p = HiPruner(object_layer=4, alpha=0.5, retain=0.5, visual_token_num=K,
                 spatial_merge_size=sms).to(V.device)
    attn_list = ctx["attn_list"]
    split_sizes = (gthw.prod(-1) // (sms ** 2)).tolist()
    keep, offset = [], 0
    for i, n in enumerate(split_sizes):
        token_num = min(K, n)
        if token_num >= n:
            keep.append(torch.arange(offset, offset + n, device=V.device))
        else:
            idx = p.select_indices(attn_list, gthw, i, token_num)
            keep.append(offset + idx.to(V.device))
        offset += n
    return torch.cat(keep)


SELECTORS = {
    "identity": _identity,
    "random": _random,
    "b1": _b1,
    "b2": _b2,
    "divprune": _divprune,
    "cdpruner": _cdpruner,
    "hiprune": _hiprune,
}


def register(name: str, fn):
    SELECTORS[name] = fn


_BASELINE_MODULES = {"fastv", "pdrop", "sparsevlm", "visionzip"}


def run_selector(name: str, K: int, ctx: dict) -> torch.Tensor:
    if name in _BASELINE_MODULES:
        import importlib
        mod = importlib.import_module(f"model.baselines.{name}")
        return mod.selector(K, ctx)
    if name not in SELECTORS:
        raise KeyError(f"unknown selector {name!r}; have {sorted(SELECTORS)}")
    return SELECTORS[name](K, ctx)
