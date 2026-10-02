"""Stage-1 visual-calibration discovery — shared plumbing.

Round: visual NOVELTY / coverage-potential calibration of the EADP
importance map (user brief 2026-10-02).  Reuses the FROZEN
anchor-merge-pilot manifest (DEV 100/ds = 300; CONFIRM 200/ds = 600) and
its BASE shards, exactly like the stage1_grounding round.

Fixed for the whole round (Stage-2 freeze, user directive):
  official EADP Stage-1 scorer -> official facility-location selector
  (b1), K=256, identity gather (no Anchor Completion), training-free,
  token budget unchanged.  NO changes to text entropy / grounding
  filtering; NO changes to Facility or Anchor Completion.

The ONLY change an arm makes: after the official importance map is
computed (min-max -> smoothing k3 s1 -> ^2, verbatim), a visual signal is
min-max normalized and ADDED:

    importance' = importance + beta * minmax(visual_signal)

so beta = 0 is bitwise OFFICIAL (gate G1).  Facility then runs verbatim
on importance'.

Visual signals (on the official visual-visual similarity matrix, cos =
2*sim-1; main visual features only; no extra forward):

  v1_knn  kNN novelty : novelty_i = 1 - mean(top-k off-diagonal cosine),
          k = 16 frozen (brief allows 8 or 16; one value, not swept).
          Higher = the token's local feature neighbourhood cannot
          substitute for it.
  v3_cov  coverage potential : mass_i = sum_{j!=i} relu(cos_ij - tau),
          tau = mean off-diagonal cosine (self-adaptive threshold, not
          tuned).  Higher = the token represents many others.  NOT
          facility-location re-run -- a per-token representation-utility
          prior only.
  v2_local_global is SKIPPED (user brief allows: implementation cost not
          justified vs v1/v3 for a discovery round).

Arms (frozen before any accuracy run):
  BASE  official EADP (b1), reused from the amp round-2 shards
  A     v1_knn, beta=0.10
  B     v1_knn, beta=0.25
  C     v1_knn, beta=0.50
  D     v3_cov, beta=0.25 (single most-reasonable beta per brief)

GO/KILL (user brief, frozen): GO iff best arm macro >= BASE + 0.3 AND no
single benchmark collapses AND paired trend positive AND extra selector
latency negligible AND selected set is not identity.  Otherwise KILL and
freeze official EADP Stage 1 for DCC.  No re-tuning after the DEV readout.
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(os.path.dirname(_HERE))
DISC_DIR = os.path.join(QWEN_ROOT, "scripts", "discovery")
AMP_DIR = os.path.join(QWEN_ROOT, "scripts", "anchor_merge_pilot")
E0_DIR = os.path.join(QWEN_ROOT, "scripts", "e0")
for _p in (os.path.join(QWEN_ROOT, "VLMEvalKit"), QWEN_ROOT, DISC_DIR, E0_DIR,
           AMP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402  (LMUData / HF_HOME / offline env defaults)

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "stage1_visual_calibration")
ACC_DIR = os.path.join(OUT_DIR, "acc")
DIAG_DIR = os.path.join(OUT_DIR, "diag")
AMP_OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "anchor_merge_pilot")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ACC_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

DS_LIST = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
K = 256

# frozen arm registry (user brief section 4; documented in PROTOCOL_NOTES.md)
ARMS: dict[str, dict] = {
    "A": dict(variant="v1_knn", knn_k=16, beta=0.10),
    "B": dict(variant="v1_knn", knn_k=16, beta=0.25),
    "C": dict(variant="v1_knn", knn_k=16, beta=0.50),
    "D": dict(variant="v3_cov", knn_k=16, beta=0.25),
}
IDENTITY_ARM = dict(variant="v1_knn", knn_k=16, beta=0.0)  # gate G1 only


def arm_selector_name(arm: str) -> str:
    return f"s1n_{arm}"


def ensure_selectors() -> None:
    """Register the round's selectors into model.e0_selectors.  Idempotent."""
    from model import e0_selectors
    for name, cfg in [("id", IDENTITY_ARM)] + list(ARMS.items()):
        e0_selectors.register(arm_selector_name(name),
                              make_selector(cfg["variant"], cfg["beta"],
                                            cfg["knn_k"]))


def load_manifest() -> dict:
    import json
    p = os.path.join(AMP_OUT_DIR, "manifest.json")
    with open(p) as f:
        return json.load(f)


def shard_path(split: str, arm: str, ds: str) -> str:
    d = os.path.join(ACC_DIR, split, arm, f"K{K}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ds}.json")


def load_shard(path):
    import json
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"records": {}}


def save_shard(path, shard):
    import json
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=1)
    os.replace(tmp, path)


def degeneracy(text: str) -> dict:
    t = text.strip()
    toks = t.split()
    rep = 0
    if len(toks) >= 8:
        grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)]
        rep = max((grams.count(g) for g in set(grams)), default=0)
    return dict(empty=len(t) == 0, n_chars=len(t), repeat4=rep)


# ---------------------------------------------------------------------------
# visual signals + fused selector
# ---------------------------------------------------------------------------
def _minmax(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    rng = hi - lo
    return (x - lo) / rng if rng > 0 else torch.zeros_like(x)


def visual_signal(sim_matrix: torch.Tensor, variant: str,
                  knn_k: int = 16) -> torch.Tensor:
    """Per-visual-token signal from the OFFICIAL similarity matrix.

    sim_matrix: (1, N, N) official [0,1] visual-visual similarity
    (= 0.5*(cos+1), model/pruner.py::_sim_visual_impl).
    Returns (N,) raw signal; higher = more worth keeping.
    """
    cos = 2.0 * sim_matrix[0] - 1.0                  # (N, N) actual cosine
    offd = cos.clone()
    offd.fill_diagonal_(float("-inf"))               # exclude self
    if variant == "v1_knn":
        k = min(knn_k, cos.shape[0] - 1)
        nn = offd.topk(k, dim=-1).values.mean(-1)    # mean top-k cos
        return 1.0 - nn                              # novelty
    if variant == "v3_cov":
        N = cos.shape[0]
        tau = cos.sum() / (N * (N - 1))              # off-diagonal mean
        return torch.relu(offd - tau).sum(-1)        # neighbourhood mass
    raise KeyError(variant)


def make_selector(variant: str, beta: float, knn_k: int = 16):
    """Selector factory mirroring e0_selectors._eadp_parts verbatim, with
    the fused importance in place of the official one.  Selection itself
    is the verbatim official facility-location on the official sim."""
    from common import CudaTimer          # scripts/discovery/common.py
    import instrumented                   # scripts/discovery/instrumented.py
    from model.e0_selectors import _EADP_CACHE

    def _sel(K, ctx):
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
            if beta != 0.0:
                sig = _minmax(visual_signal(sim_matrix, variant, knn_k))
                importance = importance + beta * sig.view(1, -1).to(
                    importance.dtype)
            sel, _ = instrumented.SELECTORS["facility"](
                importance, sim_matrix, token_num)
            idx = sel[0].sort().values
            keep.append(offset + idx.to(V.device))
            offset += n
        return torch.cat(keep)

    return _sel


# ---------------------------------------------------------------------------
# one-sample runner: ONLINE selection via engine.generate (records
# selector_ms natively), keep_idx captured from the result.
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_one(eng, msg, ds, arm: str, max_new_tokens: int = 2048,
            timings: dict | None = None):
    out = eng.generate(msg, ds, K=K, selector=arm_selector_name(arm),
                       max_new_tokens=max_new_tokens, timings=timings)
    keep = out.get("keep_idx")
    keep = keep.detach().cpu().tolist() if keep is not None else None
    return out, keep
