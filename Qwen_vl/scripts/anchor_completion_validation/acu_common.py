"""Anchor Completion Validation — shared plumbing.

Supplemental protocol: docs/anchor_completion_validation_protocol.md (frozen
2026-10-02, commit 7638b54).  Main judgment / stats / prereg fallback live in
docs/anchor_merge_full_eval_prereg.md; merge math, deterministic reduction,
bank freezing and scoring rules are inherited verbatim from the
anchor_merge_pilot / stage1_cross_stream_pilot rounds.

This round adds THREE operator ablations on the SAME frozen bank (same
keep/gid) as BASE/MAIN025, restricted to the main feature stream:

  BASE        identity gather (selector b1 = official facility, K=256)
  MAIN025     uniform group mean, lam=0.25, main only      (formal method)
  MAIN100     uniform group mean, lam=1.00, main only      (amplitude ablation)
  MAIN_SIM025 softmax(cos/0.1) weights incl anchor logit 1/0.1,
              then lam=0.25 update, main only              (weight-form ablation)
  BOTH025     uniform lam=0.25, main+DS (ONLY if the prereg §4 mechanical
              trigger fires; conditional secondary analysis)

PruMerge / Libra full-method controls: NOT implementable faithfully on this
path (protocol §4.3) — recorded as not-completed, never faked.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from typing import Optional

import torch

# -- path bootstrap (same convention as amp_common) --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(os.path.dirname(_HERE))
AMP_DIR = os.path.join(QWEN_ROOT, "scripts", "anchor_merge_pilot")
DISC_DIR = os.path.join(QWEN_ROOT, "scripts", "discovery")
E0_DIR = os.path.join(QWEN_ROOT, "scripts", "e0")
for _p in (os.path.join(QWEN_ROOT, "VLMEvalKit"), QWEN_ROOT, DISC_DIR, E0_DIR,
           AMP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs",
                       "anchor_completion_validation")
os.makedirs(OUT_DIR, exist_ok=True)

DS_MAIN = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
DS_NONREG = ["ChartQA_TEST", "MMBench_DEV_EN_V11", "MMStar", "RealWorldQA",
             "POPE"]
DS_ALL = DS_MAIN + DS_NONREG
K = 256

# frozen statistics seeds (prereg + protocol §8)
BOOT_SEED = 20261002          # primary 5000-draw stream
N_BOOT_PRIMARY = 5000
N_BOOT_SENS = 20000           # sensitivity = same stream, longer (nested)

FRESH_SEED = 20261002         # protocol §6.3 cluster shuffle
FRESH_MIN_Q = 200

ARMS_FORMAL = ["BASE", "MAIN025"]
ARMS_ABLATION = ["MAIN100", "MAIN_SIM025"]
ARMS_CONDITIONAL = ["BOTH025"]


def arm_cfg(name: str) -> dict:
    if name == "BASE":
        return dict(kind="base", lam=0.0, scope="main", selector="b1", K=256)
    if name == "MAIN025":
        return dict(kind="uniform", lam=0.25, scope="main",
                    selector="b1", K=256)
    if name == "MAIN100":
        return dict(kind="uniform", lam=1.00, scope="main",
                    selector="b1", K=256)
    if name == "MAIN_SIM025":
        return dict(kind="sim", lam=0.25, tau=0.1, scope="main",
                    selector="b1", K=256)
    if name == "BOTH025":     # conditional prereg fallback only
        return dict(kind="uniform", lam=0.25, scope="both",
                    selector="b1", K=256)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# bank (full-split frozen anchors; lean record: keep/gid/gsize/n_vis)
# ---------------------------------------------------------------------------
def bank_path(ds: str) -> str:
    return os.path.join(OUT_DIR, f"bank_full_{ds}.json.gz")


def load_bank(ds: str) -> dict:
    p = bank_path(ds)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with gzip.open(p, "rt") as f:
        return json.load(f)


def save_bank(ds: str, bank: dict) -> str:
    tmp = bank_path(ds) + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(bank, f)
    os.replace(tmp, bank_path(ds))
    return bank_sha256(ds)


def bank_sha256(ds: str) -> str:
    with open(bank_path(ds), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# generation shards (per panel/arm/ds), resumable
# ---------------------------------------------------------------------------
def shard_path(panel: str, arm: str, ds: str) -> str:
    d = os.path.join(OUT_DIR, "acc", panel, arm, f"K{K}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ds}.json")


def load_shard(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"records": {}}


def save_shard(path: str, shard: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=1)
    os.replace(tmp, path)


def shard_meta(panel: str, arm: str, ds: str, base_commit: str,
               max_new_tokens: int) -> dict:
    return dict(panel=panel, arm=arm, ds=ds, K=K, selector="b1",
                bank_sha256=bank_sha256(ds),
                base_commit=base_commit,
                max_new_tokens=max_new_tokens,
                protocol="anchor_completion_validation")


def degeneracy(text: str) -> dict:
    t = text.strip()
    toks = t.split()
    rep = 0
    if len(toks) >= 8:
        grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)]
        rep = max((grams.count(g) for g in set(grams)), default=0)
    return dict(empty=len(t) == 0, n_chars=len(t), repeat4=rep)


def repo_commit() -> str:
    import subprocess
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=QWEN_ROOT,
        text=True).strip()[:12]


def load_engine(max_new_tokens: int = 2048):
    model = common.load_model(common.BASELINE_MODEL,
                              max_new_tokens=max_new_tokens)
    from model.native_qwen3 import NativeEngine
    return NativeEngine(model)


# reuse the pilot's frozen math (uniform/sim merge, assignment, group_sum,
# official facility) — import, never copy-modify
from amp_common import (  # noqa: E402,F401
    compute_assignment,
    group_sum,
    merge_stream,
    official_facility_keep,
    run_one,
)


# ---------------------------------------------------------------------------
# per-image assignment (frozen rule: 多图不可跨图分组)
#
# amp_common.compute_assignment does a GLOBAL cos-argmax over all anchors;
# every sample in this round's panels is single-image, where per-image ==
# global bitwise.  For hypothetical multi-image inputs the frozen spec
# demands that a dropped token only be assigned to anchors of ITS OWN image,
# so the acu run/bank/gate paths use this wrapper.  Single-image behaviour
# is unchanged (same slices, same FP32 math).
# ---------------------------------------------------------------------------
def split_sizes_from_gthw(prep, spatial_merge_size: int):
    gthw = prep["gthw"]
    return (gthw.prod(-1) // (spatial_merge_size ** 2)).tolist()


def compute_assignment_per_image(V: torch.Tensor, keep: torch.Tensor,
                                 split_sizes):
    """Per-image cos-argmax assignment.  Returns (dropped_idx, gid, sim)
    with gid = GLOBAL anchor ranks into `keep`; dropped tokens of image i
    can only map to anchors inside image i."""
    dropped_all, gid_all, sim_all = [], [], []
    tok_offset = 0
    rank_offset = 0
    have_sim = None
    for n in split_sizes:
        lo, hi = tok_offset, tok_offset + n
        local_keep = keep[(keep >= lo) & (keep < hi)] - lo
        di, g, sim = compute_assignment(V[lo:hi], local_keep)
        dropped_all.append(di + lo)
        gid_all.append(g + rank_offset)
        if sim is not None:
            have_sim = True
            sim_all.append(sim)
        rank_offset += int(local_keep.numel())
        tok_offset = hi
    if not dropped_all:
        e = torch.empty(0, dtype=torch.long, device=V.device)
        return e, e, None
    dropped = torch.cat(dropped_all)
    gid = torch.cat(gid_all)
    sim_out = torch.cat(sim_all, dim=0) if have_sim else None
    return dropped, gid, sim_out


@torch.no_grad()
def run_one(eng, msg, ds, bank_rec, cfg: dict, max_new_tokens: int = 2048,
            ignore_eos: bool = False, timings: Optional[dict] = None,
            diag: Optional[dict] = None,
            deepstack: bool = True, pos: str = "mrope3d"):
    """acu variant of amp_common.run_one — the ONLY delta is that the
    assignment uses compute_assignment_per_image (frozen multi-image rule);
    for the single-image samples of every panel in this round the two are
    bitwise identical (verified by gate G3).

    ``deepstack``/``pos`` pass straight through to ``eng.prefill``; defaults
    keep the native path bitwise unchanged.  ``deepstack=False, pos='1d'``
    is the official EADP legacy config (DeepStack off, 1-D positions) —
    engine flags verified bit-exact against archived official predictions
    by e0 gate N5 and s1_dualpath."""
    timings = timings if timings is not None else {}
    K_ = int(cfg.get("K", 256))
    wall0 = time.perf_counter()
    prep = eng.prepare(msg, ds)
    timings["image_preprocess_ms"] = (time.perf_counter() - wall0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    V, DS = eng.encode(prep)
    ev[1].record()
    torch.cuda.synchronize()
    timings["vision_ms"] = ev[0].elapsed_time(ev[1])

    n_vis = prep["n_vis"]
    keep = torch.as_tensor(bank_rec["keep"], dtype=torch.long, device=V.device)
    if n_vis <= K_:                       # degenerate: keep everything
        keep = torch.arange(n_vis, device=V.device)

    kind, lam = cfg["kind"], cfg.get("lam", 0.0)
    scope = cfg.get("scope", "both")
    V_sel = DS_sel = None
    perm = None
    if kind not in ("base",):
        t0 = time.perf_counter()
        split_sizes = split_sizes_from_gthw(prep, eng.inner.visual.spatial_merge_size)
        dropped_idx, gid, sim = compute_assignment_per_image(V, keep,
                                                             split_sizes)
        if kind == "shuf":
            raise KeyError("shuf controls are not part of this round")
        if scope in ("both", "main"):
            y = merge_stream(V, keep, dropped_idx, gid, kind, lam,
                             cfg.get("tau", 0.1), sim)
            V_sel = y
        if scope in ("both", "ds"):
            ys = [merge_stream(ds_, keep, dropped_idx, gid, kind, lam,
                               cfg.get("tau", 0.1), sim) for ds_ in DS]
            DS_sel = ys
        torch.cuda.synchronize()
        timings["merge_ms"] = (time.perf_counter() - t0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    st = eng.prefill(prep, V, DS, keep, V_sel=V_sel, DS_sel=DS_sel,
                     deepstack=deepstack, pos=pos)
    ev[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = ev[0].elapsed_time(ev[1])
    timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3

    wall1 = time.perf_counter()
    gen_ids, text = eng.decode(st, max_new_tokens, ignore_eos=ignore_eos)
    timings["decode_wall_ms"] = (time.perf_counter() - wall1) * 1e3
    torch.cuda.synchronize()

    meta = eng.invariants(prep, st, V, DS, n_decode=len(gen_ids))
    if diag is not None:
        diag["merge_ms"] = timings.get("merge_ms")
    return dict(text=text, gen_ids=gen_ids, keep_idx=st.keep_idx, meta=meta,
                timings=timings, state=st)
