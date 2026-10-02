"""Stage-1 Round-trip Textual Guidance Pilot (RTG) — shared plumbing.

Protocol: docs/stage1_roundtrip_pilot_protocol.md (frozen commit 3f39aa3,
BEFORE any new-arm downstream score existed).

RTG replaces ONLY the EADP entropy-filter + textual-weighting stage with a
round-trip text weight:
  P      = softmax_vis(100 * A)          (A = local_sim_all 2-D slice,
                                          ORIGINAL negated-cosine symbols)
  B      = row-normalised P (via logP - logsumexp over text)
  c_t    = sum_i P_it * B_it             (round-trip probability of text t)
  w      = c / c.sum()
  local  = sum_t w_t * A_it
  fused  = 0.5 * g + 0.5 * local         (official alpha=0.5)
then the OFFICIAL post-processing verbatim: min-max (original eps) ->
3x3 sigma=1 Gaussian smoothing on the original grid -> beta=2.0.
No entropy filter, no softmax(-H/.01), no transform of c.
FLAT: w = 1/L.  SHUF: w computed from per-column-shuffled A, applied to
the real A (seeded per 20261002|ds|qid|img|t, local CPU generator).

Stage 2 is the OFFICIAL facility on the OFFICIAL main-path similarity with
the RTG importance as coverage weight (every scorer re-selects anchors).
Completion = MAIN025 exactly as frozen (cos-argmax assignment, uniform
group mean incl. anchor, y = a + 0.25*(mean(G)-a), main stream only).

LoFTR-inspired (ideas only): arXiv:2104.00680 ;
github.com/zju3dv/LoFTR (coarse_matching.py).  Bidirectional matching,
normalisation and the chi-square identity are standard constructions —
NOT claimed as inventions.
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
import torch.nn.functional as F

# -- path bootstrap ----------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(os.path.dirname(_HERE))
ACV_DIR = os.path.join(QWEN_ROOT, "scripts", "anchor_completion_validation")
XSP_DIR = os.path.join(QWEN_ROOT, "scripts", "stage1_cross_stream_pilot")
AMP_DIR = os.path.join(QWEN_ROOT, "scripts", "anchor_merge_pilot")
DISC_DIR = os.path.join(QWEN_ROOT, "scripts", "discovery")
E0_DIR = os.path.join(QWEN_ROOT, "scripts", "e0")
for _p in (os.path.join(QWEN_ROOT, "VLMEvalKit"), QWEN_ROOT, DISC_DIR, E0_DIR,
           XSP_DIR, ACV_DIR, AMP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402
import amp_common as AC  # noqa: E402  (facility/assignment/merge frozen math)

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "stage1_roundtrip_pilot")
os.makedirs(OUT_DIR, exist_ok=True)

DS_LIST = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
K = 256
INV_TEMP = 100.0                # frozen: original EADP visual-dim temperature
SEED_PREFIX = "20261002"        # frozen SHUF seed prefix

SCORERS = ["rtg", "flat", "shuf"]
ARMS = {
    "R_GATHER":  dict(scorer="rtg",  lam=0.0,  scope=None),
    "R_MAIN025": dict(scorer="rtg",  lam=0.25, scope="main"),
    "F_MAIN025": dict(scorer="flat", lam=0.25, scope="main"),
    "S_MAIN025": dict(scorer="shuf", lam=0.25, scope="main"),
}
NEW_ARMS = list(ARMS.keys())
REUSED_ARMS = ["E_GATHER", "E_MAIN025"]

BOOT_SEED = 20261002
N_BOOT = 20000
GPU_BUDGET_H = 2.0

_XSP_OUT = os.path.join(common.QWEN_ROOT, "outputs",
                        "stage1_cross_stream_pilot")


def git_commit() -> str:
    import subprocess
    return subprocess.check_output(
        ["git", "-C", os.path.abspath(os.path.join(QWEN_ROOT, "..")),
         "rev-parse", "HEAD"], text=True).strip()[:12]


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ===========================================================================
# RTG text weighting (protocol §2-§3) — per image, FP32, no_grad
# ===========================================================================
def shuf_seed(ds: str, qid, img_idx: int, t: int) -> int:
    h = hashlib.sha256(
        f"{SEED_PREFIX}|{ds}|{qid}|{img_idx}|{t}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 63)


def per_column_shuffle(A: torch.Tensor, ds: str, qid, img_idx: int = 0):
    """A_shuf[:, t] = A[perm_t, t] with an independent frozen permutation
    per text column.  Values within each column are preserved as a set
    (sort/entropy invariant); cross-column spatial pairing is destroyed."""
    N, L = A.shape
    out = A.clone()
    for t in range(L):
        g = torch.Generator(device="cpu")
        g.manual_seed(shuf_seed(ds, qid, img_idx, t))
        perm = torch.randperm(N, generator=g)
        out[:, t] = A[perm.to(A.device), t]
    return out


def text_weight(A: torch.Tensor, mode: str, ds: str = "", qid=None,
                img_idx: int = 0):
    """A [N, L] fp32 (original negated-cosine symbols).  Returns w [L]
    summing to 1, computed from A (rtg/flat) or from the per-column-shuffled
    A_shuf (shuf).  The CALLER applies w to the real A (protocol §3)."""
    N, L = A.shape
    if mode == "flat":
        return torch.full((L,), 1.0 / L, dtype=torch.float32,
                          device=A.device), A
    if mode == "shuf":
        A_eff = per_column_shuffle(A, ds, qid, img_idx)
    elif mode == "rtg":
        A_eff = A
    else:
        raise KeyError(mode)
    logP = torch.log_softmax(INV_TEMP * A_eff, dim=0)          # P(i|t)
    logB = logP - torch.logsumexp(logP, dim=1, keepdim=True)   # B(t|i)
    c = torch.exp(logP + logB).sum(dim=0)                      # [L]
    w = c / c.sum()
    return w, A_eff


@torch.no_grad()
def rtg_importance(feats: torch.Tensor, te: torch.Tensor, ts: torch.Tensor,
                   grid_h: int, grid_w: int, mode: str, ds: str = "",
                   qid=None, img_idx: int = 0):
    """Official EADP scoring pipeline with the entropy+textual stage
    replaced; everything after `fused` is VERBATIM official
    (eadp_stage1.stage1_importance / model.pruner ops, FP32)."""
    from model.pruner import _spatial_smoothing_impl

    ie = feats.float()
    tef = te.float()
    tsf = ts.float()
    ie = ie / ie.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tef = tef / tef.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tsf = tsf / tsf.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    global_sim = -torch.einsum("bnc,mc->bnm", ie, tef)          # [1,N,1]
    local_sim_all = -torch.einsum("bnc,mlc->bnml", ie, tsf)     # [1,N,1,L]
    A = local_sim_all[0, :, 0, :]                               # [N, L]
    g = global_sim[0, :, 0]                                     # [N]
    w, _ = text_weight(A, mode, ds, qid, img_idx)
    local = (A * w.unsqueeze(0)).sum(dim=1)                     # [N]
    fused = 0.5 * g + 0.5 * local                               # official alpha

    # --- verbatim official post-processing (same ops, same order) ---------
    text_sim = fused.unsqueeze(0)                               # [1, N]
    importance = (text_sim - text_sim.min(dim=-1, keepdim=True).values
                  + 1e-6) / (text_sim.max(dim=-1, keepdim=True).values
                             - text_sim.min(dim=-1, keepdim=True).values
                             + 1e-6)
    importance = _spatial_smoothing_impl(importance, grid_h, grid_w,
                                         kernel_size=3, sigma=1.0)
    importance = importance ** 2.0                              # beta = 2.0
    return importance, w.detach()


# ===========================================================================
# official facility with an RTG-family importance (per image)
# ===========================================================================
_EADP_CACHE = {}


def _get_pruner(V, eng):
    key = id(eng)
    pr = _EADP_CACHE.get(key)
    if pr is None:
        import instrumented as _inst
        dim = V.shape[-1]
        pr = _inst.TimedEADPPruner(visual_token_num=K, alpha=0.5, beta=2.0,
                                   visual_dim=dim,
                                   spatial_merge_size=
                                   eng.inner.visual.spatial_merge_size)
        pr.eval()
        _EADP_CACHE[key] = pr
    return pr


@torch.no_grad()
def select_keep(scorer: str, ctx: dict, K_: int, want_diag: bool = False):
    """Official facility selection for an RTG-family scorer.  ctx needs
    prep/V/DS/engine/text_mean/text_seq and the sample identity for SHUF."""
    eng, prep, V = ctx["engine"], ctx["prep"], ctx["V"]
    import instrumented
    from model.pruner import _sim_visual_impl
    sms = eng.inner.visual.spatial_merge_size
    split_sizes = (prep["gthw"].prod(-1) // (sms ** 2)).tolist()
    pruner = _get_pruner(V, eng)
    keep, offset, diags = [], 0, []
    for i, n in enumerate(split_sizes):
        token_num = min(K_, n)
        if token_num >= n:
            keep.append(torch.arange(offset, offset + n, device=V.device))
            offset += n
            continue
        feats = V[offset:offset + n].unsqueeze(0)
        sim01 = _sim_visual_impl(feats)                   # official matrix
        gh = int(prep["gthw"][i, 1]) // sms
        gw = int(prep["gthw"][i, 2]) // sms
        qid = ctx.get("qid")
        imp, w = rtg_importance(feats, ctx["text_mean"][i:i + 1],
                                ctx["text_seq"][i:i + 1], gh, gw, scorer,
                                ds=ctx.get("ds", ""), qid=qid,
                                img_idx=i)
        sel, _ = instrumented.SELECTORS["facility"](imp, sim01, token_num)
        idx = sel[0].sort().values
        keep.append(offset + idx.to(V.device))
        offset += n
        if want_diag:
            eff = float(1.0 / (w ** 2).sum().item())
            diags.append(dict(eff_tokens=eff, L=int(w.numel())))
    out = torch.cat(keep)
    return (out, diags) if want_diag else (out, None)


# ===========================================================================
# banks
# ===========================================================================
def bank_path(scorer: str, ds: str) -> str:
    return os.path.join(OUT_DIR, f"bank_dev_{scorer}_{ds}.json.gz")


def load_bank(scorer: str, ds: str) -> dict:
    with gzip.open(bank_path(scorer, ds), "rt") as f:
        return json.load(f)


def bank_config(scorer: str, ds: str) -> dict:
    return dict(scorer=scorer, split="dev", ds=ds, K=K,
                params=dict(inverse_temperature=INV_TEMP,
                            alpha=0.5, beta=2.0, smoothing="3x3 sigma=1",
                            minmax_eps=1e-6, shuf_seed_prefix=SEED_PREFIX),
                manifest_sha256=sha256_file(
                    os.path.join(AC.OUT_DIR, "manifest.json")),
                base_model=AC.common.BASELINE_MODEL,
                code_commit=git_commit())


def verify_bank_meta(bank: dict, scorer: str, ds: str) -> None:
    want = bank_config(scorer, ds)
    got = bank.get("meta", {})
    for k in ("scorer", "split", "ds", "K", "params", "manifest_sha256",
              "base_model", "code_commit"):
        if got.get(k) != want[k]:
            raise RuntimeError(
                f"bank meta mismatch {scorer}/{ds}: {k}: "
                f"disk={got.get(k)!r} want={want[k]!r}")


@torch.no_grad()
def build_scorer_bank(scorer: str, ds: str, limit: int | None = None,
                      model=None, eng=None):
    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    if model is None:
        model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                     max_new_tokens=64)
        from model.native_qwen3 import NativeEngine
        eng = NativeEngine(model)
    dataset = AC.common.build_dataset(ds)
    model.set_dump_image(dataset.dump_image)
    items = manifest["datasets"][ds]["dev"]
    if limit:
        items = items[:limit]
    out_path = bank_path(scorer, ds)
    bank = {"meta": bank_config(scorer, ds), "samples": {}}
    if os.path.exists(out_path):
        with gzip.open(out_path, "rt") as f:
            bank = json.load(f)
        verify_bank_meta(bank, scorer, ds)
    t0 = time.time()
    for n, it in enumerate(items):
        key = str(it["idx"])
        if key in bank["samples"]:
            continue
        row = dataset.data.iloc[it["idx"]]
        msg = AC.common.build_message(model, dataset, ds, row)
        prep = eng.prepare(msg, ds)
        V, DS = eng.encode(prep)
        if prep["n_vis"] <= K:
            bank["samples"][key] = dict(
                idx=int(it["idx"]), image_key=it["image_key"],
                keep=list(range(prep["n_vis"])), gid=[], gsize=[],
                w_diag=dict(eff_tokens=float(prep["n_vis"]),
                            L=int(prep["n_vis"])))
            continue
        text_mean, text_seq = eng.instruction_embeds(msg, ds)
        ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=eng,
                   text_mean=text_mean, text_seq=text_seq,
                   ds=ds, qid=int(it["idx"]))
        keep, diags = select_keep(scorer, ctx, K, want_diag=True)
        dropped_idx, gid, _ = AC.compute_assignment(V, keep)
        gid = gid.cpu()
        counts = torch.bincount(gid, minlength=int(keep.numel()))
        bank["samples"][key] = dict(
            idx=int(it["idx"]), image_key=it["image_key"],
            keep=keep.cpu().tolist(), gid=gid.tolist(),
            gsize=counts.tolist(), w_diag=diags[0])
        if (n + 1) % 20 == 0 or n + 1 == len(items):
            el = time.time() - t0
            print(f"[bank dev {scorer} {ds}] {n+1}/{len(items)} "
                  f"({el/(n+1):.2f}s/q)", flush=True)
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(bank, f)
    os.replace(tmp, out_path)
    print(f"[saved] {out_path} ({len(bank['samples'])} records, "
          f"sha {sha256_file(out_path)[:12]})", flush=True)
    return out_path, model, eng


# ===========================================================================
# one-sample runner (bank-based; frozen MAIN025 completion math via acu)
# ===========================================================================
def arm_cfg(arm: str) -> dict:
    a = ARMS[arm]
    if a["lam"] == 0.0:
        return dict(kind="base", lam=0.0, scope="main", scorer=a["scorer"])
    return dict(kind="uniform", lam=a["lam"], scope="main",
                scorer=a["scorer"])


def shard_path(arm: str, ds: str) -> str:
    d = os.path.join(OUT_DIR, "acc", arm)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ds}.json")


def load_shard(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"records": {}}


def save_shard(path: str, shard: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=1)
    os.replace(tmp, path)
