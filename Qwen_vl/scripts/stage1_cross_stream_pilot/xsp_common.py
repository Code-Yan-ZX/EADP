"""Stage-1 Cross-Stream Redundancy Pilot — shared plumbing.

Protocol: docs/stage1_cross_stream_pilot_protocol.md (frozen before any
accuracy/latency number of this round was produced).

Hypothesis: whether a token's main-path similar neighbours remain substitutable
in the OTHER (DeepStack) streams is a forward-only proxy for token utility.
`cross_stream` Stage-1 scores each token by the mean over the three DS streams
of its kNN residual (m=8 main-space neighbours), floor 0.1, and feeds that as
the coverage weight to the OFFICIAL facility-location selector (original
similarity matrix, original greedy, original tie-breaking).  No EADP
importance, no entropy filtering / min-max / smoothing / polarization.

Stage 2 is frozen but anchors are NOT: every scorer re-selects its own anchors
and re-groups.  Completion (for the *MAIN025 arms): cos-argmax assignment on
RAW main features, uniform group mean incl. anchor,
y_a = v_a + 0.25*(mean(G_a) - v_a), MAIN stream only; the three DS streams
stay DS[s][keep].
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

# -- path bootstrap; reuse the anchor-merge pilot plumbing verbatim ----------
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
import amp_common as AC  # noqa: E402  (official facility, assignment, merge)

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "stage1_cross_stream_pilot")
os.makedirs(OUT_DIR, exist_ok=True)

DS_LIST = list(AC.DS_LIST)          # TextVQA_VAL, DocVQA_VAL, OCRBench
K = 256
SPLIT = "dev"

# frozen Stage-1 constants (protocol §3)
M_NB = 8
EPS = 1e-8
FLOOR = 0.1
SCORERS = ["eadp", "cross_stream", "uniform", "main_residual"]

# frozen arm registry (protocol §4)
ARMS = {
    "E_GATHER":  dict(scorer="eadp",         lam=0.0,  scope=None),
    "E_MAIN025": dict(scorer="eadp",         lam=0.25, scope="main"),
    "X_GATHER":  dict(scorer="cross_stream", lam=0.0,  scope=None),
    "X_MAIN025": dict(scorer="cross_stream", lam=0.25, scope="main"),
    "U_MAIN025": dict(scorer="uniform",      lam=0.25, scope="main"),
    "V_MAIN025": dict(scorer="main_residual", lam=0.25, scope="main"),
}
PERF_ARMS = ["E_GATHER", "E_MAIN025", "X_GATHER", "X_MAIN025"]

BOOT_SEED = 20261002                # protocol §6 (frozen)
N_BOOT = 20000
MARGIN = 0.5                        # macro points, non-inferiority screening


_REPO_ROOT = os.path.abspath(os.path.join(QWEN_ROOT, ".."))


def git_commit() -> str:
    if hasattr(AC.common, "git_commit"):
        return AC.common.git_commit()
    import subprocess
    return subprocess.check_output(
        ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
        text=True).strip()


def env_meta() -> dict:
    import torch as _t
    import transformers
    return dict(python=sys.version.split()[0], torch=_t.__version__,
                transformers=transformers.__version__,
                cuda=_t.version.cuda, gpu=_t.cuda.get_device_name(0),
                base_model=AC.common.BASELINE_MODEL)


# ===========================================================================
# Stage-1 scoring (protocol §3) — per image, FP32, no_grad
# ===========================================================================
def l2_rows(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation with the frozen eps floor."""
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(EPS)


def neighbor_index(sim: torch.Tensor, m: int = M_NB) -> torch.Tensor:
    """sim [N,N] fp32 (the OFFICIAL facility matrix, a monotone transform of
    cosine — identical neighbour ORDER).  Self is excluded by masking the
    diagonal of an INDEPENDENT COPY (the caller's matrix is never touched).
    Ties -> smaller original index first (stable descending sort).
    Returns nbr [N, m] with m = min(m, N-1)."""
    N = sim.shape[0]
    m = min(m, N - 1)
    s = sim.clone()
    s.fill_diagonal_(float("-inf"))
    order = torch.sort(s, dim=1, descending=True, stable=True).indices
    return order[:, :m]


def knn_residual(H: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
    """d[i] = || H[i] - mean(H[nbr[i]]) ||^2 (mean NOT re-normalised)."""
    mu = H[nbr].mean(dim=1)
    return ((H - mu) ** 2).sum(dim=1)


@torch.no_grad()
def stage1_importance(scorer: str, V: torch.Tensor, DS_list, sim: torch.Tensor,
                      m: int = M_NB, eps: float = EPS, floor: float = FLOOR):
    """Per-image Stage-1 importance.  V [N,D], DS_list = list of [N,D] (three
    DeepStack streams), sim [N,N] fp32 official main-path similarity matrix.

    Returns (w [N] fp32, diag dict).  Diag records zero-norm row counts (rows
    whose PRE-normalisation norm <= eps) and per-stream mean residuals.
    """
    N = V.shape[0]
    diag = dict(scorer=scorer, m=min(m, N - 1), zero_norm=[], d_mean=[])
    assert torch.isfinite(sim).all(), "sim matrix has non-finite entries"
    if scorer == "uniform":
        return torch.ones(N, dtype=torch.float32, device=V.device), diag
    if scorer not in ("cross_stream", "main_residual"):
        raise KeyError(scorer)
    nbr = neighbor_index(sim, m)
    if scorer == "main_residual":
        streams = [V]
    else:
        streams = list(DS_list)
        assert len(streams) == 3, "expected all three DeepStack streams"
    w = torch.full((N,), floor, dtype=torch.float32, device=V.device)
    for st in streams:
        nrm = st.float().norm(dim=-1)
        diag["zero_norm"].append(int((nrm <= eps).sum().item()))
        H = l2_rows(st)
        d = knn_residual(H, nbr)
        dm = float(d.mean().item())
        diag["d_mean"].append(dm)
        w = w + d / (dm + eps)          # z_s = d_s / (mean(d_s)+eps)
    assert torch.isfinite(w).all() and bool((w >= 0).all())
    return w, diag


# ===========================================================================
# official facility with an arbitrary Stage-1 weight (per image loop)
# ===========================================================================
_EADP_CACHE = {}


def _get_pruner(V, eng) -> "instrumented.TimedEADPPruner":
    """Same construction as e0_selectors._eadp_parts (alpha=0.5, beta=2.0)."""
    key = id(eng)
    pr = _EADP_CACHE.get(key)
    if pr is None:
        import instrumented as _inst
        dim = V.shape[-1]
        pr = _inst.TimedEADPPruner(visual_token_num=K, alpha=0.5, beta=2.0,
                                   visual_dim=dim,
                                   spatial_merge_size=eng.inner.visual.spatial_merge_size)
        pr.eval()
        _EADP_CACHE[key] = pr
    return pr


@torch.no_grad()
def select_keep(scorer: str, ctx: dict, K_: int, want_diag: bool = False):
    """keep indices (ascending, duplicate-free) for any scorer.

    'eadp' goes through the OFFICIAL assembly e0_selectors._eadp_parts
    (bit-identical to the b1 path of rounds 1-2).  New scorers reuse the
    official similarity matrix + official facility greedy, feeding their own
    w.  N<=K per image -> keep whole image, no scoring (protocol §3.7).
    """
    eng, prep, V, DS = ctx["engine"], ctx["prep"], ctx["V"], ctx["DS"]
    if scorer == "eadp":
        return AC.official_facility_keep(ctx, K_), None
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
        sim01 = _sim_visual_impl(feats)                  # official matrix
        DS_img = [d[offset:offset + n] for d in DS]
        w, diag = stage1_importance(scorer, V[offset:offset + n], DS_img,
                                    sim01[0])
        sel, _ = instrumented.SELECTORS["facility"](w.unsqueeze(0), sim01,
                                                    token_num)
        idx = sel[0].sort().values
        keep.append(offset + idx.to(V.device))
        offset += n
        if want_diag:
            diags.append(diag)
    out = torch.cat(keep)
    return (out, diags) if want_diag else (out, None)


# ===========================================================================
# banks (protocol §5)
# ===========================================================================
def bank_path(scorer: str, ds: str, split: str = SPLIT) -> str:
    return os.path.join(OUT_DIR, f"bank_{split}_{scorer}_{ds}.json.gz")


def legacy_bank_path(ds: str) -> str:
    return os.path.join(AC.OUT_DIR, f"bank_dev_{ds}.json.gz")


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_bank(scorer: str, ds: str, split: str = SPLIT) -> dict:
    with gzip.open(bank_path(scorer, ds, split), "rt") as f:
        return json.load(f)


def load_legacy_bank(ds: str) -> dict:
    with gzip.open(legacy_bank_path(ds), "rt") as f:
        return json.load(f)


def bank_config(scorer: str, ds: str, split: str = SPLIT) -> dict:
    return dict(scorer=scorer, split=split, ds=ds, K=K,
                params=dict(m=M_NB, eps=EPS, floor=FLOOR, alpha=0.5, beta=2.0)
                if scorer != "uniform" else dict(m=M_NB, eps=EPS, floor=FLOOR),
                manifest_sha256=sha256_file(os.path.join(AC.OUT_DIR,
                                                         "manifest.json")),
                base_model=AC.common.BASELINE_MODEL,
                code_commit=git_commit(),
                env=env_meta())


def verify_bank_meta(bank: dict, scorer: str, ds: str, split: str = SPLIT) -> None:
    """Strict resume check: the on-disk meta must match the frozen config."""
    want = bank_config(scorer, ds, split)
    got = bank.get("meta", {})
    for k in ("scorer", "split", "ds", "K", "params", "manifest_sha256",
              "base_model", "code_commit"):
        if got.get(k) != want[k]:
            raise RuntimeError(
                f"bank meta mismatch for {scorer}/{ds}: {k}: "
                f"disk={got.get(k)!r} want={want[k]!r}")


def build_record(it: dict, keep: torch.Tensor, dropped_idx, gid,
                 V: torch.Tensor, DS, w, diag) -> dict:
    """One bank record: keep/gid/gsize + Stage-1 diagnostics."""
    Kd = int(keep.numel())
    rec = dict(idx=int(it["idx"]), image_key=it["image_key"],
               keep=keep.cpu().tolist(),
               gid=gid.cpu().tolist() if gid is not None else [])
    counts = torch.bincount(gid.cpu(), minlength=Kd) if gid is not None \
        else torch.zeros(Kd, dtype=torch.long)
    rec["gsize"] = counts.tolist()
    if w is not None:
        rec["w"] = [round(float(x), 6) for x in w.cpu()]
    if diag is not None:
        rec["zero_norm"] = diag["zero_norm"]
        rec["d_mean"] = [round(x, 8) for x in diag["d_mean"]]
    return rec


@torch.no_grad()
def build_scorer_bank(scorer: str, ds: str, split: str = SPLIT,
                      limit: int | None = None,
                      model=None, eng=None) -> str:
    """Freeze keep/gid + diagnostics for one (scorer, dataset).  Returns path."""
    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    if model is None:
        model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                     max_new_tokens=64)
        from model.native_qwen3 import NativeEngine
        eng = NativeEngine(model)
    dataset = AC.common.build_dataset(ds)
    model.set_dump_image(dataset.dump_image)
    items = manifest["datasets"][ds][split]
    if limit:
        items = items[:limit]
    out_path = bank_path(scorer, ds, split)
    bank = {"meta": bank_config(scorer, ds, split), "samples": {}}
    if os.path.exists(out_path):
        with gzip.open(out_path, "rt") as f:
            bank = json.load(f)
        verify_bank_meta(bank, scorer, ds, split)     # strict resume check
    t0 = time.time()
    for n, it in enumerate(items):
        key = str(it["idx"])
        if key in bank["samples"]:
            continue
        row = dataset.data.iloc[it["idx"]]
        msg = AC.common.build_message(model, dataset, ds, row)
        prep = eng.prepare(msg, ds)
        V, DS = eng.encode(prep)
        if prep["n_vis"] <= K:                        # degenerate keep-all
            rec = dict(idx=int(it["idx"]), image_key=it["image_key"],
                       keep=list(range(prep["n_vis"])), gid=[], gsize=[],
                       w=[1.0] * prep["n_vis"], zero_norm=[], d_mean=[])
            bank["samples"][key] = rec
            continue
        ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=eng)
        if scorer == "eadp":
            # official importance diagnostics (selection itself below uses
            # the official assembly verbatim)
            from common import CudaTimer
            sms = eng.inner.visual.spatial_merge_size
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            ctx["text_mean"], ctx["text_seq"] = text_mean, text_seq
            ws = []
            offset = 0
            split_sizes = (prep["gthw"].prod(-1) // (sms ** 2)).tolist()
            for i, n_img in enumerate(split_sizes):
                if min(K, n_img) >= n_img:
                    offset += n_img
                    continue
                gh = int(prep["gthw"][i, 1]) // sms
                gw = int(prep["gthw"][i, 2]) // sms
                imp = _get_pruner(V, eng)._score(
                    V[offset:offset + n_img].unsqueeze(0),
                    text_mean[i:i + 1], text_seq[i:i + 1], gh, gw,
                    CudaTimer())
                ws.append(imp[0].detach().float().cpu())
                offset += n_img
            w_cat = torch.cat(ws) if ws else None
            diag = None
        else:
            # diagnostics-only recompute of w for the bank record (selection
            # itself goes through select_keep, the gate-verified path)
            from model.pruner import _sim_visual_impl
            sms = eng.inner.visual.spatial_merge_size
            offset = 0
            split_sizes = (prep["gthw"].prod(-1) // (sms ** 2)).tolist()
            ws, zero_acc, dmean_acc = [], [], []
            for i, n_img in enumerate(split_sizes):
                if min(K, n_img) >= n_img:
                    offset += n_img
                    continue
                sim01 = _sim_visual_impl(
                    V[offset:offset + n_img].unsqueeze(0))
                w_img, diag_i = stage1_importance(
                    scorer, V[offset:offset + n_img],
                    [d[offset:offset + n_img] for d in DS], sim01[0])
                ws.append(w_img.detach().float().cpu())
                for j, z in enumerate(diag_i["zero_norm"]):
                    zero_acc.append(z)
                for j, dm in enumerate(diag_i["d_mean"]):
                    dmean_acc.append(dm)
                offset += n_img
            w_cat = torch.cat(ws) if ws else torch.ones(prep["n_vis"])
            diag = dict(scorer=scorer, zero_norm=zero_acc,
                        d_mean=dmean_acc)
        keep, _ = select_keep(scorer, ctx, K)
        dropped_idx, gid, _ = AC.compute_assignment(V, keep)
        rec = build_record(it, keep, dropped_idx, gid, V, DS, w_cat, diag)
        bank["samples"][key] = rec
        if (n + 1) % 20 == 0 or n + 1 == len(items):
            el = time.time() - t0
            print(f"[bank {split} {scorer} {ds}] {n + 1}/{len(items)} "
                  f"({el / (n + 1):.2f}s/q)", flush=True)
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(bank, f)
    os.replace(tmp, out_path)
    print(f"[saved] {out_path} ({len(bank['samples'])} records, "
          f"sha {sha256_file(out_path)[:12]})", flush=True)
    return out_path, model, eng


def bank_rec(bank: dict, idx) -> dict:
    return bank["samples"][str(idx)]


# ===========================================================================
# one-sample runners
# ===========================================================================
def arm_run_cfg(arm: str) -> dict:
    """amp_common.run_one-compatible cfg for a bank-based generation."""
    a = ARMS[arm]
    if a["lam"] == 0.0:
        return dict(kind="base", lam=0.0, scorer=a["scorer"], arm=arm)
    assert a["scope"] == "main"
    return dict(kind="uniform", lam=a["lam"], scope="main",
                scorer=a["scorer"], arm=arm)


@torch.no_grad()
def run_one_live(eng, msg, ds, scorer: str, arm: str, max_new_tokens: int,
                 ignore_eos: bool = False, timings: dict | None = None,
                 lam: float | None = None):
    """FULLY live path (no bank): encode -> [sim] -> Stage-1 scoring ->
    official facility -> assignment/merge -> prefill -> decode.

    Used for the paired efficiency measurement; accuracy runs read the frozen
    banks.  Segment timings: image_preprocess, vision, text_embeds (eadp
    only), similarity, stage1, facility, completion, llm_prefill, ttft,
    decode_wall."""
    timings = timings if timings is not None else {}
    a = ARMS[arm]
    lam = a["lam"] if lam is None else lam
    wall0 = time.perf_counter()
    prep = eng.prepare(msg, ds)
    timings["image_preprocess_ms"] = (time.perf_counter() - wall0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    V, DS = eng.encode(prep)
    ev[1].record()
    torch.cuda.synchronize()
    timings["vision_ms"] = ev[0].elapsed_time(ev[1])

    K_ = K
    n_vis = prep["n_vis"]
    V_sel = None
    if n_vis <= K_:
        keep = torch.arange(n_vis, device=V.device)
    else:
        if scorer == "eadp":
            t0 = time.perf_counter()
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            torch.cuda.synchronize()
            timings["text_embeds_ms"] = (time.perf_counter() - t0) * 1e3
            ctx = dict(prep=prep, V=V, DS=DS, K=K_, engine=eng,
                       text_mean=text_mean, text_seq=text_seq)
            t0 = time.perf_counter()
            keep = AC.official_facility_keep(ctx, K_)
            torch.cuda.synchronize()
            # official path: scoring+similarity+facility combined (the
            # official assembly does not separate them; NOT double counted)
            timings["eadp_stage1_facility_ms"] = \
                (time.perf_counter() - t0) * 1e3
        else:
            from model.pruner import _sim_visual_impl
            import instrumented
            sms = eng.inner.visual.spatial_merge_size
            split_sizes = (prep["gthw"].prod(-1) // (sms ** 2)).tolist()
            pruner = _get_pruner(V, eng)
            keep_parts, offset = [], 0
            for i, n in enumerate(split_sizes):
                token_num = min(K_, n)
                if token_num >= n:
                    keep_parts.append(
                        torch.arange(offset, offset + n, device=V.device))
                    offset += n
                    continue
                feats = V[offset:offset + n].unsqueeze(0)
                t0 = time.perf_counter()
                sim01 = _sim_visual_impl(feats)
                torch.cuda.synchronize()
                timings["similarity_ms"] = timings.get("similarity_ms", 0.0) \
                    + (time.perf_counter() - t0) * 1e3
                t0 = time.perf_counter()
                w, _ = stage1_importance(scorer, V[offset:offset + n],
                                         [d[offset:offset + n] for d in DS],
                                         sim01[0])
                torch.cuda.synchronize()
                timings["stage1_ms"] = timings.get("stage1_ms", 0.0) \
                    + (time.perf_counter() - t0) * 1e3
                t0 = time.perf_counter()
                sel, _ = instrumented.SELECTORS["facility"](
                    w.unsqueeze(0), sim01, token_num)
                torch.cuda.synchronize()
                timings["facility_ms"] = timings.get("facility_ms", 0.0) \
                    + (time.perf_counter() - t0) * 1e3
                keep_parts.append(offset + sel[0].sort().values.to(V.device))
                offset += n
            keep = torch.cat(keep_parts)

    if lam > 0.0 and n_vis > K_:
        t0 = time.perf_counter()
        dropped_idx, gid, _ = AC.compute_assignment(V, keep)
        y = AC.merge_stream(V, keep, dropped_idx, gid, "uniform", lam)
        V_sel = y
        torch.cuda.synchronize()
        timings["completion_ms"] = (time.perf_counter() - t0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    st = eng.prefill(prep, V, DS, keep, V_sel=V_sel, DS_sel=None)
    ev[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = ev[0].elapsed_time(ev[1])
    timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3

    wall1 = time.perf_counter()
    gen_ids, text = eng.decode(st, max_new_tokens, ignore_eos=ignore_eos)
    timings["decode_wall_ms"] = (time.perf_counter() - wall1) * 1e3
    torch.cuda.synchronize()
    meta = eng.invariants(prep, st, V, DS, n_decode=len(gen_ids))
    if ignore_eos:
        assert len(gen_ids) == max_new_tokens, \
            f"fixed-length decode violated: {len(gen_ids)}"
    return dict(text=text, gen_ids=gen_ids, keep_idx=st.keep_idx, meta=meta,
                timings=timings, state=st)
