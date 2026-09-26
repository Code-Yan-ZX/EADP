"""
M5 (Safe Removal) step 2 -- THE EARLY GATE.

The brief's §3: before any student is trained and before a single generation is
run, answer one question.

    Is "this retained token can be removed safely" more predictable from
    deployable pre-LLM features than "this dropped token is the critical miss"?

Four stages of this project (S2-C2, S3-B, M3-v0, M3-v2) measured the second
question and closed it.  If the first is materially easier, Safe Removal has a
motivation and the line continues; if it is not, the line stops here and the
cost of finding out is one afternoon of NLL forwards.

Labels
------
    d_i = L(y | S0 \\ {i}) - L(y | S0)          teacher-forced gold NLL delta

The brief's §4 asks for a SELECTIVE problem, not a regression: we do not need
to predict d_i precisely, we need to be confident about a few tokens.

    SAFE       d_i <= +eps      deleting it is not resolvably harmful
    HARMFUL    d_i >= +delta    deleting it clearly costs
    IGNORE     eps < d_i < delta

`eps` and `delta` are fixed from the noise floor the teacher measures on every
instance, never from an accuracy number:

    eps   = the pooled median per-instance floor.  "The deletion moves the
            answer loss by less than the same measurement moves when nothing
            about the set changes."
    delta = the pooled p90 floor.  A conservative bar for the harmful class,
            set by the measurement's own resolution rather than by taste.

Sampling and weighting
----------------------
The teacher's sample is stratified on purpose (the tail strata are enriched),
so every population-level number is reported TWICE:

    unweighted   over the measured tokens -- "can the probe rank the tails"
    weighted     inverse-inclusion weights, estimating the population of all
                 256 tokens of S0 -- the deployable question, since at inference
                 the probe is applied to every retained token.

Both are printed.  Neither is allowed to be reported alone.

Usage
    python scripts/discovery/m5_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m5_common import (FEATURES, IMP_COL, N_FEAT, RECON_COL,        # noqa: E402
                       RED_COL, SafeRisk, load_m5_bank, vis_row)
from m5_teacher import STRATA                                       # noqa: E402

KS = (4, 8, 12, 16)
PROBE_KINDS = ("lr", "mlp", "vis")


# ===========================================================================
# metrics -- all weighted-capable, all written here so nothing is implicit
# ===========================================================================
def w_auroc(score: np.ndarray, pos: np.ndarray, w: np.ndarray) -> float:
    """Weighted AUROC, computed as the weighted Mann-Whitney concordance.

        AUC = sum_{i in pos, j in neg} w_i w_j [1(s_i > s_j) + 0.5*1(s_i = s_j)]
              / (W_pos * W_neg)

    The unweighted rank shortcut (sum of positive ranks minus n_pos(n_pos+1)/2,
    over n_pos*n_neg) does NOT generalise to weights: it normalises by the
    unweighted counts while the numerator carries weighted mass, and it returns
    ~0.05 for a random score on the inverse-inclusion weights this file uses.
    The random baseline is what caught it.  Sorting once and carrying a running
    sum of negative weight gives the exact quantity in O(n log n).
    """
    pos = pos.astype(bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    s, p, ww = score[order], pos[order], w[order].astype(np.float64)
    wneg = np.where(p, 0.0, ww)
    # weight of negatives strictly below, and exactly equal to, each element
    below = np.concatenate([[0.0], np.cumsum(wneg)[:-1]])
    eq = np.zeros(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        eq[i:j + 1] = wneg[i:j + 1].sum()
        i = j + 1
    num = float((ww * p * (below + 0.5 * eq)).sum())
    den = float((ww * p).sum() * wneg.sum())
    return num / den if den > 0 else float("nan")


def w_auprc(score: np.ndarray, pos: np.ndarray, w: np.ndarray) -> float:
    """Weighted average precision (step-wise, tie-aware on the score)."""
    if pos.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    p, ww = pos[order], w[order]
    tp = np.cumsum(ww * p)
    fp = np.cumsum(ww * (~p))
    prec = tp / np.clip(tp + fp, 1e-12, None)
    drec = np.diff(np.concatenate([[0.0], tp / tp[-1]]))
    return float((prec * drec).sum())


def w_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    def rank(a):
        o = np.argsort(a, kind="stable")
        r = np.empty(len(a))
        r[o] = np.arange(len(a), dtype=float)
        return r
    rx, ry = rank(x), rank(y)
    mx = np.average(rx, weights=w)
    my = np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    sx = np.sqrt(np.average((rx - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((ry - my) ** 2, weights=w))
    return float(cov / max(sx * sy, 1e-12))


def bottom_k_metrics(risk: np.ndarray, d: np.ndarray, k: int,
                     eps: float, delta: float, kid: np.ndarray = None) -> dict:
    """What the method actually does: delete the k lowest-risk tokens.

    **Per instance.**  At inference the probe ranks the 256 tokens of ONE image's
    `S0` and deletes its own k; it never sees another image's tokens.  Pooling
    the split and taking a global bottom-k would measure a different and much
    easier decision -- "find the k safest tokens anywhere in 150 images" -- and
    would flatter every arm.  With `kid` given, the metrics are computed inside
    each instance and averaged over instances.
    """
    if kid is not None:
        per = [bottom_k_metrics(risk[kid == g], d[kid == g], k, eps, delta)
               for g in np.unique(kid) if (kid == g).sum() >= k]
        if not per:
            return dict(k=int(k), n_instances=0, safe_precision=float("nan"),
                        harm_rate=float("nan"), mean_d=float("nan"),
                        worst_d=float("nan"))
        return dict(k=int(k), n_instances=len(per),
                    safe_precision=float(np.mean([r["safe_precision"]
                                                  for r in per])),
                    harm_rate=float(np.mean([r["harm_rate"] for r in per])),
                    mean_d=float(np.mean([r["mean_d"] for r in per])),
                    worst_d=float(np.mean([r["worst_d"] for r in per])),
                    tokens=None)
    order = np.argsort(risk, kind="stable")
    sel = order[:k]
    return dict(k=int(k),
                safe_precision=float(np.mean(d[sel] <= eps)),
                harm_rate=float(np.mean(d[sel] >= delta)),
                mean_d=float(np.mean(d[sel])),
                worst_d=float(np.max(d[sel])),
                tokens=[int(t) for t in sel])


# ===========================================================================
# sample table
# ===========================================================================
N_MC = 8000          # Monte-Carlo draws used to recover the inclusion weights
                     # (validated against the teacher's own sampler: the
                     # vectorised reconstruction matches it to |dp| < 0.03 at
                     # 6000 draws, which is Monte-Carlo error and nothing else)


def inclusion_weights(X: torch.Tensor, s0: np.ndarray, rng_seed: int,
                      n_mc: int = N_MC) -> np.ndarray:
    """Exact-in-expectation P(token of S0 was measured), by simulating the sampler.

    The teacher's `sample_tokens` draws the `rand` stratum FIRST and then walks
    each tail stratum in score order, skipping tokens `rand` already took.  The
    strata are therefore NOT independent, and the naive inclusion probability
    `k/|S0|` is wrong for the tokens that sit just past each tail's quota: a
    token ranked 7th by redundancy is measured whenever `rand` happened to take
    one of the top 6, which is ~44 %, not 9.4 %.  Weighting by `k/|S0|` would
    then over-weight exactly the tokens the rules are about.

    Simulating the sampler is exact to Monte-Carlo error and cannot drift from
    the teacher, because it re-runs the teacher's own code path on the teacher's
    own score columns.  It is cheap: a few numpy ops over 256 elements per draw.
    """
    s0l = np.asarray(s0)
    n = s0l.size
    rng = np.random.default_rng(rng_seed)
    # A uniform `rand` quota per draw.  Any construction of a uniform 24-subset
    # gives the same distribution as the teacher's `rng.choice`, which is what
    # the weight depends on -- not the bit-stream of its generator.
    keys = rng.random((n_mc, n))
    rand_idx = np.argpartition(keys, STRATA[0][1], axis=1)[:, :STRATA[0][1]]
    in_rand = np.zeros((n_mc, n), dtype=bool)
    np.put_along_axis(in_rand, rand_idx, True, axis=1)

    # The tail strata are NOT disjoint -- a token can be both high-redundancy
    # and high-importance -- so each stratum sees the tokens every EARLIER
    # stratum already took, exactly as the teacher's `take()` does.  Getting
    # this wrong over-weights precisely the tokens the rules are about.
    used = in_rand.copy()
    for name, k in STRATA[1:]:
        order = _stratum_order(X, s0l, name)            # positions into s0
        free = ~used[:, order]                          # (n_mc, n)
        taken = free & (np.cumsum(free, axis=1) <= k)   # the first k unused
        scat = np.zeros((n_mc, n), dtype=bool)
        np.put_along_axis(scat, np.broadcast_to(order, taken.shape),
                          taken, axis=1)
        used |= scat
    return used.mean(axis=0)


def _stratum_order(X: torch.Tensor, s0l: np.ndarray, name: str) -> np.ndarray:
    """The teacher's within-stratum order, as POSITIONS into s0."""
    if name == "maxred":
        col, rev = RED_COL, True
    elif name == "minred":
        col, rev = RED_COL, False
    elif name == "highimp":
        col, rev = IMP_COL, True
    elif name == "lowimp":
        col, rev = IMP_COL, False
    elif name == "recon":
        col, rev = RECON_COL, False
    else:
        raise KeyError(name)
    vals = X[:, col].numpy()[s0l]
    return np.argsort(-vals if rev else vals, kind="stable")


def build_samples(bank: dict, teacher: dict, split: str):
    """(X, d, key_id, token, stratum, weight) for one split.

    The weight is the inverse of the Monte-Carlo inclusion probability of that
    token in that instance, so a weighted statistic estimates the same quantity
    over the full 256-token S0 population that the probe would face at
    inference.
    """
    idx = bank["_index"]
    rows = [k for k in teacher["done_keys"]
            if str(bank["split"][idx[k]]) == split]
    Xs, ds, kids, toks, strata, ws = [], [], [], [], [], []
    for r, k in enumerate(rows):
        i = idx[k]
        m = teacher["meas"][k]
        s0 = np.asarray(m["s0"], dtype=np.int64)
        pos = {int(t): j for j, t in enumerate(s0)}
        # `inclusion_weights` re-runs the teacher's sampler, which indexes the
        # FULL (1024, F) matrix by token id -- so it must be handed the full
        # matrix, not the S0 slice.
        Xfull = torch.from_numpy(bank["X"][i].astype(np.float32))
        p = inclusion_weights(Xfull, s0, rng_seed=zlib.crc32(k.encode()))
        for s in m["singles"]:
            t = int(s["token"])
            Xs.append(bank["X"][i][t])
            ds.append(float(s["d"]))
            kids.append(r)
            toks.append(t)
            strata.append(s["stratum"])
            ws.append(1.0 / max(float(p[pos[t]]), 1.0 / (4 * N_MC)))
    X = np.asarray(Xs, dtype=np.float32)
    d = np.asarray(ds, dtype=np.float64)
    strata = np.asarray(strata)
    kid = np.asarray(kids)
    w = np.asarray(ws, dtype=np.float64)
    return dict(X=X, d=d, kid=kid, token=np.asarray(toks), stratum=strata,
                w=w, keys=rows, bank=bank, idx=idx)


def stack_vis(sample, bank):
    """(N, D) vision features for the sampled (instance, token) pairs.

    The samples are ordered by instance, so only one instance's (1024, 4096)
    row is held at a time: caching all 300 would be 5 GB for a 265 MB result.
    """
    out = np.zeros((len(sample["d"]), 4096), dtype=np.float32)
    cur, cur_r = None, -1
    for j in range(len(sample["d"])):
        r = int(sample["kid"][j])
        if r != cur_r:
            i = sample["idx"][sample["keys"][r]]
            cur = vis_row(bank, i).float().numpy()
            cur_r = r
        out[j] = cur[sample["token"][j]]
    return out


# ===========================================================================
# the probe fit
# ===========================================================================
def fit_probe(kind, Xtr, Vtr, ytr, Xva, Vva, yva, seed, epochs=300, lr=3e-3,
              kid_tr=None, es_frac=0.2, patience=10, every=10):
    """Fit one probe on the fit split; return val risks and the fitted model.

    Two things this is careful about.

    **Grad is enabled explicitly.**  Importing `scoring` (and through it
    vlmeval) leaves the process with autograd globally disabled, so a probe fit
    that trusted the ambient state would silently train nothing at all and
    still report a loss.

    **Early stopping splits by INSTANCE, not by token.**  The `vis` family has
    ~262k parameters and reaches a train loss of 0.016 unaided, so without a
    stopping rule it memorises the fit set and the val number measures the
    stopping rule rather than the features.  The holdout is drawn from the FIT
    instances only -- val is never touched -- and tokens of one instance never
    straddle the split, because tokens within an instance are strongly
    correlated and a token-level split would leak the instance's own answer.
    All three families get the same rule, so the comparison between them stays
    like-for-like.
    """
    with torch.enable_grad():
        torch.manual_seed(seed)
        mu = Xtr.mean(0)
        sd = Xtr.std(0).clamp_min(1e-6)
        Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd
        lossf = torch.nn.BCEWithLogitsLoss()

        n = Xtr.shape[0]
        assert kid_tr is not None, "the instance id of every fit token is needed"
        n_inst = int(kid_tr.max()) + 1
        g = torch.Generator().manual_seed(seed)
        es_inst = set(torch.randperm(n_inst, generator=g)[
            :max(1, int(es_frac * n_inst))].tolist())
        es = torch.tensor([int(k) in es_inst for k in kid_tr.tolist()])
        tr = ~es
        if int(tr.sum()) < 32 or int(es.sum()) < 32:
            tr = torch.ones(n, dtype=torch.bool)
            es = torch.zeros(n, dtype=torch.bool)

        def batch(sel, V):
            return (Xtr_n[sel], (V[sel] if V is not None else None),
                    ytr[sel])

        Xa, Va, ya = batch(tr, Vtr)
        Xe, Ve, ye = batch(es, Vtr)
        model = SafeRisk(kind=kind, d_hand=N_FEAT, hidden=1)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        best, best_state, best_ep, bad = float("inf"), None, 0, 0
        for ep in range(epochs):
            opt.zero_grad()
            lossf(model(Xa, Va), ya).backward()
            opt.step()
            if es.sum() > 0 and (ep + 1) % every == 0:
                with torch.no_grad():
                    v = float(lossf(model(Xe, Ve), ye).item())
                if v < best - 1e-5:
                    best, best_ep, bad = v, ep + 1, 0
                    best_state = {k: t.detach().clone()
                                  for k, t in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        with torch.no_grad():
            final = float(lossf(model(Xtr_n, Vtr if kind == "vis" else None),
                                ytr).item())
    with torch.no_grad():
        rva = model.risk(Xva_n, Vva if kind == "vis" else None).numpy()
        rtr = model.risk(Xtr_n, Vtr if kind == "vis" else None).numpy()
    return dict(model=model, mu=mu, sd=sd, risk_va=rva, risk_tr=rtr,
                train_loss=final, es_loss=best, es_epoch=best_ep)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="m5_teacher")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="m5_probe")
    args = ap.parse_args()

    teacher = json.load(open(os.path.join(OUTPUT_DIR, f"{args.teacher}.json")))
    bank = load_m5_bank()
    print(f"[data] teacher has {len(teacher['done_keys'])} instances")

    fit = build_samples(bank, teacher, "fit")
    val = build_samples(bank, teacher, "val")
    d_all = np.concatenate([fit["d"], val["d"]])
    floors = np.array([teacher["meas"][k]["floor"] for k in teacher["done_keys"]])

    # ---- thresholds, fixed from the noise floor (brief §4) -----------------
    eps = float(np.median(floors))
    delta = float(np.percentile(floors, 90))
    rep = dict(teacher=args.teacher, n_fit=len(fit["d"]), n_val=len(val["d"]),
               thresholds=dict(eps=eps, delta=delta,
                               floor_median=float(np.median(floors)),
                               floor_p90=float(np.percentile(floors, 90)),
                               floor_max=float(floors.max()),
                               rule="eps = median per-instance floor; "
                                    "delta = p90 per-instance floor; both "
                                    "from train/val NLL resolution only"),
               d_stats=dict(p10=float(np.percentile(d_all, 10)),
                            p25=float(np.percentile(d_all, 25)),
                            p50=float(np.percentile(d_all, 50)),
                            p75=float(np.percentile(d_all, 75)),
                            p90=float(np.percentile(d_all, 90)),
                            frac_safe=float(np.mean(d_all <= eps)),
                            frac_harm=float(np.mean(d_all >= delta)),
                            frac_ignore=float(np.mean((d_all > eps)
                                                      & (d_all < delta)))),
               strata={s: int((fit["stratum"] == s).sum()) for s, _ in STRATA})
    print(f"[thr] eps={eps:.4f} delta={delta:.4f}  "
          f"safe={rep['d_stats']['frac_safe']:.3f} "
          f"harm={rep['d_stats']['frac_harm']:.3f} "
          f"ignore={rep['d_stats']['frac_ignore']:.3f}")

    # ---- the three label definitions, all reported -------------------------
    LABELS = dict(
        hurts=lambda d: d > 0,
        harmful=lambda d: d >= delta,
        safe=lambda d: d <= eps,
    )

    Xtr = torch.from_numpy(fit["X"])
    Xva = torch.from_numpy(val["X"])
    Vtr = torch.from_numpy(stack_vis(fit, bank)) if True else None
    Vva = torch.from_numpy(stack_vis(val, bank)) if True else None

    # ---- baselines: the training-free rules, on the SAME tokens ------------
    # Every arm is a RISK score: high means "deleting this one is likely to
    # hurt".  The rules are therefore oriented -- a redundant token is LOW risk
    # and a high-importance token is HIGH risk -- so that all seven arms can be
    # read on the same axes and the same bottom-k decision.
    def rule_risk(name, X):
        if name == "random":
            return np.random.default_rng(0).random(X.shape[0])
        if name == "maxred":                      # high redundancy -> safe
            return -X[:, RED_COL]
        if name == "lowimp":                      # high importance -> risky
            return X[:, IMP_COL]
        if name == "recon":                       # well reconstructed -> safe
            return X[:, RECON_COL]
        raise KeyError(name)

    base_rows = []
    for name in ("random", "maxred", "lowimp", "recon"):
        for split, smp in (("fit", fit), ("val", val)):
            base_rows.append(_score_row(f"rule:{name}", split,
                                        rule_risk(name, smp["X"]), smp, LABELS,
                                        eps, delta))
    rep["rules"] = base_rows

    # ---- the probes --------------------------------------------------------
    # `kind` is the probe family, `seed` the fit seed.  The deployable arm is
    # `mlp` seed 0 (pre-registered: the brief asks for a small MLP, and it sits
    # between the linear probe and the vision-view probe in capacity); `lr`
    # seed 0 is carried alongside so the accuracy grid can price the capacity.
    # Two targets, because the brief's §4 selective framing and its §5 decision
    # rule are not quite the same problem.  `harmful` (d >= delta) is the
    # literal "risk = P(removal harms the answer)" and is the PRE-REGISTERED
    # primary.  `notsafe` (d > eps) is the complement of the class the method
    # acts on -- a token is deleted precisely when it is predicted safe -- and
    # is carried so the gate can show whether the two disagree.  Both are
    # reported; the accuracy grid runs the primary.
    TARGETS = dict(harmful=lambda d: (d >= delta).astype(np.float32),
                   notsafe=lambda d: (d > eps).astype(np.float32))
    probe_rows, saved = [], {}
    kid_tr = torch.from_numpy(fit["kid"])
    for tname, tfn in TARGETS.items():
        ytr_t = torch.from_numpy(tfn(fit["d"]))
        print(f"[target] {tname}: fit positives {float(ytr_t.mean()):.3f}",
              flush=True)
        for kind in PROBE_KINDS:
            for seed in range(args.seeds):
                f = fit_probe(kind, Xtr, Vtr, ytr_t, Xva, Vva, None, seed,
                              kid_tr=kid_tr)
                if seed == 0 and tname == "harmful":
                    saved[f"{kind}|s{seed}"] = dict(
                        state_dict=f["model"].state_dict(), mu=f["mu"],
                        sd=f["sd"], kind=kind, hidden=1, d_hand=N_FEAT,
                        target=f"d >= delta ({delta:.4f})",
                        fitted_on=f"fit split, n={len(fit['d'])}")
                for split, smp, risk in (("fit", fit, f["risk_tr"]),
                                         ("val", val, f["risk_va"])):
                    probe_rows.append(_score_row(
                        f"probe:{kind}", split, risk, smp, LABELS, eps, delta,
                        seed=seed, extra=dict(train_loss=f["train_loss"],
                                              target=tname)))
                print(f"[probe] {tname} {kind} seed{seed} done "
                      f"(loss {f['train_loss']:.4f})", flush=True)
    rep["probes"] = probe_rows
    torch.save(saved, os.path.join(OUTPUT_DIR, "m5_probe.pt"))
    print(f"[saved] m5_probe.pt  keys={sorted(saved)}")

    # ---- the formulation comparison (brief §7) -----------------------------
    rep["head_comparison"] = head_comparison(bank, teacher, fit, val, eps, delta)

    # ---- the aggregation the verdict reads ---------------------------------
    rep["summary"] = summarize(rep)
    from m2_gdep import dump_json
    dump_json(f"{args.tag}.json", rep)


def _score_row(name, split, score, smp, LABELS, eps, delta, seed=None,
               extra=None):
    d, w = smp["d"], smp["w"]
    row = dict(arm=name, split=split, seed=seed, n=int(len(d)),
               auprc_harmful=w_auprc(score, LABELS["harmful"](d), w),
               auprc_safe=w_auprc(-score, LABELS["safe"](d), w),
               spearman_d=w_spearman(score, d, w))
    for lname, fn in LABELS.items():
        row[f"auroc_{lname}"] = w_auroc(score, fn(d), w)
    # The same statistics computed INSIDE each instance and averaged.  The
    # probe ranks one image's 256 tokens at a time, so this is the protocol the
    # method actually runs under; the pooled numbers above answer the different
    # question of how the score behaves over the whole population.
    per_a, per_s, per_ap = [], [], []
    for g_ in np.unique(smp["kid"]):
        sel = smp["kid"] == g_
        if sel.sum() < 8:
            continue
        per_a.append(w_auroc(score[sel], LABELS["harmful"](d[sel]),
                             np.ones(int(sel.sum()))))
        per_ap.append(w_auprc(score[sel], LABELS["harmful"](d[sel]),
                              np.ones(int(sel.sum()))))
        per_s.append(w_spearman(score[sel], d[sel], np.ones(int(sel.sum()))))
    row["per_instance"] = dict(
        n_instances=len(per_a),
        auroc_harmful=float(np.nanmean(per_a)) if per_a else float("nan"),
        auprc_harmful=float(np.nanmean(per_ap)) if per_ap else float("nan"),
        spearman_d=float(np.nanmean(per_s)) if per_s else float("nan"))
    row["bottom"] = [bottom_k_metrics(score, d, k, eps, delta, kid=smp["kid"])
                     for k in KS]
    if extra:
        row.update(extra)
    return row


def head_comparison(bank, teacher, fit, val, eps, delta):
    """The critical-head formulation, on the SAME features and probe family.

    Label: a DROPPED token is a "head" token if the frozen gradient teacher
    ranks it inside the top-K of all 1024.  This is the target S2-C2, S3-B,
    M3-v0 and M3-v2 spent four stages failing to predict, and the comparison is
    exact because the features, the probe family, the target-fitting procedure
    and the split are all identical to the safe-removal run above -- the only
    thing that differs is which tokens are in the population and what the label
    means.

    It calls `fit_probe`, not a private copy of it, so the early-stopping rule
    cannot drift between the two formulations and flatter one of them.
    """
    out = dict(pos_k=[64, 256], probe="mlp (same as the safe-removal primary)",
               note="same features, same probe, same fit procedure, same split; "
                    "population is the 768 DROPPED tokens, not S0")
    for K in (64, 256):
        rows = []
        for split, smp in (("fit", fit), ("val", val)):
            Xs, ys, kids, oracle = [], [], [], []
            for r, k in enumerate(smp["keys"]):
                i = smp["idx"][k]
                g = np.asarray(bank["g2"][i], dtype=np.float64)
                s0 = set(int(t) for t in bank["s0"][i])
                head = set(np.argsort(-g, kind="stable")[:K].tolist())
                dropped = np.array([t for t in range(len(g)) if t not in s0])
                # M3-v0's oracle: the teacher's own top-16 WITHIN the dropped
                # set.  overlap@16 against this is the 0.236 figure.
                orc = set(dropped[np.argsort(-g[dropped], kind="stable")[:16]].tolist())
                for t in dropped:
                    Xs.append(bank["X"][i][t])
                    ys.append(1.0 if t in head else 0.0)
                    kids.append(r)
                    oracle.append(1.0 if t in orc else 0.0)
            rows.append((split,
                         torch.from_numpy(np.asarray(Xs, dtype=np.float32)),
                         torch.from_numpy(np.asarray(ys, dtype=np.float32)),
                         torch.from_numpy(np.asarray(kids)),
                         np.asarray(oracle)))
        if not len(rows[0][1]) or not len(rows[1][1]):
            out[f"head_K{K}"] = dict(checked=False, reason="a split is empty")
            continue
        Xtr, ytr, ktr = rows[0][1], rows[0][2], rows[0][3]
        f = fit_probe("mlp", Xtr, None, ytr, Xtr, None, None, 0, kid_tr=ktr)
        for split, X, y, kid, orc in rows:
            with torch.no_grad():
                s = f["model"].risk((X - f["mu"]) / f["sd"]).numpy()
            yy = y.numpy().astype(bool)
            rec = dict(split=split, K=K, n=int(len(yy)),
                       pos_rate=float(yy.mean()),
                       auroc=w_auroc(s, yy, np.ones(len(yy))),
                       auprc=w_auprc(s, yy, np.ones(len(yy))))
            # PER INSTANCE, which is the decision the method actually makes and
            # the protocol M3-v0 used.  A pooled top-16 over 46 080 tokens from
            # 60 images is a different, much easier question.
            p16, ov16 = [], []
            for g_ in np.unique(kid):
                sel = kid == g_
                top = np.argsort(-s[sel], kind="stable")[:16]
                p16.append(yy[sel][top].mean())
                ov16.append(orc[sel][top].sum() / 16.0)
            rec["precision_at_16_per_instance"] = float(np.mean(p16))
            rec["overlap_at_16_per_instance"] = float(np.mean(ov16))
            rec["chance_overlap_at_16"] = float(16.0 / 768.0)
            out[f"head_K{K}_{split}"] = rec
            print(f"[head] K={K} {split}: AUROC {rec['auroc']:.3f} "
                  f"AUPRC {rec['auprc']:.3f} (base {rec['pos_rate']:.3f})  "
                  f"precision@16 {rec['precision_at_16_per_instance']:.3f}  "
                  f"overlap@16 {rec['overlap_at_16_per_instance']:.3f} "
                  f"(chance {rec['chance_overlap_at_16']:.4f})", flush=True)
    return out


def summarize(rep):
    """The numbers the verdict reads, aggregated over seeds."""
    def agg(rows, name, split, key):
        v = [r[key] for r in rows if r["arm"] == name and r["split"] == split]
        v = [x for x in v if x == x]
        return (float(np.mean(v)) if v else float("nan"),
                float(np.std(v)) if v else float("nan"))
    out = dict(by_split={})
    for split in ("fit", "val"):
        block = {}
        for arm in ("rule:random", "rule:maxred", "rule:lowimp", "rule:recon",
                    "probe:lr", "probe:mlp", "probe:vis"):
            m, s = agg(rep["probes"] + rep["rules"], arm, split, "auroc_harmful")
            sp, _ = agg(rep["probes"] + rep["rules"], arm, split, "spearman_d")
            block[arm] = dict(auroc_harmful_mean=m, auroc_harmful_sd=s,
                              spearman_d_mean=sp)
        out["by_split"][split] = block
    return out


if __name__ == "__main__":
    main()
