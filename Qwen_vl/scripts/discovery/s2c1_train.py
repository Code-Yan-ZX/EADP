"""
S2-C1 step 2: train the tiny learned scorers on the cached prefix features.

The LLM and the vision tower are frozen and are not even loaded -- the scorer
only ever sees the cached arrays from ``s2c1_features.py``. Nothing about the
Qwen3-VL weights is trained, and the teacher is never recomputed: the target
comes from the cached S2-B P1-G2 maps.

Two scorer families, both deliberately minimal:

  LIN_<L>  token-only linear probe        score_i = w.h_i + b
           -- is the teacher mostly detecting "information-bearing visual tokens"?

  QRY_<L>  query-conditioned low rank     score_i = u_i . u_q / sqrt(r) + w_b.h_i + c
           u_i = Wv h_i, u_q = Wq q, rank r = 64
           -- does question conditioning add anything over the token's own content?

Layers L in {2, 4}, 0-based (the S2-C0 convention). No hidden-size sweep, no
layer sweep, no loss sweep: one fixed configuration per family, as the brief
specifies. The feature standardization is one fixed preprocessing step (fit-split
mean/std), not a tuned hyperparameter.

Targets. Two teacher readings are available and both are computed:
  T1  y_i = 1 iff token i is in the teacher's Top-256   -> the primary loss
  T2  y_i = rank percentile of G2_i                     -> secondary, off by default
No MSE against raw G2: its scale is not stable across images, and S2-C0 showed
a pure rank-agreement objective selects the *least* useful layers.

Split is image-level throughout (fit 240 / val 60 / test 150). Validation is
used for early stopping and for choosing the layer; the held-out 150 is never
touched by any selection decision in this file.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import time_callable
from s1_audit import OUT

D_MODEL = 4096
N_VIS = 1024
N_POS = 256
N_NEG = N_VIS - N_POS
RANK_DIM = 64
MARGIN = 1.0
LAMBDA_RANK = 0.5
POS_WEIGHT = N_NEG / N_POS          # 3.0 -> balanced BCE
LR = 1e-3
WD = 1e-4
BATCH = 8
MAX_EPOCHS = 80
PATIENCE = 20


# ---------------------------------------------------------------------------
class TokenOnly(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.out = nn.Linear(d, 1)

    def forward(self, h, q=None):
        return self.out(h).squeeze(-1)


class QueryConditioned(nn.Module):
    def __init__(self, d=D_MODEL, rank=RANK_DIM):
        super().__init__()
        self.rank = rank
        self.Wv = nn.Linear(d, rank, bias=False)
        self.Wq = nn.Linear(d, rank, bias=False)
        self.bias = nn.Linear(d, 1)

    def forward(self, h, q):
        u = self.Wv(h)                                  # (B, N, r)
        uq = self.Wq(q)                                 # (B, r)
        bilinear = (u * uq.unsqueeze(1)).sum(-1) * self.rank ** -0.5
        return bilinear + self.bias(h).squeeze(-1)


FAMILIES = {"LIN": TokenOnly, "QRY": QueryConditioned}


# ---------------------------------------------------------------------------
def teacher_targets(T, keys):
    """T1 membership mask + the pos/neg index blocks, from the cached S2-B maps.

    The 15 S2-A causal cases sit outside the frozen bank and have no cached P1-G2
    map; they are placeholders here (zeros) and are never read -- only the
    fit/val/test rows are ever trained or scored against. The causal tier is
    evaluated against occlusion labels, not against the teacher.
    """
    y, pos_idx, neg_idx = [], [], []
    for k in keys:
        if k in T.files:
            order = np.argsort(-T[k].astype(np.float64), kind="stable")
        else:
            order = np.arange(N_VIS)                     # placeholder, unusable
        m = np.zeros(N_VIS, dtype=np.float32)
        m[order[:N_POS]] = 1.0
        y.append(m)
        pos_idx.append(order[:N_POS])
        neg_idx.append(order[N_POS:])
    return (np.stack(y), np.stack(pos_idx), np.stack(neg_idx))


def standardize_stats(H, rows):
    """Fit-split per-dimension mean/std, accumulated in float64 (no 4 GB blowup)."""
    d = H.shape[-1]
    s1 = np.zeros(d, dtype=np.float64)
    s2 = np.zeros(d, dtype=np.float64)
    n = 0
    for j in rows:
        x = H[j].astype(np.float32)
        s1 += x.sum(axis=0, dtype=np.float64)
        s2 += np.square(x, dtype=np.float64).sum(axis=0)
        n += x.shape[0]
    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 1e-12)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)


class FeatureSource:
    """Random access into the cached memmaps, with the fixed normalization.

    ``cache=True`` pulls the requested rows into RAM once. The fit split is read
    on every epoch, and re-reading 2.5 GB of memmap per epoch would be pure I/O
    waste; the eval splits are read a handful of times and stay on the memmap.
    """

    def __init__(self, L, rows, mu, sd, dev, cache=False):
        H = np.load(os.path.join(OUT, f"s2c1_feats_L{L}.npy"), mmap_mode="r")
        Q = np.load(os.path.join(OUT, f"s2c1_query_L{L}.npy"), mmap_mode="r")
        if cache:
            self.H = np.ascontiguousarray(H[rows])
            self.Q = np.ascontiguousarray(Q[rows])
            self.rows = np.arange(len(rows))
        else:
            self.H, self.Q, self.rows = H, Q, np.asarray(rows)
        self.mu = torch.from_numpy(mu).to(dev)
        self.sd = torch.from_numpy(sd).to(dev)
        self.dev = dev

    def batch(self, sel, need_q):
        """sel: positions into self.rows."""
        idx = self.rows[sel]
        h = torch.from_numpy(np.ascontiguousarray(self.H[idx])).to(self.dev).float()
        h = (h - self.mu) / self.sd
        if not need_q:
            return h, None
        q = torch.from_numpy(np.ascontiguousarray(self.Q[idx])).to(self.dev).float()
        q = (q - self.mu) / self.sd
        return h, q


# ---------------------------------------------------------------------------
def compute_loss(s, y, pos_idx, neg_idx, use_pct=False, pct=None):
    """balanced BCE + lambda * pairwise margin ranking (all pos/neg pairs)."""
    bce = F.binary_cross_entropy_with_logits(
        s, y, pos_weight=torch.tensor(POS_WEIGHT, device=s.device))
    sp = torch.gather(s, 1, torch.from_numpy(pos_idx).to(s.device))
    sn = torch.gather(s, 1, torch.from_numpy(neg_idx).to(s.device))
    rank = F.relu(MARGIN - sp.unsqueeze(-1) + sn.unsqueeze(1)).mean()
    total = bce + LAMBDA_RANK * rank
    if use_pct:
        z = (s - s.mean(dim=1, keepdim=True)) / (s.std(dim=1, keepdim=True) + 1e-6)
        total = total + F.smooth_l1_loss(z, torch.from_numpy(pct).to(s.device))
    return total, float(bce), float(rank)


@torch.no_grad()
def score_all(model, fs, need_q, bs=16):
    outs = []
    for a in range(0, len(fs.rows), bs):
        sel = list(range(a, min(a + bs, len(fs.rows))))
        h, q = fs.batch(sel, need_q)
        outs.append(model(h, q).float().cpu())
    return torch.cat(outs, 0).numpy()


def ranking_metrics(s, y):
    """AUROC / AP / Top-256 overlap / recall, all threshold-free."""
    order = np.argsort(-s, kind="stable")
    top = set(order[:N_POS].tolist())
    truth = set(np.nonzero(y > 0.5)[0].tolist())
    inter = len(top & truth)
    # rank-based AUROC (Mann-Whitney), no sklearn dependency
    r = np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(np.float64) + 1
    n_pos, n_neg = len(truth), len(y) - len(truth)
    auroc = (r[np.array(sorted(truth))].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # average precision
    hit = np.isin(order, np.array(sorted(truth)))
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    ap = float((prec * hit).sum() / n_pos)
    return dict(auroc=float(auroc), ap=ap, overlap256=inter / N_POS,
                recall256=inter / n_pos)


def evaluate_split(model, fs, y, need_q):
    s = score_all(model, fs, need_q)
    m = [ranking_metrics(s[i], y[i]) for i in range(len(s))]
    return s, {k: float(np.mean([x[k] for x in m])) for k in m[0]}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["LIN_L2", "LIN_L4", "QRY_L2", "QRY_L4"],
                    help="<FAMILY>_L<layer>, FAMILY in {LIN, QRY}")
    ap.add_argument("--loss", default="bce_rank", choices=["bce_rank", "bce_rank_pct"],
                    help="bce_rank_pct adds the T2 percentile SmoothL1 term (secondary)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank-dim", type=int, default=RANK_DIM)
    ap.add_argument("--tag", default="s2c1")
    ap.add_argument("--latency", action="store_true", default=True)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(OUT, f"{args.tag}_features.json")))
    plan = {p["key"]: p for p in meta["plan"]}
    keys = [p["key"] for p in meta["plan"]]
    rows_of = {s: [i for i, p in enumerate(meta["plan"]) if p["split"] == s]
               for s in ("fit", "val", "test", "causal")}
    print(f"[split] fit={len(rows_of['fit'])} val={len(rows_of['val'])} "
          f"test={len(rows_of['test'])} causal={len(rows_of['causal'])}")

    G = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    missing = [p["key"] for p in meta["plan"]
               if not p["causal"] and p["key"] not in G.files]
    assert not missing, f"frozen instances without a cached teacher map: {missing[:5]}"
    Y, P, Ng = teacher_targets(G, keys)
    PCT = np.stack([np.argsort(np.argsort(G[k].astype(np.float64))) / (N_VIS - 1)
                    if k in G.files else np.zeros(N_VIS, dtype=np.float64)
                    for k in keys]).astype(np.float32)
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results, score_bank, timings = {}, {}, {}
    for arm in args.arms:
        fam, ls = arm.split("_L")
        L = int(ls)
        t_start = time.time()
        H = np.load(os.path.join(OUT, f"{args.tag}_feats_L{L}.npy"), mmap_mode="r")
        mu, sd = standardize_stats(H, rows_of["fit"])
        del H
        fs_fit = FeatureSource(L, rows_of["fit"], mu, sd, dev, cache=True)
        fs_val = FeatureSource(L, rows_of["val"], mu, sd, dev, cache=True)
        fs_test = FeatureSource(L, rows_of["test"], mu, sd, dev)
        fs_cau = FeatureSource(L, rows_of["causal"], mu, sd, dev)
        need_q = fam == "QRY"

        model = FAMILIES[fam](D_MODEL, args.rank_dim) if fam == "QRY" else FAMILIES[fam](D_MODEL)
        model = model.to(dev)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

        y_fit, p_fit, n_fit = Y[rows_of["fit"]], P[rows_of["fit"]], Ng[rows_of["fit"]]
        y_val = Y[rows_of["val"]]
        n_fit_img = len(rows_of["fit"])
        rng = np.random.default_rng(args.seed)

        best = dict(val_overlap=-1.0, epoch=-1, state=None)
        hist = []
        for ep in range(MAX_EPOCHS):
            model.train()
            perm = rng.permutation(n_fit_img)
            ep_loss = []
            for a in range(0, n_fit_img, BATCH):
                sel = perm[a:a + BATCH]
                h, q = fs_fit.batch(sel, need_q)
                s = model(h, q)
                loss, bce, rk = compute_loss(
                    s, torch.from_numpy(y_fit[sel]).to(dev),
                    p_fit[sel], n_fit[sel],
                    use_pct=args.loss == "bce_rank_pct",
                    pct=PCT[rows_of["fit"]][sel] if args.loss == "bce_rank_pct" else None)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss.append(float(loss))
            model.eval()
            _, vm = evaluate_split(model, fs_val, y_val, need_q)
            hist.append(dict(epoch=ep, loss=float(np.mean(ep_loss)), **vm))
            if vm["overlap256"] > best["val_overlap"]:
                best = dict(val_overlap=vm["overlap256"], epoch=ep,
                            state={k: v.detach().clone() for k, v in model.state_dict().items()})
            if ep - best["epoch"] >= PATIENCE:
                break
        model.load_state_dict(best["state"])
        model.eval()
        train_s = time.time() - t_start

        # ---- held-out tier + the diagnostic tiers -------------------------
        _, val_m = evaluate_split(model, fs_val, y_val, need_q)
        _, test_m = evaluate_split(model, fs_test, Y[rows_of["test"]], need_q)
        _, cau_m = (evaluate_split(model, fs_cau, Y[rows_of["causal"]], need_q)
                    if len(rows_of["causal"]) else (None, None))
        _, train_m = evaluate_split(model, fs_fit, y_fit, need_q)

        # every instance, for the causal / agreement diagnostics and the
        # generation arms (which look scores up by '<arm>__<ds>_<idx>')
        s_all = score_all(model, FeatureSource(
            L, list(range(len(keys))), mu, sd, dev), need_q)
        for k, v in zip(keys, s_all):
            score_bank[f"{arm}__{k}"] = v.astype(np.float32)

        lat = {}
        if args.latency:
            fs_one = FeatureSource(L, rows_of["test"][:1], mu, sd, dev)
            h1, q1 = fs_one.batch([0], need_q)
            with torch.no_grad():
                sc_mean, sc_std = time_callable(lambda: model(h1, q1), repeat=30, warmup=5)
                tk_mean, tk_std = time_callable(
                    lambda: torch.topk(model(h1, q1), N_POS, dim=-1), repeat=30, warmup=5)
            lat = dict(scorer_ms=sc_mean, scorer_ms_std=sc_std,
                       topk_ms=tk_mean, topk_ms_std=tk_std)
            prefix = meta["latency"].get(str(L), {})
            if prefix:
                lat["prefix_ms"] = prefix["prefix_ms_mean"]
                lat["scoring_only_ms"] = prefix["prefix_ms_mean"] + sc_mean + tk_mean

        results[arm] = dict(
            family=fam, layer=L, seed=args.seed, loss=args.loss,
            rank_dim=args.rank_dim if fam == "QRY" else None,
            n_params=int(n_params), train_seconds=float(train_s),
            best_epoch=int(best["epoch"]), epochs_run=len(hist),
            train=train_m, val=val_m, test=test_m, causal=cau_m,
            val_history=hist, latency=lat,
        )
        timings[arm] = train_s
        print(f"\n[{arm}] params={n_params}  best_epoch={best['epoch']}  "
              f"train_s={train_s:.1f}")
        print(f"    fit  overlap256={train_m['overlap256']:.3f} auroc={train_m['auroc']:.3f} "
              f"ap={train_m['ap']:.3f}")
        print(f"    val  overlap256={val_m['overlap256']:.3f} auroc={val_m['auroc']:.3f} "
              f"ap={val_m['ap']:.3f} recall={val_m['recall256']:.3f}")
        print(f"    test overlap256={test_m['overlap256']:.3f} auroc={test_m['auroc']:.3f} "
              f"ap={test_m['ap']:.3f} recall={test_m['recall256']:.3f}")
        if cau_m:
            print(f"    causal AUROC-target overlap256={cau_m['overlap256']:.3f}")
        if lat:
            print(f"    latency: scorer={lat['scorer_ms']:.2f} ms topk={lat['topk_ms']:.2f} ms"
                  + (f" prefix={lat['prefix_ms']:.1f} ms" if "prefix_ms" in lat else ""))
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(OUT, f"{args.tag}_{arm}.pt"))
        del fs_fit, fs_val, fs_test, fs_cau, model
        torch.cuda.empty_cache()

    np.savez_compressed(os.path.join(OUT, f"{args.tag}_scores.npz"), **score_bank)
    json.dump({"config": vars(args), "results": results,
               "split_counts": {k: len(v) for k, v in rows_of.items()},
               "keys": keys},
              open(os.path.join(OUT, f"{args.tag}_train.json"), "w"), indent=1)
    print(f"\n[saved] {args.tag}_scores.npz ({len(score_bank)} vectors) "
          f"+ {args.tag}_train.json")


if __name__ == "__main__":
    main()
