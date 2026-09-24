"""
M3-v2 step 2 -- fit the query-conditioned HEAD auditor.

Two axes, both frozen before any held-out number exists (brief §4, §5, §11):

    objective   h1  balanced BCE, positives = teacher ranks 0..15 of the
                    dropped set, negatives = ranks 16..127 (hard negatives);
                    ranks >= 128 are ignored, never sampled
                h2  pairwise logistic ranking over the same two sets, with a
                    1/sqrt(rank+1) weight on the positive -- the teacher's own
                    head is ordered, not its tail
                h3  h1 + h2
    structure   A   token-local          (v0's shape, new objective)
                B   query-conditioned    (visual queries -> instruction tokens)
                C   query + retained-set context (learned slots over S0)

Selection is on `val`, on the deployment metric: **Top-16 recall@16** -- the
share of the teacher's own 16 best dropped tokens the auditor would rescue --
tie-broken by the head-weighted rank metric `hw@16`.  Not token AUC: AUC over
768 dropped tokens is dominated by the easy middle of the ranking, which is
exactly the regime v0 died in (mean rescue rank 74-132).  `test` (the held-out
150) is scored once per checkpoint and reported as `test_posthoc`; it never
chooses anything.

Usage
    python scripts/discovery/m3v2_train.py --configs h1:A h2:A h3:A
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m3v2_common import (N_NEG_SAMPLE, POS_RANK, BankV2, HeadAuditor,  # noqa: E402
                         aggregate, fit_stats, instance_head_metrics,
                         selection_key, standardise)

ABLATIONS = {"A": (False, False), "B": (True, False), "C": (True, True)}
OBJECTIVES = ("h1", "h2", "h3")
BATCH = 8
LR = 1e-3
WD = 1e-2
P_DROP = 0.1
EVAL_EVERY = 2
PATIENCE = 10               # in evaluations, not epochs (== 20 epochs)
MAX_EPOCHS = 120            # the same budget v0's student grid ran under


# ---------------------------------------------------------------------------
def loss_h1(sp, sn):
    """Balanced BCE: the positive class carries the weight of the negative one,
    so the 16 positives of an image are not drowned by its 112 hard negatives."""
    B, P = sp.shape
    K = sn.shape[1]
    lg = torch.cat([sp, sn], dim=1)
    tg = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)], dim=1)
    return F.binary_cross_entropy_with_logits(
        lg, tg, pos_weight=torch.tensor(float(K) / max(P, 1), device=lg.device))


def loss_h2(sp, sn, w):
    """Pairwise logistic ranking, w(rank) = 1/sqrt(rank+1) normalised to mean 1.

    The target is the head ORDER, not the teacher's score values: nothing here
    tries to regress the teacher, only to put p above n."""
    diff = sp[:, :, None] - sn[:, None, :]                 # (B,P,K)
    l = F.softplus(-diff)
    return (l * w[:, :, None]).sum() / (w.sum() * diff.shape[2])


LOSSES = {"h1": lambda sp, sn, w: loss_h1(sp, sn),
          "h2": lambda sp, sn, w: loss_h2(sp, sn, w),
          "h3": lambda sp, sn, w: loss_h1(sp, sn) + loss_h2(sp, sn, w)}


# ---------------------------------------------------------------------------
class Cached:
    """fit+val rows with the standardised tensors materialised once on the device.

    The standardisation is `m3v2_common.prep_inputs` -- the same function the
    live pruner calls -- so there is no train/serve skew by construction, and
    the fp32 arithmetic here is the fp32 arithmetic there (the bank stores vis
    at fp16 and both sides start from that same fp16 tensor).
    """

    def __init__(self, bank: BankV2, stats, rows, device, chunk: int = 24):
        self.rows = np.asarray(rows)
        self.device = device
        st = {k: torch.from_numpy(np.asarray(v, np.float32)).to(device)
              for k, v in stats.items()}
        # Standardise in chunks: the fp32 (n, 1024, 4096) result is 5 GB for the
        # 300 fit+val rows and the fp16->fp32 upcast would double it if done in
        # one shot.  Chunking changes no arithmetic -- it calls the same
        # `standardise` the live pruner calls.
        def build(src, key, out_shape):
            out = torch.empty(out_shape, dtype=torch.float32, device=device)
            with torch.no_grad():
                for a in range(0, self.rows.size, chunk):
                    sl = slice(a, min(a + chunk, self.rows.size))
                    blk = torch.from_numpy(src[self.rows[sl]]).to(device)
                    out[sl] = standardise(blk, st[f"mu_{key}"], st[f"sd_{key}"])
                    del blk
            return out
        self.vis = build(bank.vis, "vis", (self.rows.size, bank.vis.shape[1],
                                           bank.vis.shape[2]))
        self.txt = build(bank.txt, "txt", (self.rows.size, bank.txt.shape[1],
                                           bank.txt.shape[2]))
        with torch.no_grad():
            Xr = torch.from_numpy(bank.X[self.rows]).to(device)
            self.X = ((Xr - st["mu_hand"]) / st["sd_hand"]).contiguous()
            del Xr
        self.txt_mask = (torch.arange(self.txt.shape[1], device=device)[None, :]
                         < torch.from_numpy(bank.txt_len[self.rows]).to(device)[:, None])
        self.s0 = torch.from_numpy(bank.s0[self.rows]).to(device)
        drop = np.stack([bank.drop[i] for i in self.rows])
        self.drop = torch.from_numpy(drop).to(device)
        pos, neg, w = [], [], []
        for i in self.rows:
            p, n, rk = bank.head_pools(i)
            pos.append(p)
            neg.append(n)
            ww = 1.0 / np.sqrt(rk + 1.0)
            w.append(ww / ww.mean())
        self.pos = torch.from_numpy(np.stack(pos)).to(device)          # (n,16)
        self.neg_pool = torch.from_numpy(np.stack(neg)).to(device)     # (n,112)
        self.w = torch.from_numpy(np.stack(w)).float().to(device)      # (n,16)
        self.n = len(self.rows)
        assert self.pos.shape[1] == POS_RANK, self.pos.shape

    def scores(self, model, idx, grad=False):
        """(B, 768) auditor scores on the dropped set for rows `idx`."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            sc = model(self.vis[idx], self.txt[idx], self.X[idx],
                       self.s0[idx], self.txt_mask[idx])
        return sc.gather(1, self.drop[idx])

    def batch_terms(self, idx, rng):
        """(pos scores, sampled hard-negative scores, positive weights)."""
        pos = self.pos[idx]                                        # (B,16)
        k = min(N_NEG_SAMPLE, self.neg_pool.shape[1])
        # sample K of the 112 hard negatives per image, without replacement
        r = torch.rand(pos.shape[0], self.neg_pool.shape[1], device=self.device)
        sel = r.argsort(dim=1)[:, :k]
        neg = self.neg_pool[idx].gather(1, sel)
        return pos, neg


def evaluate(model, cache, bank, rows, device, batch=16):
    """Head metrics on `rows`, computed exactly as the live pruner would pick."""
    model.eval()
    per = []
    for a in range(0, len(rows), batch):
        idx = torch.arange(a, min(a + batch, len(rows)), device=device)
        sc = cache.scores(model, idx).float().cpu().numpy()
        for j, r in enumerate(idx.tolist()):
            i = cache.rows[r]
            per.append(instance_head_metrics(bank.g2[i], bank.drop[i], sc[j]))
    model.train()
    return aggregate(per)


def run_config(bank, cache_fit, cache_val, name, abl, seed, device, epochs,
               lr=LR, wd=WD, p_drop=P_DROP):
    use_q, use_r = ABLATIONS[abl]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HeadAuditor(len(bank.features), bank.vis.shape[-1],
                        use_query=use_q, use_retained=use_r,
                        p_drop=p_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)

    # Two diagnostics that say whether the architecture is being trained at all:
    # what an UNTRAINED auditor already scores (the floor the grid has to beat),
    # and what the selected checkpoint scores on the rows it was fitted on (the
    # generalisation gap).  Neither feeds selection.
    init_val = evaluate(model, cache_val, bank, cache_val.rows.tolist(), device)

    best = dict(key=(-1.0, -1.0), state=None, ep=-1, metrics=None)
    since_best = 0
    hist = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(cache_fit.n)
        tot, nb = 0.0, 0
        for a in range(0, cache_fit.n, BATCH):
            idx = torch.from_numpy(perm[a:a + BATCH]).to(device)
            sc = cache_fit.scores(model, idx, grad=True)
            pos, neg = cache_fit.batch_terms(idx, rng)
            sp = sc.gather(1, pos)
            sn = sc.gather(1, neg)
            w = cache_fit.w[idx]
            loss = LOSSES[name](sp, sn, w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1

        if (ep + 1) % EVAL_EVERY == 0 or ep == epochs - 1:
            met = evaluate(model, cache_val, bank, cache_val.rows.tolist(), device)
            k = selection_key(met)
            hist.append(dict(ep=ep + 1, loss=tot / max(nb, 1), **{
                m: met[m] for m in ("top16_recall@16", "mean_rank@16", "hw@16",
                                    "ndcg@16")}))
            if k > best["key"]:
                best = dict(key=k, ep=ep + 1, metrics=met,
                            state={kk: v.detach().clone()
                                   for kk, v in model.state_dict().items()})
                since_best = 0
            else:
                since_best += 1
                if since_best >= PATIENCE:
                    break
    model.load_state_dict(best["state"])
    model.eval()
    fit_met = evaluate(model, cache_fit, bank, cache_fit.rows.tolist(), device)
    model.train()
    return model, best, hist, init_val, fit_met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["h1:C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--out", default="m3v2_auditor")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--wd", type=float, default=WD)
    ap.add_argument("--p-drop", type=float, default=P_DROP)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    bank = BankV2()
    print(f"[bank] {len(bank.key)} rows  vis {bank.vis.shape}  txt {bank.txt.shape}"
          f"  features {len(bank.features)}  ({time.time()-t0:.0f}s)", flush=True)

    fit_rows, val_rows = bank.rows("fit"), bank.rows("val")
    stats = fit_stats(bank, fit_rows)
    cache_fit = Cached(bank, stats, fit_rows, device)
    cache_val = Cached(bank, stats, val_rows, device)
    print(f"[cache] fit {cache_fit.n}  val {cache_val.n}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    out_path = os.path.join(OUTPUT_DIR, f"{args.out}.json")
    report = dict(seed_grid=list(args.seeds), ablations=ABLATIONS,
                  objectives=list(OBJECTIVES), batch=BATCH, lr=args.lr, wd=args.wd,
                  p_drop=args.p_drop, max_epochs=args.epochs, eval_every=EVAL_EVERY,
                  patience=PATIENCE, pos_rank=POS_RANK, n_neg_sample=N_NEG_SAMPLE,
                  selection="argmax (Top16 recall@16, hw@16) on val",
                  bank=dict(vis="m3_bank_v1", txt="m3v2_text"),
                  runs={})
    if os.path.exists(out_path):
        report["runs"] = json.load(open(out_path)).get("runs", {})

    for cfg in args.configs:
        name, abl = cfg.split(":")
        assert name in OBJECTIVES and abl in ABLATIONS, cfg
        for seed in args.seeds:
            run_key = f"{cfg}|s{seed}"
            tc = time.time()
            model, best, hist, init_val, fit_met = run_config(
                bank, cache_fit, cache_val, name, abl, seed, device, args.epochs,
                lr=args.lr, wd=args.wd, p_drop=args.p_drop)
            met = best["metrics"]
            ck = dict(state_dict=best["state"], **model.config(),
                      features=list(bank.features), objective=name, ablation=abl,
                      seed=seed, best_epoch=best["ep"], val_metrics=met,
                      init_val_metrics=init_val, fit_metrics=fit_met,
                      selection=dict(top16_recall_at_16=met["top16_recall@16"],
                                     hw_at_16=met["hw@16"],
                                     mean_rank_at_16=met["mean_rank@16"]),
                      **{k: np.asarray(v, np.float32) for k, v in stats.items()})
            torch.save(ck, os.path.join(OUTPUT_DIR, f"{args.out}_{name}{abl}_s{seed}.pt"))
            report["runs"][run_key] = dict(
                objective=name, ablation=abl, seed=seed, best_epoch=best["ep"],
                n_params=int(sum(p.numel() for p in model.parameters())),
                val=met, init_val=init_val, fit=fit_met, history=hist,
                seconds=time.time() - tc)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=1)
            print(f"  {run_key:10s} ep{best['ep']:3d}  Top16rec@16 "
                  f"{met['top16_recall@16']:.4f}  meanrank@16 {met['mean_rank@16']:7.2f}"
                  f"  hw@16 {met['hw@16']:.4f}  ndcg@16 {met['ndcg@16']:.4f}"
                  f"  | init {init_val['top16_recall@16']:.4f}"
                  f"  fit {fit_met['top16_recall@16']:.4f}"
                  f"  ({time.time()-tc:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()

    print(f"[saved] {out_path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
