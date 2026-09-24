"""
M3-v0 step 2 -- fit the forward-only miss student on the fit+val rows.

Protocol (frozen here, before any held-out number exists):

* labels come from the P1-G2 teacher cache and ONLY for `fit` + `val` rows;
* the hand-feature and vision-feature standardisers are fitted on `fit` rows;
* **model selection is on `val`, and on the deployment metric** -- the overlap
  between the student's top-r and the ORACLE's top-r among the dropped tokens,
  at r = 8/16/32.  Not token AUC: AUC over 768 dropped tokens is dominated by
  easy mid-ranked tokens, while the whole value of a rescue sits in the head
  (S2-C2: 53 % of the teacher gap is closed by 8 tokens).  S2-C5A already
  produced one selection artefact in this project by ranking on the wrong
  metric; this is the metric the arm is actually deployed on.
* `test` (the held-out 150) is scored once, after the model is frozen, and is
  reported as `test_posthoc` -- descriptive only, never used to choose.

Configs (the brief's "pilot" scope: two natural positive definitions, plus one
head-shaped objective, crossed with one normalisation switch):

    strict  BCE on teacher Top-64  \\ S0
    loose   BCE on teacher Top-256 \\ S0
    head    listwise KL against a geometric-decay target over the teacher's
            own ranking of the dropped tokens -- directly shaped for top-r
    per_instance_z  z-score the hand features within each image's dropped set

Usage
    python scripts/discovery/m3_train.py
"""
from __future__ import annotations

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

from common import OUTPUT_DIR                                       # noqa: E402
from m3_common import (D_VIS, FEATURES, MissStudent, POS_K,         # noqa: E402
                       Standardiser, teacher_topk)

R_GRID = (8, 16, 32)
SEED = 0
N_NEG = 256                 # negatives sampled per instance per epoch (BCE)
HEAD_DECAY = 16.0           # geometric-decay scale of the `head` target
HEAD_T = 1.0                # softmax temperature on the student side


class Bank:
    """The frozen 450 with their cached pre-LLM audit state."""

    def __init__(self, tag: str = "m3_bank"):
        z = np.load(os.path.join(OUTPUT_DIR, f"{tag}.npz"), allow_pickle=False)
        self.key = [str(k) for k in z["key"]]
        self.ds = np.array([str(s) for s in z["ds"]])
        self.split = np.array([str(s) for s in z["split"]])
        self.X = z["X"].astype(np.float32)                    # (N,1024,F)
        self.s0 = z["s0"].astype(np.int64)                    # (N,256) sorted
        self.g2 = z["g2"].astype(np.float32)                  # (N,1024)
        self.vis = z["vis"]                                   # (N,1024,4096) fp16
        self.drop = []
        for i in range(self.X.shape[0]):
            keep = np.zeros(self.X.shape[1], dtype=bool)
            keep[self.s0[i]] = True
            self.drop.append(np.where(~keep)[0])
        self.n_vis = self.X.shape[1]
        # Column names come from the bank when it carries them; otherwise the
        # bank predates the field and its columns are the FEATURES prefix
        # (features are only ever APPENDED, never reordered or removed).
        self.features = ([str(x) for x in z["feature_names"]]
                         if "feature_names" in z.files else list(FEATURES)[:self.X.shape[2]])
        assert self.features == list(FEATURES)[:len(self.features)], \
            "bank feature columns are not a prefix of FEATURES"

    def rows(self, split):
        return np.where(self.split == split)[0]


def vis_stats_fit(bank: Bank, fit_rows, chunk=64):
    """Per-dimension mean/std of the vision feature over the FIT rows only."""
    n = 0
    s = np.zeros(D_VIS, dtype=np.float64)
    sq = np.zeros(D_VIS, dtype=np.float64)
    for i in fit_rows:
        v = bank.vis[i].astype(np.float32)
        s += v.sum(0, dtype=np.float64)
        sq += np.square(v, dtype=np.float64).sum(0)
        n += v.shape[0]
    mu = s / n
    var = np.maximum(sq / n - mu * mu, 1e-12)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)


# ---------------------------------------------------------------------------
def flat_pool(bank: Bank, rows, labels, rng, n_neg=N_NEG):
    """(instance, row) pairs for pointwise training: positives + n_neg negatives."""
    inst, idx, yy = [], [], []
    for i in rows:
        d = bank.drop[i]
        lab = labels[i][d]                    # labels are 1024-long; slice to dropped
        pos = d[lab > 0]
        neg = d[lab <= 0]
        take = min(n_neg, neg.size)
        sel = np.concatenate([pos, rng.choice(neg, take, replace=False)]) \
            if take else pos
        inst.append(np.full(sel.size, i, dtype=np.int64))
        idx.append(sel)
        yy.append(labels[i][sel])
    return np.concatenate(inst), np.concatenate(idx), np.concatenate(yy)


def grid_pool(bank: Bank, rows, device):
    """(n_inst, n_dropped, ...) device tensors -- the per-instance listwise view."""
    d0 = bank.drop[rows[0]].size
    assert all(bank.drop[i].size == d0 for i in rows), "ragged dropped set"
    idx = np.stack([bank.drop[i] for i in rows])
    inst = np.repeat(rows, d0).reshape(len(rows), d0)
    Xh = torch.from_numpy(bank.X[inst, idx]).to(device)
    V = torch.from_numpy(bank.vis[inst, idx].astype(np.float32)).to(device)
    g2 = torch.from_numpy(bank.g2[inst, idx]).to(device)
    return Xh, V, g2


def run_config(bank, cfg, mu_h, sd_h, mu_v, sd_v, device, epochs):
    """Train one config; return the val-selected checkpoint and its metrics."""
    name, per_z = cfg
    fit_rows, val_rows = bank.rows("fit"), bank.rows("val")

    # Labels live in the FULL 1024-token index space (not the 768 dropped
    # positions), because every pool below indexes with absolute token ids.
    labels = None
    if name != "head":
        labels = {}
        for i in range(bank.X.shape[0]):
            lab = np.zeros(bank.n_vis, dtype=np.float32)
            lab[teacher_topk(bank.g2[i], bank.s0[i], POS_K[name])] = 1.0
            labels[i] = lab

    mu_h_t = torch.from_numpy(mu_h).to(device)
    sd_h_t = torch.from_numpy(sd_h).to(device)
    mu_v_t = torch.from_numpy(mu_v).to(device)
    sd_v_t = torch.from_numpy(sd_v).to(device)

    def prep(Xh, V):
        zh = (Xh - mu_h_t) / sd_h_t
        zv = (V - mu_v_t) / sd_v_t
        if per_z:
            zh = (zh - zh.mean(0, keepdim=True)) / zh.std(0, keepdim=True).clamp_min(1e-6)
        return zh, zv

    # ---- frozen pools -----------------------------------------------------
    if name == "head":
        tXh, tV, tg2 = grid_pool(bank, fit_rows, device)
        # geometric-decay target over each image's OWN teacher ranking
        order = torch.argsort(torch.argsort(tg2, dim=1), dim=1)
        w = torch.exp(-order.float() / HEAD_DECAY)
        tgt = w / w.sum(dim=1, keepdim=True)
    else:
        ti, tx, ty = flat_pool(bank, fit_rows, labels, np.random.default_rng(SEED))
        tXh = torch.from_numpy(bank.X[ti, tx]).to(device)
        tV = torch.from_numpy(bank.vis[ti, tx].astype(np.float32)).to(device)
        ty = torch.from_numpy(ty).to(device)
        pos_w = torch.tensor([(ty == 0).sum().item() / max((ty == 1).sum().item(), 1)],
                             device=device)

    vXh, vV, _ = grid_pool(bank, val_rows, device)
    val_slices, o = [], 0
    for i in val_rows:
        m = bank.drop[i].size
        val_slices.append((i, o, o + m))
        o += m

    model = MissStudent(len(bank.features)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best = dict(score=-1.0, state=None, ep=-1, metrics=None)
    for ep in range(epochs):
        model.train()
        if name == "head":
            perm = torch.randperm(tXh.shape[0], device=device)
            for b in range(0, perm.numel(), 8):
                sel = perm[b:b + 8]
                zh, zv = prep(tXh[sel], tV[sel])
                s = model(zh, zv)                       # (g, 768)
                loss = -(tgt[sel] * F.log_softmax(s / HEAD_T, dim=1)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            perm = torch.randperm(tXh.shape[0], device=device)
            for b in range(0, perm.numel(), 16384):
                sel = perm[b:b + 16384]
                zh, zv = prep(tXh[sel], tV[sel])
                loss = F.binary_cross_entropy_with_logits(
                    model(zh, zv), ty[sel], pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                opt.step()

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                zh, zv = prep(vXh.reshape(-1, vXh.shape[-1]),
                              vV.reshape(-1, vV.shape[-1]))
                sv = model(zh, zv).float().cpu().numpy()
            met = head_metrics(bank, val_rows, val_slices, sv)
            if met["overlap"][16] > best["score"]:
                best = dict(score=met["overlap"][16], ep=ep + 1,
                            state={k: v.detach().clone()
                                   for k, v in model.state_dict().items()},
                            metrics=met)
    model.load_state_dict(best["state"])
    model.eval()
    return model, best


def head_metrics(bank, rows, slices, scores):
    """overlap@r with the oracle, and recall@r of the positive set."""
    ov, rec = {}, {}
    for r in R_GRID:
        o_, c_ = [], []
        for i, a, b in slices:
            s = scores[a:b]
            d = bank.drop[i]
            pick = d[np.argsort(-s)[:r]]
            top_t = d[np.argsort(-bank.g2[i][d])[:r]]
            o_.append(len(set(pick.tolist()) & set(top_t.tolist())) / r)
            pos = set(d[np.isin(d, teacher_topk(bank.g2[i], bank.s0[i], POS_K["strict"]))].tolist())
            c_.append(len(set(pick.tolist()) & pos) / max(min(r, len(pos)), 1))
        ov[r] = float(np.mean(o_))
        rec[r] = float(np.mean(c_))
    return dict(overlap=ov, recall=rec, chance_overlap=16.0 / 768.0)


def score_split(bank, model, mu_h, sd_h, mu_v, sd_v, rows, device, per_z):
    Xh, V, _ = grid_pool(bank, rows, device)
    with torch.no_grad():
        zh = (Xh - torch.from_numpy(mu_h).to(device)) / torch.from_numpy(sd_h).to(device)
        zv = (V - torch.from_numpy(mu_v).to(device)) / torch.from_numpy(sd_v).to(device)
        if per_z:
            zh = (zh - zh.mean(0, keepdim=True)) / zh.std(0, keepdim=True).clamp_min(1e-6)
        s = model(zh.reshape(-1, zh.shape[-1]), zv.reshape(-1, zv.shape[-1]))
        s = s.float().cpu().numpy()
    slices, o = [], 0
    for i in rows:
        m = bank.drop[i].size
        slices.append((i, o, o + m))
        o += m
    return head_metrics(bank, rows, slices, s)


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m3_bank")
    ap.add_argument("--out", default="m3_miss")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    SEED = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    bank = Bank(args.tag)
    print(f"[bank] fit {len(bank.rows('fit'))}  val {len(bank.rows('val'))}  "
          f"test {len(bank.rows('test'))}   ({time.time()-t0:.0f}s)")

    fit_rows = bank.rows("fit")
    mu_v, sd_v = vis_stats_fit(bank, fit_rows)
    hand = np.concatenate([bank.X[i][bank.drop[i]] for i in fit_rows])
    std_h = Standardiser.fit(hand)
    print(f"[stats] hand mu/sd over {hand.shape[0]} dropped-token rows; "
          f"vis mu/sd over {len(fit_rows)*bank.n_vis} rows  ({time.time()-t0:.0f}s)")

    report = dict(seed=args.seed, features=list(bank.features), pos_k=POS_K,
                  r_grid=list(R_GRID), head_decay=HEAD_DECAY, n_neg=N_NEG,
                  n_fit=len(fit_rows), n_val=len(bank.rows("val")),
                  n_test=len(bank.rows("test")), configs={})

    best = None
    for name in ("strict", "loose", "head"):
        for per_z in (False, True):
            cname = f"{name}|z{int(per_z)}"
            tc = time.time()
            model, b = run_config(bank, (name, per_z), std_h.mu, std_h.sd,
                                  mu_v, sd_v, device, args.epochs)
            tm = score_split(bank, model, std_h.mu, std_h.sd, mu_v, sd_v,
                             bank.rows("test"), device, per_z)
            report["configs"][cname] = dict(
                target=name, per_instance_z=per_z, best_epoch=b["ep"],
                val=b["metrics"], test_posthoc=tm, train_seconds=time.time() - tc,
                note="test_posthoc is descriptive only; selection was on val overlap@16")
            print(f"  {cname:12s} val ovl@8/16/32 "
                  f"{b['metrics']['overlap'][8]:.3f}/{b['metrics']['overlap'][16]:.3f}/"
                  f"{b['metrics']['overlap'][32]:.3f}  rec@16 "
                  f"{b['metrics']['recall'][16]:.3f}  ep{b['ep']:3d}"
                  f"  | test(post-hoc) ovl@16 {tm['overlap'][16]:.3f}"
                  f"  ({time.time()-tc:.0f}s)", flush=True)
            if best is None or b["score"] > best[1]:
                best = (cname, b["score"], name, per_z, b, model)

    cname, score, name, per_z, b, model = best
    ck = dict(state_dict=b["state"], mu_hand=std_h.mu, sd_hand=std_h.sd,
              mu_vis=mu_v, sd_vis=sd_v, d_hand=len(bank.features), d_vis=D_VIS,
              features=list(bank.features), target=name, per_instance_z=per_z,
              config_name=cname, val_overlap16=score, seed=args.seed,
              head_decay=HEAD_DECAY)
    torch.save(ck, os.path.join(OUTPUT_DIR, f"{args.out}.pt"))
    report["selected"] = dict(config=cname, target=name, per_instance_z=per_z,
                              val_overlap16=score,
                              selection_rule="argmax val overlap@16 over the 6 pilot configs")
    with open(os.path.join(OUTPUT_DIR, f"{args.out}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"[selected] {cname}  val overlap@16 {score:.4f}  "
          f"(chance {16/768:.4f})")
    print(f"[saved] {args.out}.pt / {args.out}.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
