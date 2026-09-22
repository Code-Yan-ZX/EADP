"""
S2-C3 step 1: retarget the S2-C1 linear scorer at the head of the teacher's
ranking, everything else held fixed.

The model, the features, the split, the loss form, the optimizer and the early
stopping criterion are all the S2-C1 ones. The only thing that changes is the
teacher reading the loss is computed against (see ``s2c3_common`` for the four
targets). Each target is trained with three seeds so that every headline
difference can be read against the retraining noise rather than against zero --
this is a variance control, not a hyperparameter search: no configuration is
tuned, and the three seeds differ only in weight initialisation and batch order.

BASE is trained here as well as being available as the cached S2-C1 ``LIN_L4``
weights. The retrained BASE is a procedural control: if it lands on the cached
one within seed spread, then a difference between the head arms and the cached
baseline is attributable to the target rather than to the reimplementation.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import time_callable                       # noqa: E402
from s1_audit import OUT                               # noqa: E402
from s2c1_train import FeatureSource, TokenOnly, standardize_stats  # noqa: E402
from s2c3_common import (ARMS, LAYER, N_VIS, SEEDS, TEACHER, aggregate,  # noqa: E402
                         head_metrics, load_plan, pos_and_weights,
                         target_description, teacher_orders)

LR = 1e-3
WD = 1e-4
BATCH = 8
MAX_EPOCHS = 80
PATIENCE = 20
LAMBDA_RANK = 0.5
MARGIN = 1.0


def build_targets(arm, orders, keys):
    """Per-image pos_idx / neg_idx / token weight / pair weight / membership y."""
    P, N, W, PW, Y = [], [], [], [], []
    for k in keys:
        pos, tw, pw, pos_weight = pos_and_weights(arm, orders[k])
        neg = np.setdiff1d(orders[k], pos, assume_unique=False)
        y = np.zeros(N_VIS, dtype=np.float32)
        y[pos] = 1.0
        P.append(pos.astype(np.int64))
        N.append(neg.astype(np.int64))
        W.append(tw)
        PW.append(pw.astype(np.float32))
        Y.append(y)
    return (np.stack(P), np.stack(N), np.stack(W), PW, np.stack(Y), pos_weight)


def compute_loss(s, y, pos_idx, neg_idx, tok_w, pair_w, pos_weight):
    """Weighted balanced BCE + lambda * weighted all-pairs margin ranking.

    Both weight tensors have mean 1 (BCE weights over the 1 024 tokens, pair
    weights over the positives), so ``lambda`` keeps the same meaning in every
    arm; with uniform weights this is exactly ``s2c1_train.compute_loss``.
    """
    bce = F.binary_cross_entropy_with_logits(
        s, y, weight=tok_w,
        pos_weight=torch.tensor(float(pos_weight), device=s.device))
    sp = torch.gather(s, 1, pos_idx)                       # (B, P)
    sn = torch.gather(s, 1, neg_idx)                       # (B, N)
    pair = F.relu(MARGIN - sp.unsqueeze(-1) + sn.unsqueeze(1))
    # pair_w has mean 1 per image, so this is the weighted mean over all B*P*N
    # pairs -- identical to ``s2c1_train``'s ``.mean()`` when the weights are 1.
    rank = (pair * pair_w.unsqueeze(-1)).sum() / (pair.shape[0] * pair.shape[1]
                                                  * pair.shape[2])
    return bce + LAMBDA_RANK * rank, float(bce.detach()), float(rank.detach())


@torch.no_grad()
def score_rows(model, fs, bs=16):
    outs = []
    for a in range(0, len(fs.rows), bs):
        sel = list(range(a, min(a + bs, len(fs.rows))))
        h, _ = fs.batch(sel, False)
        outs.append(model(h, None).float().cpu())
    return torch.cat(outs, 0).numpy()


def eval_split(model, fs, orders, keys):
    s = score_rows(model, fs)
    per = [head_metrics(s[i], orders[k]) for i, k in enumerate(keys)]
    return s, aggregate(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--tag", default="s2c3")
    args = ap.parse_args()

    meta, plan, keys, rows_of = load_plan()
    G = np.load(os.path.join(OUT, TEACHER))
    orders = teacher_orders(G, keys)
    print(f"[split] fit={len(rows_of['fit'])} val={len(rows_of['val'])} "
          f"test={len(rows_of['test'])} causal={len(rows_of['causal'])}")

    # The standardization is one fixed preprocessing step (fit-split mean/std).
    # It does not depend on the target or the seed, so it is computed once here
    # rather than once per arm -- same arithmetic as s2c1_train, same numbers.
    t0 = time.time()
    H = np.load(os.path.join(OUT, f"s2c1_feats_L{LAYER}.npy"), mmap_mode="r")
    mu, sd = standardize_stats(H, rows_of["fit"])
    del H
    print(f"[stats] fit-split mean/std over {len(rows_of['fit'])} images "
          f"({time.time() - t0:.0f} s)")

    fs_fit = FeatureSource(LAYER, rows_of["fit"], mu, sd, "cuda", cache=True)
    fs_val = FeatureSource(LAYER, rows_of["val"], mu, sd, "cuda", cache=True)
    fs_test = FeatureSource(LAYER, rows_of["test"], mu, sd, "cuda")
    fit_keys = [keys[i] for i in rows_of["fit"]]
    val_keys = [keys[i] for i in rows_of["val"]]
    test_keys = [keys[i] for i in rows_of["test"]]

    results, score_bank = {}, {}
    for arm in args.arms:
        P, Ng, W, PW, Y, pos_weight = build_targets(arm, orders, keys)
        P_fit, N_fit, W_fit, Y_fit = (P[rows_of["fit"]], Ng[rows_of["fit"]],
                                      W[rows_of["fit"]], Y[rows_of["fit"]])
        PW_fit = np.stack([PW[i] for i in rows_of["fit"]])
        y_val = Y[rows_of["val"]]
        n_fit = len(rows_of["fit"])
        print(f"\n=== {arm}: {target_description(arm)}")

        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            model = TokenOnly(4096).to("cuda")
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            t_start = time.time()
            best = dict(val_overlap=-1.0, epoch=-1, state=None, val=None)
            hist = []
            for ep in range(MAX_EPOCHS):
                model.train()
                perm = rng.permutation(n_fit)
                ep_loss = []
                for a in range(0, n_fit, BATCH):
                    sel = perm[a:a + BATCH]
                    h, _ = fs_fit.batch(sel, False)
                    s = model(h, None)
                    loss, bce, rk = compute_loss(
                        s, torch.from_numpy(Y_fit[sel]).to("cuda"),
                        torch.from_numpy(P_fit[sel]).to("cuda"),
                        torch.from_numpy(N_fit[sel]).to("cuda"),
                        torch.from_numpy(W_fit[sel]).to("cuda"),
                        torch.from_numpy(PW_fit[sel]).to("cuda"),
                        pos_weight)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    ep_loss.append(float(loss.detach()))
                model.eval()
                _, vm = eval_split(model, fs_val, orders, val_keys)
                hist.append(dict(epoch=ep, loss=float(np.mean(ep_loss)), **vm))
                if vm["overlap256"] > best["val_overlap"]:
                    best = dict(val_overlap=vm["overlap256"], epoch=ep, val=vm,
                                state={k: v.detach().clone()
                                       for k, v in model.state_dict().items()})
                if ep - best["epoch"] >= PATIENCE:
                    break
            model.load_state_dict(best["state"])
            model.eval()

            _, test_m = eval_split(model, fs_test, orders, test_keys)
            _, train_m = eval_split(model, fs_fit, orders, fit_keys)
            s_all = score_rows(model, FeatureSource(LAYER, list(range(len(keys))),
                                                    mu, sd, "cuda"))
            tag = f"{arm}_s{seed}"
            for k, v in zip(keys, s_all):
                score_bank[f"{tag}__{k}"] = v.astype(np.float32)

            # diagnostic only: what a val-head-recall early stop would have
            # picked. Recorded so the choice of stopping metric is inspectable;
            # never used for selection or for the verdict.
            hb = max(hist, key=lambda r: r["head_recall32"])
            results[tag] = dict(
                arm=arm, seed=seed, target=target_description(arm), layer=LAYER,
                n_params=int(sum(p.numel() for p in model.parameters())),
                train_seconds=float(time.time() - t_start),
                best_epoch=int(best["epoch"]), epochs_run=len(hist),
                train=train_m, val=best["val"], test=test_m,
                val_head32_epoch=dict(epoch=int(hb["epoch"]),
                                      val_head32=float(hb["head_recall32"]),
                                      val_head8=float(hb["head_recall8"])),
                val_history=hist,
            )
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       os.path.join(OUT, f"{args.tag}_{tag}.pt"))
            print(f"[{tag}] ep={best['epoch']}/{len(hist)} "
                  f"val_ov={best['val']['overlap256']:.3f} "
                  f"val_head8={best['val']['head_recall8']:.3f} "
                  f"val_head32={best['val']['head_recall32']:.3f} | "
                  f"test_ov={test_m['overlap256']:.3f} "
                  f"h8={test_m['head_recall8']:.3f} "
                  f"h32={test_m['head_recall32']:.3f} "
                  f"a8={test_m['head_agree8']:.3f} | {time.time() - t_start:.0f}s")

    # latency of the retrained scorer (identical form to LIN_L4)
    fs_one = FeatureSource(LAYER, rows_of["test"][:1], mu, sd, "cuda")
    h1, _ = fs_one.batch([0], False)
    m = TokenOnly(4096).to("cuda").eval()
    with torch.no_grad():
        sc_mean, sc_std = time_callable(lambda: m(h1, None), repeat=30, warmup=5)
        tk_mean, tk_std = time_callable(lambda: torch.topk(m(h1, None), 256, dim=-1),
                                        repeat=30, warmup=5)

    np.savez_compressed(os.path.join(OUT, f"{args.tag}_scores.npz"), **score_bank)
    json.dump({"config": vars(args), "results": results,
               "split_counts": {k: len(v) for k, v in rows_of.items()},
               "keys": keys, "layer": LAYER, "seeds": args.seeds,
               "latency": dict(scorer_ms=sc_mean, scorer_ms_std=sc_std,
                               topk_ms=tk_mean, topk_ms_std=tk_std,
                               prefix_ms=meta["latency"][str(LAYER)]["prefix_ms_mean"]),
               "arms": {a: target_description(a) for a in args.arms}},
              open(os.path.join(OUT, f"{args.tag}_train.json"), "w"), indent=1)
    print(f"\n[saved] {args.tag}_scores.npz ({len(score_bank)} vectors) "
          f"+ {args.tag}_train.json")


if __name__ == "__main__":
    main()
