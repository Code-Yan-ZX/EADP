"""
S2-C5 step 1: the controlled scorer ladder.

Everything except the scorer form is the S2-C3 HEAD_RANK configuration, reused
verbatim: same feature cache (L4, 1024x4096), same split (fit 240 / val 60 /
held-out 150), same target (positives = teacher Top-32, rank-graded weights
32/(r+1) mean-normalised), same loss family (balanced BCE + lambda * weighted
all-pairs margin ranking, lambda = 0.5, margin = 1.0), same optimizer and
schedule, same early-stopping criterion (validation Top-256 overlap), same three
seeds. The frozen reference arm A is the cached S2-C3 HEAD_RANK itself, so it is
not retrained here -- it is the same artifact the S2-C4 verdict was computed on.

The arms are defined in ``s2c5_models``; the point of the ladder is that the
trunk is identical across arms and only the information reaching the read-out
changes, so a difference cannot be attributed to parameter count.

Two mechanism controls are run on every contextual arm, at held-out inference:

  WRONG_HELDOUT   the image's context / key-value set is replaced by another
                  held-out image's (a derangement -- no image keeps its own)
  WRONG_FIT       the same, but the substitute comes from the fit split, so the
                  substituted context is drawn from images the scorer never saw
                  as evaluation targets

and the diagnostic LOCAL-MLP-WIDE (2.16 M params, deliberately unmatched) is run
to bound the token-local hypothesis from above rather than only from the matched
arm's landing point.

Primary metric throughout: held-out teacher Top-8 recall@256. Supplementary:
Top-16/32 recall, exact Top-8/16/32 agreement, Top-256 overlap. Top-256 overlap
is never used to veto a head-recovery gain (S2-C3 recorded HEAD_RANK at
overlap256 = 0.52 against BASE's 0.56 while gaining head recall).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import time_callable                                  # noqa: E402
from s1_audit import OUT                                          # noqa: E402
from s2c1_train import FeatureSource, standardize_stats            # noqa: E402
from s2c3_common import (LAYER, N_VIS, SEEDS, TEACHER, aggregate,  # noqa: E402
                         head_metrics, load_plan, teacher_orders)
from s2c3_train import build_targets, compute_loss                 # noqa: E402
from s2c5_models import ARMS, CONTEXTUAL, build, n_params          # noqa: E402

TARGET_ARM = "HEAD_RANK"        # the S2-C3 target/loss family, unchanged
TAG = "s2c5"
LR = 1e-3
WD = 1e-4
BATCH = 8
MAX_EPOCHS = 80
PATIENCE = 20
BS_EVAL = 16


# ------------------------------------------------------------- evaluation ----
@torch.no_grad()
def context_bank(model, fs, bs=BATCH):
    """Per-image context (GLOBAL-CTX) or per-image token set (SET-CTX)."""
    outs = []
    for a in range(0, len(fs.rows), bs):
        sel = list(range(a, min(a + bs, len(fs.rows))))
        h, _ = fs.batch(sel, False)
        if model.kind == "global":
            outs.append(model.context(h).float().cpu())
        else:
            outs.append(model.trunk(h).float().cpu())
    return torch.cat(outs, 0).numpy()


@torch.no_grad()
def score_all(model, fs, perm=None, bank=None, bs=BS_EVAL):
    """Score every row of ``fs``; with ``perm``/``bank``, read borrowed context.

    ``perm`` maps a row index to the row whose context it should read, so
    ``perm = arange(n)`` is the honest forward pass and any derangement is the
    wrong-image control. ``bank`` is indexed in the split's own row space, which
    for a fresh FeatureSource is exactly the batch's ``sel`` positions.
    """
    outs = []
    for a in range(0, len(fs.rows), bs):
        sel = list(range(a, min(a + bs, len(fs.rows))))
        h, _ = fs.batch(sel, False)
        if bank is None or model.kind == "local" or model.kind == "local_wide":
            s = model(h)
        else:
            idx = np.asarray(perm)[sel]
            borrowed = torch.from_numpy(np.ascontiguousarray(bank[idx])).to(h.device)
            s = model(h, borrowed) if model.kind == "global" else model(h, kv=borrowed)
        outs.append(s.float().cpu())
    return torch.cat(outs, 0).numpy()


def eval_split(model, fs, orders, keys):
    s = score_all(model, fs)
    per = [head_metrics(s[i], orders[k]) for i, k in enumerate(keys)]
    return s, aggregate(per), per


def derangement(n: int, seed: int) -> np.ndarray:
    """A permutation of range(n) with no fixed point."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    for i in range(n):
        if p[i] == i:
            j = (i + 1) % n
            p[i], p[j] = p[j], p[i]
    assert not (p == np.arange(n)).any()
    return p


def control_eval(model, fs, orders, keys, bank, seed):
    """Score the held-out set with every image reading a *different* image's
    context (a derangement, so no image keeps its own)."""
    perm = derangement(len(keys), seed)
    s_wrong = score_all(model, fs, perm=perm, bank=bank)
    return aggregate([head_metrics(s_wrong[i], orders[k])
                      for i, k in enumerate(keys)]), perm


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()

    meta, plan, keys, rows_of = load_plan()
    G = np.load(os.path.join(OUT, TEACHER))
    orders = teacher_orders(G, keys)
    print(f"[split] fit={len(rows_of['fit'])} val={len(rows_of['val'])} "
          f"test={len(rows_of['test'])} causal={len(rows_of['causal'])} "
          f"| target={TARGET_ARM} layer=L{LAYER}")

    t0 = time.time()
    H = np.load(os.path.join(OUT, f"s2c1_feats_L{LAYER}.npy"), mmap_mode="r")
    mu, sd = standardize_stats(H, rows_of["fit"])
    del H
    print(f"[stats] fit-split mean/std ({time.time() - t0:.0f} s)")

    P, Ng, W, PW, Y, pos_weight = build_targets(TARGET_ARM, orders, keys)
    P_fit, N_fit, W_fit, Y_fit = (P[rows_of["fit"]], Ng[rows_of["fit"]],
                                  W[rows_of["fit"]], Y[rows_of["fit"]])
    PW_fit = np.stack([PW[i] for i in rows_of["fit"]])

    fs_fit = FeatureSource(LAYER, rows_of["fit"], mu, sd, "cuda", cache=True)
    fs_val = FeatureSource(LAYER, rows_of["val"], mu, sd, "cuda", cache=True)
    fs_test = FeatureSource(LAYER, rows_of["test"], mu, sd, "cuda")
    val_keys = [keys[i] for i in rows_of["val"]]
    test_keys = [keys[i] for i in rows_of["test"]]
    fit_keys = [keys[i] for i in rows_of["fit"]]
    n_fit = len(rows_of["fit"])

    results, score_bank, perimg = {}, {}, {}
    for arm in args.arms:
        model0 = build(arm)
        print(f"\n=== {arm}: {model0.describe()}  "
              f"params={n_params(model0):,}")
        del model0

        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            model = build(arm).to("cuda")
            np_ = n_params(model)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            t_start = time.time()
            best = dict(val_overlap=-1.0, epoch=-1, state=None, val=None)
            hist = []
            for ep in range(args.max_epochs):
                model.train()
                perm = rng.permutation(n_fit)
                ep_loss = []
                for a in range(0, n_fit, BATCH):
                    sel = perm[a:a + BATCH]
                    h, _ = fs_fit.batch(sel, False)
                    s = model(h)
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
                _, vm, _ = eval_split(model, fs_val, orders, val_keys)
                hist.append(dict(epoch=ep, loss=float(np.mean(ep_loss)), **vm))
                if vm["overlap256"] > best["val_overlap"]:
                    best = dict(val_overlap=vm["overlap256"], epoch=ep, val=vm,
                                state={k: v.detach().clone()
                                       for k, v in model.state_dict().items()})
                if ep - best["epoch"] >= PATIENCE:
                    break
            model.load_state_dict(best["state"])
            model.eval()

            s_test, test_m, test_per = eval_split(model, fs_test, orders, test_keys)
            _, train_m, _ = eval_split(model, fs_fit, orders, fit_keys)

            # ---- mechanism controls (contextual arms only) -----------------
            ctrl = {}
            if arm in CONTEXTUAL:
                bank = context_bank(model, fs_test)
                ctrl["WRONG_HELDOUT"], _ = control_eval(
                    model, fs_test, orders, test_keys, bank, seed)
                # substitute context from fit-split images the scorer never saw
                bank_fit = context_bank(model, fs_fit)
                perm_fit = np.random.default_rng(seed + 991).integers(
                    0, len(bank_fit), size=len(test_keys))
                s_fit = score_all(model, fs_test, perm=perm_fit, bank=bank_fit)
                ctrl["WRONG_FIT"] = aggregate(
                    [head_metrics(s_fit[i], orders[k])
                     for i, k in enumerate(test_keys)])
                # honest pass through the same code path, as a self-check that
                # the borrowed-context plumbing did not change the real score
                s_self = score_all(model, fs_test, perm=np.arange(len(test_keys)),
                                   bank=bank)
                ctrl["SELF_CHECK_max_abs_diff"] = float(
                    np.abs(s_self - s_test).max())

            # ---- score vectors for every key -------------------------------
            s_all = score_all(model, FeatureSource(LAYER, list(range(len(keys))),
                                                  mu, sd, "cuda"))
            tag = f"{arm}_s{seed}"
            for k, v in zip(keys, s_all):
                score_bank[f"{tag}__{k}"] = v.astype(np.float32)
            perimg[tag] = np.stack([
                [p["head_recall8"], p["head_recall16"], p["head_recall32"],
                 p["head_agree8"], p["head_agree16"], p["head_agree32"],
                 p["overlap256"]] for p in test_per]).astype(np.float32)

            results[tag] = dict(
                arm=arm, seed=seed, kind=model.kind, n_params=np_,
                access=model.describe(), target=TARGET_ARM, layer=LAYER,
                train_seconds=float(time.time() - t_start),
                best_epoch=int(best["epoch"]), epochs_run=len(hist),
                train=train_m, val=best["val"], test=test_m,
                controls=ctrl,
                val_history=[{k: v for k, v in hh.items()} for hh in hist],
            )
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       os.path.join(OUT, f"{args.tag}_{tag}.pt"))
            print(f"[{tag}] ep={best['epoch']}/{len(hist)} "
                  f"val_ov={best['val']['overlap256']:.3f} | "
                  f"test_ov={test_m['overlap256']:.3f} "
                  f"h8={test_m['head_recall8']:.3f} "
                  f"h16={test_m['head_recall16']:.3f} "
                  f"h32={test_m['head_recall32']:.3f} "
                  f"a8={test_m['head_agree8']:.3f}"
                  + ("" if not ctrl else
                     f" | wrong-h8={ctrl['WRONG_HELDOUT']['head_recall8']:.3f} "
                     f"wrongfit-h8={ctrl['WRONG_FIT']['head_recall8']:.3f} "
                     f"selfdiff={ctrl['SELF_CHECK_max_abs_diff']:.1e}")
                  + f" | {time.time() - t_start:.0f}s", flush=True)

    # latency of every arm, same protocol as S2-C3
    fs_one = FeatureSource(LAYER, rows_of["test"][:1], mu, sd, "cuda")
    h1, _ = fs_one.batch([0], False)
    latency = {}
    for arm in args.arms:
        m = build(arm).to("cuda").eval()
        with torch.no_grad():
            sc, scs = time_callable(lambda: m(h1), repeat=30, warmup=5)
            tk, tks = time_callable(lambda: torch.topk(m(h1), 256, dim=-1),
                                    repeat=30, warmup=5)
        latency[arm] = dict(scorer_ms=sc, scorer_ms_std=scs,
                            topk_ms=tk, topk_ms_std=tks,
                            n_params=n_params(m))
    np.savez_compressed(os.path.join(OUT, f"{args.tag}_scores.npz"), **score_bank)
    np.savez_compressed(os.path.join(OUT, f"{args.tag}_perimage.npz"), **perimg)
    json.dump({"config": vars(args), "results": results, "latency": latency,
               "metric_columns": ["head_recall8", "head_recall16",
                                  "head_recall32", "head_agree8", "head_agree16",
                                  "head_agree32", "overlap256"],
               "keys": keys, "test_keys": test_keys, "layer": LAYER,
               "split_counts": {k: len(v) for k, v in rows_of.items()},
               "prefix_ms": meta["latency"][str(LAYER)]["prefix_ms_mean"]},
              open(os.path.join(OUT, f"{args.tag}_train.json"), "w"), indent=1)
    print(f"\n[saved] {args.tag}_scores.npz ({len(score_bank)} vectors) + "
          f"{args.tag}_perimage.npz + {args.tag}_train.json")


if __name__ == "__main__":
    main()
