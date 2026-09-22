"""
S2-C6 step 1: the three training ladders (Parts A, B, C).

Everything that is not named in the pre-registration as a degree of freedom is
the S2-C5 / S2-C5A configuration, verbatim. In particular the training loop below
draws its RNG in the same order as ``s2c5a_train.main`` and halts on the same rule
(overlap patience, ``PATIENCE = 20``), so the trajectory is the trajectory
S2-C5A audited; only the index into it that gets frozen changes.

Two checkpoints are tracked inside each trajectory:

  H8   argmax over epochs of validation teacher Top-8 recall@256   <- selected
  OV   argmax over epochs of validation Top-256 overlap            <- recorded

Every verdict in this stage is computed on H8. OV is carried so that each result
can be checked for selection-invariance; it is never used to select.

Also recorded, and used by nothing:

  INIT   the untrained model, scored on validation and held-out before the first
         gradient step. This says how much of an arm's number is architecture and
         initialization rather than learning, which matters here because the H8
         rule freezes an early checkpoint.

  WRONG-TRAJECTORY / WRONG-QUERY   held-out inference with the arm's second
         channel (or its query) replaced by another held-out image's, a
         derangement so no image keeps its own. Two variants for the dual arms,
         borrowing for channel 0 and for channel 1 separately: an arm that reads
         the *pairing* is hurt by both, an arm that merely reads one channel hard
         is hurt by exactly one. The honest pass is re-run through the same code
         path as a plumbing self-check.

Usage
    python s2c6_train.py --part A          # n in {60,120,180,240} x 3 seeds
    python s2c6_train.py --part B          # six arms x 3 seeds, full fit set
    python s2c6_train.py --part C          # L4+QUERY x 3 seeds
"""
import argparse
import os
import sys
import time

# CUBLAS_WORKSPACE_CONFIG must be set before CUDA is initialised (S2-C5A's
# convention). Deterministic kernels cost nothing for scorers this small and they
# make every number here exactly re-runnable -- which is what lets Part A's n=240
# point and Part B's L4 reference be asserted equal rather than argued equal.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                              # noqa: E402
from s2c3_common import (LAYER, SEEDS, TEACHER, aggregate,            # noqa: E402
                         head_metrics, load_plan, teacher_orders)
from s2c3_train import build_targets, compute_loss                    # noqa: E402
from s2c6_common import (BS_EVAL, BATCH, COL, LR, MAX_EPOCHS, NS,    # noqa: E402
                         PATIENCE, PERIMAGE_COLS, SELECTIONS, SEL_KEY,
                         TARGET_ARM, WD, ViewSource, channel_stats,
                         delta_noise_report, derangement, dump,
                         nested_fit_rows)
from s2c6_models import (ARMS, DUAL_ARMS, PART_A_ARM, PART_A_TAG,      # noqa: E402
                         build, describe, n_params)

TAG = "s2c6"


# ------------------------------------------------------------- evaluation ----
@torch.no_grad()
def score_all(model, fs, need_q, bs=BS_EVAL, ch_perm=None, q_perm=None,
              ch_which=1):
    outs = []
    for a in range(0, len(fs.rows), bs):
        sel = list(range(a, min(a + bs, len(fs.rows))))
        xs, q = fs.batch(sel, need_q, q_perm=q_perm, ch_perm=ch_perm,
                         ch_which=ch_which)
        outs.append(model(xs, q).float().cpu())
    return torch.cat(outs, 0).numpy()


def eval_split(model, fs, orders, keys, need_q=False, **kw):
    s = score_all(model, fs, need_q, **kw)
    per = [head_metrics(s[i], orders[k]) for i, k in enumerate(keys)]
    return s, aggregate(per), per


def per_image_col(per, col=COL):
    return np.array([p[col] for p in per], dtype=np.float64)


@torch.no_grad()
def query_scale(model, fs):
    """How large is the query term, and how much does it vary *across images*?

    A "the query adds nothing" conclusion only means something if the query path
    was doing something at the selected checkpoint -- because the H8 rule freezes
    an epoch 0-2 checkpoint and the query path starts at zero. Both are measured,
    on the validation split, at the untrained model and at each frozen checkpoint.

    ``across_image_ratio`` is the decisive quantity: the norm of the term after
    subtracting its across-image mean, divided by the norm of the term itself. A
    branch that has learned a per-image constant scores ~0 there -- it is doing
    something, but nothing that could condition on the question.
    """
    if not hasattr(model, "query_contribution"):
        return None
    xs, q = fs.batch(list(range(len(fs.rows))), True)
    z = model.query_contribution(xs, q)
    z = z.reshape(z.shape[0], -1)
    norm = z.norm(dim=-1)
    centred = (z - z.mean(0, keepdim=True)).norm(dim=-1)
    return {"query_term_norm_mean": float(norm.mean()),
            "query_term_norm_max": float(norm.max()),
            "across_image_norm_mean": float(centred.mean()),
            "across_image_ratio": float(centred.mean() / (norm.mean() + 1e-9))}


# --------------------------------------------------------------------- run ---
def run(args, arm, seed, fit_rows, rows_of, keys, orders, stats, targets, dev):
    """One arm, one seed, one fit-set size. Returns (record, scores, perimage).

    ``stats`` and ``targets`` are computed once per job by ``main``: they do not
    depend on the seed, and recomputing the fit-split mean/std per seed would be
    three times the I/O for the same numbers.
    """
    e = ARMS[arm]
    channels, need_q = e["channels"], e["q"]
    q_layer = 4 if need_q else None

    P, Ng, W, PW, Y, pos_weight = targets
    fi = np.asarray(fit_rows)
    P_fit, N_fit, W_fit, Y_fit = P[fi], Ng[fi], W[fi], Y[fi]
    PW_fit = np.stack([PW[i] for i in fi])

    fs_fit = ViewSource(channels, fi, stats, dev, cache=True, q_layer=q_layer)
    fs_val = ViewSource(channels, rows_of["val"], stats, dev, cache=True,
                        q_layer=q_layer)
    fs_test = ViewSource(channels, rows_of["test"], stats, dev, q_layer=q_layer)
    val_keys = [keys[i] for i in rows_of["val"]]
    test_keys = [keys[i] for i in rows_of["test"]]
    n_fit = len(fi)

    # ---- RNG order identical to s2c5a_train.main ---------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = build(arm).to(dev)
    np_ = n_params(model)

    # The query arms' fairness check: at step 0 the arm must compute exactly what
    # the L4 arm computes, so the query path is the only thing that can move the
    # number. Asserted, not assumed.
    init_identity = None
    if need_q:
        torch.manual_seed(seed)
        ref = build(PART_A_ARM).to(dev)
        torch.manual_seed(seed)
        model = build(arm).to(dev)          # rebuilt under the same seed
        x1 = [fs_val.channel(channels[0], [0])]
        init_identity = float(np.abs(
            (model(x1, fs_val.query([0])) - ref(x1)).detach().cpu().numpy()).max())
        del ref

    # ---- untrained diagnostic ----------------------------------------------
    model.eval()
    _, init_val, _ = eval_split(model, fs_val, orders, val_keys, need_q)
    _, init_test, _ = eval_split(model, fs_test, orders, test_keys, need_q)
    init_qscale = query_scale(model, fs_val)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    t_start = time.time()
    best = {s: dict(score=-np.inf, epoch=-1, state=None, val=None)
            for s in SELECTIONS}
    hist = []
    for ep in range(args.max_epochs):
        model.train()
        perm = rng.permutation(n_fit)
        ep_loss = []
        for a in range(0, n_fit, BATCH):
            sel = perm[a:a + BATCH]
            xs, q = fs_fit.batch(sel, need_q)
            s = model(xs, q)
            loss, bce, rk = compute_loss(
                s, torch.from_numpy(Y_fit[sel]).to(dev),
                torch.from_numpy(P_fit[sel]).to(dev),
                torch.from_numpy(N_fit[sel]).to(dev),
                torch.from_numpy(W_fit[sel]).to(dev),
                torch.from_numpy(PW_fit[sel]).to(dev), pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.detach()))
        model.eval()
        _, vm, _ = eval_split(model, fs_val, orders, val_keys, need_q)
        hist.append(dict(epoch=ep, loss=float(np.mean(ep_loss)), **vm))
        for sl in SELECTIONS:
            sc = vm[SEL_KEY[sl]]
            if sc > best[sl]["score"]:
                best[sl] = dict(score=float(sc), epoch=ep, val=vm,
                                state={k: v.detach().clone()
                                       for k, v in model.state_dict().items()})
        # halt rule unchanged from S2-C5: overlap patience
        if ep - best["OV"]["epoch"] >= PATIENCE:
            break
    epochs_run = len(hist)

    # ---- freeze each checkpoint, then measure it once ----------------------
    rec, scores, perimg = {}, {}, {}
    for sl in SELECTIONS:
        model.load_state_dict(best[sl]["state"])
        model.eval()
        s_test, test_m, test_per = eval_split(model, fs_test, orders, test_keys,
                                              need_q)
        _, train_m, _ = eval_split(model, fs_fit, orders,
                                   [keys[i] for i in fi], need_q)

        ctrl = {}
        if arm in DUAL_ARMS:
            # borrow another image's channel 1, and separately channel 0, so an
            # arm that reads the *pairing* is hurt by both substitutions and an
            # arm that only reads one channel hard is hurt by exactly one.
            perm_d = derangement(len(test_keys), seed)
            for tag_, cp in (("WRONG_TRAJ_ch1", perm_d),
                             ("WRONG_TRAJ_ch0", perm_d)):
                s_w = score_all(model, fs_test, need_q, ch_perm=cp,
                                ch_which=0 if tag_.endswith("ch0") else 1)
                ctrl[tag_] = aggregate([head_metrics(s_w[i], orders[k])
                                        for i, k in enumerate(test_keys)])
                perimg[f"{sl}__{tag_}"] = per_image_col(
                    [head_metrics(s_w[i], orders[k])
                     for i, k in enumerate(test_keys)])
            s_self = score_all(model, fs_test, need_q, ch_perm=np.arange(len(test_keys)))
            ctrl["SELF_CHECK_max_abs_diff"] = float(np.abs(s_self - s_test).max())
        if need_q:
            perm_q = derangement(len(test_keys), seed)
            s_w = score_all(model, fs_test, need_q, q_perm=perm_q)
            ctrl["WRONG_QUERY"] = aggregate([head_metrics(s_w[i], orders[k])
                                             for i, k in enumerate(test_keys)])
            # The unit-free form of the same control: how much does replacing the
            # query move the score at all? Zero here means the branch is inert
            # whatever its parameter count or its norm.
            ctrl["WRONG_QUERY_score_delta"] = dict(
                abs_mean=float(np.abs(s_w - s_test).mean()),
                abs_max=float(np.abs(s_w - s_test).max()))
            perimg[f"{sl}__WRONG_QUERY"] = per_image_col(
                [head_metrics(s_w[i], orders[k])
                 for i, k in enumerate(test_keys)])
            s_self = score_all(model, fs_test, need_q,
                               q_perm=np.arange(len(test_keys)))
            ctrl["SELF_CHECK_max_abs_diff"] = float(np.abs(s_self - s_test).max())

        tag = f"{arm}_s{seed}"
        for k, v in zip(test_keys, s_test):
            scores[f"{tag}__{sl}__{k}"] = v.astype(np.float32)
        perimg[f"{sl}__honest"] = per_image_col(test_per)
        perimg[f"{sl}__full"] = np.stack(
            [[p[c] for c in PERIMAGE_COLS] for p in test_per]).astype(np.float32)
        torch.save({k: v.cpu() for k, v in best[sl]["state"].items()},
                   os.path.join(OUT, f"{args.tag}_{tag}__{sl}.pt"))

        rec[sl] = dict(selection=sl, criterion=SEL_KEY[sl],
                       best_epoch=int(best[sl]["epoch"]), epochs_run=epochs_run,
                       val=best[sl]["val"], test=test_m, train=train_m,
                       query_scale_val=query_scale(model, fs_val),
                       controls=ctrl)

    h8_stop = int(best["H8"]["epoch"]) + PATIENCE
    out = dict(arm=arm, seed=seed, part=e["part"], control=bool(e["ctrl"]),
               channels=list(channels), n_params=np_,
               n_fit=n_fit, fit_rows=[int(i) for i in fi],
               n_query_params=(model.n_query_params()
                               if hasattr(model, "n_query_params") else 0),
               access=describe(arm, model), target=TARGET_ARM, layer=LAYER,
               init_identity_l4_query=init_identity,
               init=dict(val=init_val, test=init_test,
                         query_scale_val=init_qscale),
               val_stats={c: {"mu_mean": float(stats[c][0].mean()),
                              "sd_median": float(np.median(stats[c][1]))}
                          for c in stats},
               train_seconds=float(time.time() - t_start),
               epochs_run=epochs_run, selections=rec,
               head_patience_halt_epoch=h8_stop,
               head_patience_reached_within_run=bool(h8_stop <= epochs_run),
               val_history=hist)
    return out, scores, perimg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="B", choices=["A", "B", "C"])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--ns", nargs="+", type=int, default=list(NS))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()
    if args.arms is None:
        args.arms = ([PART_A_ARM] if args.part == "A"
                     else [a for a, e in ARMS.items()
                           if e["part"] == args.part and not e["ctrl"]]
                     + (["L4+L4"] if args.part == "B" else []))

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    dev = "cuda"

    meta, plan, keys, rows_of = load_plan()
    G = np.load(os.path.join(OUT, TEACHER))
    orders = teacher_orders(G, keys)
    print(f"[split] fit={len(rows_of['fit'])} val={len(rows_of['val'])} "
          f"test={len(rows_of['test'])} | target={TARGET_ARM} | part={args.part}")
    print(f"[arms] {args.arms}")

    # Part A trains the same arm at four fit-set sizes; everything else trains at
    # the full fit set.
    jobs = ([(PART_A_TAG.format(n=n), n, nested_fit_rows(rows_of, keys, n))
             for n in args.ns] if args.part == "A"
            else [("full", len(rows_of["fit"]), list(rows_of["fit"]))])

    # delta quantization check -- reported before any delta arm is trained
    if args.part == "B":
        mu_d, sd_d = channel_stats("delta", rows_of["fit"])
        rep = delta_noise_report(mu_d, sd_d)
        print("[delta] " + "  ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in rep.items() if k != "note"))
        dump(f"{args.tag}_delta_noise.json", rep)

    results, scores, perimg = {}, {}, {}
    for jt, n, fit_rows in jobs:
        # standardization follows the fit set the arm is allowed to see, so the
        # stats are per job (per n in Part A) and never inherited from n=240
        need = sorted({c for a in args.arms for c in ARMS[a]["channels"]})
        t0 = time.time()
        stats = {c: channel_stats(c, fit_rows) for c in need}
        print(f"[stats] {jt}: {need} over {n} fit rows ({time.time() - t0:.0f} s)",
              flush=True)
        targets = build_targets(TARGET_ARM, orders, keys)

        for arm in args.arms:
            for seed in args.seeds:
                rec, sc, pi = run(args, arm, seed, fit_rows, rows_of, keys,
                                  orders, stats, targets, dev)
                k = f"{jt}__{arm}_s{seed}"
                results[k] = rec
                scores.update(sc)
                perimg.update({f"{k}__{kk}": v for kk, v in pi.items()})
                m = rec["selections"]
                print(f"[{k}] fit={n} ep={m['H8']['best_epoch']}"
                      f"/{m['OV']['best_epoch']} of {rec['epochs_run']} | "
                      f"init_h8={rec['init']['test']['head_recall8']:.4f} | "
                      + " ".join(
                          f"{sl}: val={m[sl]['val'][COL]:.4f} "
                          f"test_h8={m[sl]['test']['head_recall8']:.4f} "
                          f"h16={m[sl]['test']['head_recall16']:.4f} "
                          f"ov={m[sl]['test']['overlap256']:.4f}"
                          for sl in SELECTIONS)
                      + (f" | ctrl=" +
                         ",".join(f"{c}={v['head_recall8']:.4f}"
                                  for c, v in m["H8"]["controls"].items()
                                  if isinstance(v, dict) and "head_recall8" in v)
                         + (" " + ",".join(f"{c}={v:.1e}" for c, v in
                                           m["H8"]["controls"].items()
                                           if not isinstance(v, dict))
                            if any(not isinstance(v, dict) for v in
                                   m["H8"]["controls"].values()) else "")
                         if m["H8"]["controls"] else "")
                      + f" | {rec['train_seconds']:.0f}s", flush=True)

    np.savez_compressed(os.path.join(OUT, f"{args.tag}_scores_{args.part}.npz"),
                        **scores)
    np.savez_compressed(os.path.join(OUT, f"{args.tag}_perimage_{args.part}.npz"),
                        **perimg)
    dump(f"{args.tag}_train_{args.part}.json",
         dict(config=vars(args), results=results,
              metric_columns=list(PERIMAGE_COLS), layer=LAYER,
              target=TARGET_ARM, keys=keys,
              test_keys=[keys[i] for i in rows_of["test"]],
              val_keys=[keys[i] for i in rows_of["val"]],
              split_counts={k: len(v) for k, v in rows_of.items()},
              seeds=list(args.seeds)))


if __name__ == "__main__":
    main()
