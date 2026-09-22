"""
S2-C5A step 2: checkpoint selection audit -- the same trajectory, two rules.

S2-C5 selected every checkpoint on validation Top-256 *overlap*, but its primary
scientific metric is held-out teacher Top-8 *recall@256*. S2-C2 and S2-C3 both
recorded that those two quantities are in tension: HEAD_RANK reaches the best
head recall at an overlap *below* BASE's. So a checkpoint chosen by overlap is
not a priori the checkpoint that maximises head recall, and the ~0.795 landing
point of S2-C5 cannot be read as a family ceiling until that is measured.

What this script does
---------------------
It re-runs the S2-C5 ladder with the training loop byte-identical to
``s2c5_train.main`` -- same RNG draws in the same order, same optimizer, same
batch order, same epoch budget and patience -- and tracks *two* checkpoints
inside the single resulting trajectory:

  OV    argmax over epochs of validation overlap256   (what S2-C5 used)
  H8    argmax over epochs of validation head_recall8 (what the primary metric
        says should be selected)

Because both come from the same trajectory, the comparison is exactly paired:
the weights, the data order and the epoch budget are held fixed, and only the
index into the trajectory changes (and with it the reported held-out number).

Why the training loop is not re-stopped
---------------------------------------
"Early stopping on head recall" in the strict sense would also move the halt
point. It cannot move the *selected* checkpoint here: val head_recall@8 peaks at
epoch 0-2 in every arm/seed of the published run while the runs halt at epoch
22-27, so best_h8_epoch + PATIENCE <= epochs_run holds for every run, and a
head-patience halt would have returned the same argmax. The audit therefore keeps
the original halt (identical trajectory) and records the head-stop epoch
separately so the claim is checkable rather than asserted.

Validation only
---------------
Selection reads the 60-image validation split and nothing else. The held-out 150
is touched only after both checkpoints are frozen, to measure them. No held-out
quantity is ever computed inside the epoch loop -- that is precisely the leak
this audit exists to rule out.

Nothing else changes: no architecture, no feature, no new hyperparameter, no
re-tuning. ``--arms``/``--seeds`` default to the full S2-C5 ladder.
"""
import argparse
import json
import os
import sys
import time

# CUBLAS_WORKSPACE_CONFIG must be set before CUDA is initialised, so it is set
# here rather than in the shell wrapper. Deterministic kernels cost nothing for
# scorers this small and they make this audit exactly re-runnable -- a property
# the original S2-C5 run did not have (its trajectory drifts at ~1e-8 per epoch
# from cuBLAS reduction order). The flag changes only reduction order, never the
# arithmetic being performed.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                          # noqa: E402
from s2c1_train import FeatureSource, standardize_stats            # noqa: E402
from s2c3_common import (LAYER, SEEDS, TEACHER, aggregate,        # noqa: E402
                         head_metrics, load_plan, teacher_orders)
from s2c3_train import build_targets, compute_loss                # noqa: E402
from s2c5_models import ARMS, CONTEXTUAL, build, n_params         # noqa: E402
import s2c5_train as T5                                           # noqa: E402

TAG = "s2c5a"
OV, H8 = "OV", "H8"
SELECTIONS = (OV, H8)
SEL_KEY = {OV: "overlap256", H8: "head_recall8"}
PERIMAGE_COLS = ("head_recall8", "head_recall16", "head_recall32",
                 "head_agree8", "head_agree16", "head_agree32", "overlap256")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--max-epochs", type=int, default=T5.MAX_EPOCHS)
    ap.add_argument("--deterministic", type=int, default=1,
                    help="1 = force deterministic kernels (default), 0 = leave "
                         "the S2-C5 default reduction order in place")
    args = ap.parse_args()

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        print("[repro] deterministic kernels ON")

    meta, plan, keys, rows_of = load_plan()
    G = np.load(os.path.join(OUT, TEACHER))
    orders = teacher_orders(G, keys)
    print(f"[split] fit={len(rows_of['fit'])} val={len(rows_of['val'])} "
          f"test={len(rows_of['test'])} | target={T5.TARGET_ARM} layer=L{LAYER}")

    t0 = time.time()
    H = np.load(os.path.join(OUT, f"s2c1_feats_L{LAYER}.npy"), mmap_mode="r")
    mu, sd = standardize_stats(H, rows_of["fit"])
    del H
    print(f"[stats] fit-split mean/std ({time.time() - t0:.0f} s)")

    P, Ng, W, PW, Y, pos_weight = build_targets(T5.TARGET_ARM, orders, keys)
    P_fit, N_fit, W_fit, Y_fit = (P[rows_of["fit"]], Ng[rows_of["fit"]],
                                  W[rows_of["fit"]], Y[rows_of["fit"]])
    PW_fit = np.stack([PW[i] for i in rows_of["fit"]])

    fs_fit = FeatureSource(LAYER, rows_of["fit"], mu, sd, "cuda", cache=True)
    fs_val = FeatureSource(LAYER, rows_of["val"], mu, sd, "cuda", cache=True)
    fs_test = FeatureSource(LAYER, rows_of["test"], mu, sd, "cuda")
    val_keys = [keys[i] for i in rows_of["val"]]
    test_keys = [keys[i] for i in rows_of["test"]]
    n_fit = len(rows_of["fit"])

    results, score_bank, perimg = {}, {}, {}
    for arm in args.arms:
        model0 = build(arm)
        print(f"\n=== {arm}: {model0.describe()}  params={n_params(model0):,}")
        del model0

        for seed in args.seeds:
            # ---- RNG order identical to s2c5_train.main --------------------
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            model = build(arm).to("cuda")
            np_ = n_params(model)
            opt = torch.optim.AdamW(model.parameters(), lr=T5.LR,
                                    weight_decay=T5.WD)
            t_start = time.time()
            # two trackers over one trajectory; identical init by construction
            best = {s: dict(score=-np.inf, epoch=-1, state=None, val=None)
                    for s in SELECTIONS}
            hist = []
            for ep in range(args.max_epochs):
                model.train()
                perm = rng.permutation(n_fit)
                ep_loss = []
                for a in range(0, n_fit, T5.BATCH):
                    sel = perm[a:a + T5.BATCH]
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
                _, vm, _ = T5.eval_split(model, fs_val, orders, val_keys)
                hist.append(dict(epoch=ep, loss=float(np.mean(ep_loss)), **vm))
                for sl in SELECTIONS:
                    sc = vm[SEL_KEY[sl]]
                    if sc > best[sl]["score"]:
                        best[sl] = dict(
                            score=float(sc), epoch=ep, val=vm,
                            state={k: v.detach().clone()
                                   for k, v in model.state_dict().items()})
                # halt rule unchanged from S2-C5: overlap patience
                if ep - best[OV]["epoch"] >= T5.PATIENCE:
                    break
            epochs_run = len(hist)

            # ---- freeze both checkpoints, then measure each once ------------
            rec = {}
            for sl in SELECTIONS:
                model.load_state_dict(best[sl]["state"])
                model.eval()
                s_test, test_m, test_per = T5.eval_split(
                    model, fs_test, orders, test_keys)
                _, train_m, _ = T5.eval_split(model, fs_fit, orders,
                                              [keys[i] for i in rows_of["fit"]])

                # Mechanism controls. Same construction as s2c5_train, except
                # that the per-image head_recall@8 is kept for the honest and
                # the substituted passes, so the S2-C5 cascade's USES_CTX test
                # can be re-applied here with its own paired CI instead of being
                # inherited from the published aggregates.
                ctrl, ctx_per_image = {}, {}
                if arm in CONTEXTUAL:
                    bank = T5.context_bank(model, fs_test)
                    perm = T5.derangement(len(test_keys), seed)
                    s_wrong = T5.score_all(model, fs_test, perm=perm, bank=bank)
                    w_per = np.array([head_metrics(s_wrong[i], orders[k])
                                      ["head_recall8"]
                                      for i, k in enumerate(test_keys)])
                    ctrl["WRONG_HELDOUT"] = aggregate(
                        [head_metrics(s_wrong[i], orders[k])
                         for i, k in enumerate(test_keys)])

                    bank_fit = T5.context_bank(model, fs_fit)
                    perm_fit = np.random.default_rng(seed + 991).integers(
                        0, len(bank_fit), size=len(test_keys))
                    s_fit = T5.score_all(model, fs_test, perm=perm_fit,
                                         bank=bank_fit)
                    f_per = np.array([head_metrics(s_fit[i], orders[k])
                                      ["head_recall8"]
                                      for i, k in enumerate(test_keys)])
                    ctrl["WRONG_FIT"] = aggregate(
                        [head_metrics(s_fit[i], orders[k])
                         for i, k in enumerate(test_keys)])

                    s_self = T5.score_all(model, fs_test,
                                          perm=np.arange(len(test_keys)),
                                          bank=bank)
                    ctrl["SELF_CHECK_max_abs_diff"] = float(
                        np.abs(s_self - s_test).max())
                    ctx_per_image = dict(
                        honest=np.array([p["head_recall8"] for p in test_per],
                                        dtype=np.float32),
                        wrong_helldout=w_per.astype(np.float32),
                        wrong_fit=f_per.astype(np.float32))

                tag = f"{arm}_s{seed}__{sl}"
                for k, v in zip(test_keys, s_test):
                    score_bank[f"{tag}__{k}"] = v.astype(np.float32)
                perimg[tag] = np.stack([
                    [p[c] for c in PERIMAGE_COLS] for p in test_per]
                ).astype(np.float32)
                torch.save({k: v.cpu() for k, v in
                            best[sl]["state"].items()},
                           os.path.join(OUT, f"{args.tag}_{tag}.pt"))

                rec[sl] = dict(
                    selection=sl, criterion=SEL_KEY[sl],
                    best_epoch=int(best[sl]["epoch"]),
                    epochs_run=epochs_run,
                    val=best[sl]["val"], test=test_m, train=train_m,
                    controls=ctrl)
                for cname, arr in ctx_per_image.items():
                    perimg[f"{tag}__ctx_{cname}"] = arr

            # head-patience halt point, recorded to check the claim that the
            # stop rule could not have changed the H8 selection
            h8_stop = int(best[H8]["epoch"]) + T5.PATIENCE
            both_seen = bool(h8_stop <= epochs_run)

            results[f"{arm}_s{seed}"] = dict(
                arm=arm, seed=seed, kind=model.kind, n_params=np_,
                access=model.describe(), target=T5.TARGET_ARM, layer=LAYER,
                train_seconds=float(time.time() - t_start),
                epochs_run=epochs_run, selections=rec,
                head_patience_halt_epoch=h8_stop,
                head_patience_reached_within_run=both_seen,
                val_history=hist,
            )
            print(f"[{arm}_s{seed}] {epochs_run} epochs | "
                  + " | ".join(
                      f"{sl} ep={rec[sl]['best_epoch']} "
                      f"val{sl}={best[sl]['score']:.4f} "
                      f"test_h8={rec[sl]['test']['head_recall8']:.4f} "
                      f"test_ov={rec[sl]['test']['overlap256']:.4f}"
                      for sl in SELECTIONS)
                  + f" | head-stop@{h8_stop} "
                  f"{'ok' if both_seen else 'NOT REACHED'}"
                  + f" | {time.time() - t_start:.0f}s", flush=True)

    np.savez_compressed(os.path.join(OUT, f"{args.tag}_scores.npz"), **score_bank)
    np.savez_compressed(os.path.join(OUT, f"{args.tag}_perimage.npz"), **perimg)
    json.dump({"config": vars(args), "results": results,
               "metric_columns": list(PERIMAGE_COLS),
               "selections": {OV: SEL_KEY[OV], H8: SEL_KEY[H8]},
               "keys": keys, "test_keys": test_keys, "layer": LAYER,
               "split_counts": {k: len(v) for k, v in rows_of.items()}},
              open(os.path.join(OUT, f"{args.tag}_train.json"), "w"), indent=1)
    print(f"\n[saved] {args.tag}_train.json + {args.tag}_perimage.npz "
          f"+ {args.tag}_scores.npz")


if __name__ == "__main__":
    main()
