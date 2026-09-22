"""
M1 step 3: the two training protocols.

    --protocol fixed-step    PRIMARY. Batch 8, 900 optimizer updates for every n,
                             AdamW 1e-3 / 1e-4, constant LR (there is no
                             schedule), validation every 30 updates, NO early
                             halt. The fit pool is the only thing that varies.
    --protocol fixed-epoch   SECONDARY. S2-C6's own protocol, reproduced:
                             MAX_EPOCHS 80, halt at PATIENCE 20 epochs after the
                             best validation overlap, validation once per epoch.
                             Re-entangles steps with n by construction, which is
                             why it is the control and not the primary.

Both train the same arm on the same target with the same loss:

    arm     L4 -- the S2-C6 LOCAL-MLP reference, s = w2 . GELU(P h4 + b)
    target  HEAD_RANK -- positives = teacher Top-32, pair weights 32/(r+1)
    loss    balanced BCE + 0.5 * weighted all-pairs margin ranking, margin 1.0
    select  argmax over validation points of validation teacher Top-8 recall@256

The held-out 150 is never touched inside a training loop. Every checkpoint is
frozen first and the held-out set measured once, exactly as in S2-C6.

Reproduction gate G3 (prereg 3.3). The fixed-step protocol at n = 240 is a strict
prefix of S2-C6's n=240 trajectory: one n=240 epoch is 30 updates, which is
exactly one M1 validation interval, so 30(k+1) updates is C6's epoch-k evaluation
and the reshuffling cycle redraws its permutation at exactly the epochs C6 did.
``m1_train.py --protocol fixed-step --ns 240`` therefore has to reproduce
``s2c6_train_A.json``'s A240 arm -- validation history, selected H8 epoch and
per-seed held-out R@8 -- and says so rather than assuming it. A failure exits
non-zero and the rest of M1 is not run.

Usage
    python m1_train.py --protocol fixed-step  --ns 240            # gate G3
    python m1_train.py --protocol fixed-step  --ns 60 120 180 240 480 960
    python m1_train.py --protocol fixed-epoch --ns 240 480 960
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                              # noqa: E402
from s2c3_common import (LAYER, SEEDS, TEACHER, aggregate,            # noqa: E402
                         head_metrics, teacher_orders)
from s2c3_train import build_targets, compute_loss                    # noqa: E402
from s2c6_models import build, describe, n_params                     # noqa: E402
from m1_common import (ARM, BATCH, BS_EVAL, COL, EXTRA_TEACHER,       # noqa: E402
                       LR, M1_NS, MARGIN, MAX_EPOCHS, MAX_STEPS, NBOOT,
                       PATIENCE, PERIMAGE_COLS, TARGET_ARM, VAL_EVERY, WD,
                       FeatSource, cached_stats_l4, channel_stats_l4, compare,
                       dump, ladder_fit_rows, load_m1_plan)

SELECTIONS = ("H8", "OV")
SEL_KEY = {"H8": "head_recall8", "OV": "overlap256"}


# --------------------------------------------------------------- teacher maps
class CombinedTeacher:
    """s2b_gradient_scores.npz + the 720 extension maps, keyed as one archive."""

    def __init__(self, paths):
        self.maps = [np.load(p) for p in paths]
        self._owner = {}
        self.files = []
        for m in self.maps:
            for k in m.files:
                self._owner[k] = m
                self.files.append(k)

    def __getitem__(self, k):
        return self._owner[k][k]


# ------------------------------------------------------------- evaluation ----
@torch.no_grad()
def score_all(model, fs, bs=BS_EVAL):
    outs = []
    for a in range(0, len(fs.rows), bs):
        sel = list(range(a, min(a + bs, len(fs.rows))))
        xs = fs.batch(sel)
        outs.append(model(xs, None).float().cpu())
    return torch.cat(outs, 0).numpy()


def eval_split(model, fs, orders, keys):
    s = score_all(model, fs)
    per = [head_metrics(s[i], orders[k]) for i, k in enumerate(keys)]
    return s, aggregate(per), per


def per_image_col(per, col=COL):
    return np.array([p[col] for p in per], dtype=np.float64)


# --------------------------------------------------------------------- run ---
def run(protocol, seed, fit_rows, rows_of, keys, orders, stats, targets, dev,
        tag="m1"):
    """One protocol, one seed, one fit-set size. Returns (record, scores, perimage)."""
    P, Ng, W, PW, Y, pos_weight = targets
    fi = np.asarray(fit_rows)
    P_fit, N_fit, W_fit, Y_fit = P[fi], Ng[fi], W[fi], Y[fi]
    PW_fit = np.stack([PW[i] for i in fi])
    n_fit = len(fi)

    fs_fit = FeatSource(fi, stats, dev, cache=True)
    fs_val = FeatSource(rows_of["val"], stats, dev, cache=True)
    fs_test = FeatSource(rows_of["test"], stats, dev)
    val_keys = [keys[i] for i in rows_of["val"]]
    test_keys = [keys[i] for i in rows_of["test"]]

    # ---- RNG order identical to s2c5a_train.main / s2c6_train.run -----------
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = build(ARM).to(dev)
    np_ = n_params(model)

    model.eval()
    _, init_val, init_per = eval_split(model, fs_val, orders, val_keys)
    _, init_test, _ = eval_split(model, fs_test, orders, test_keys)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best = {s: dict(score=-np.inf, epoch=-1, step=-1, state=None, val=None)
            for s in SELECTIONS}
    hist, ep_loss, step, t_start = [], [], 0, time.time()

    def _step(sel):
        xs = fs_fit.batch(sel)
        s = model(xs, None)
        loss, bce, rk = compute_loss(
            s, torch.from_numpy(Y_fit[sel]).to(dev),
            torch.from_numpy(P_fit[sel]).to(dev),
            torch.from_numpy(N_fit[sel]).to(dev),
            torch.from_numpy(W_fit[sel]).to(dev),
            torch.from_numpy(PW_fit[sel]).to(dev), pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.detach())

    def _validate(point, step_now):
        model.eval()
        _, vm, _ = eval_split(model, fs_val, orders, val_keys)
        hist.append(dict(point=point, step=step_now, epoch=point,
                         loss=float(np.mean(ep_loss)) if ep_loss else 0.0, **vm))
        ep_loss.clear()
        for sl in SELECTIONS:
            sc = vm[SEL_KEY[sl]]
            if sc > best[sl]["score"]:
                best[sl] = dict(score=float(sc), epoch=point, step=step_now,
                                val=vm,
                                state={k: v.detach().clone()
                                       for k, v in model.state_dict().items()})
        return vm

    if protocol == "fixed-step":
        # One reshuffled queue, refilled whenever it empties. At every n here,
        # n_fit is divisible by BATCH, so the refill lands exactly on the epoch
        # boundaries S2-C6 drew its own permutation on.
        queue = []
        while step < MAX_STEPS:
            model.train()
            if not queue:
                queue = list(rng.permutation(n_fit))
            sel = queue[:BATCH]
            queue = queue[BATCH:]
            ep_loss.append(_step(sel))
            step += 1
            if step % VAL_EVERY == 0:
                _validate(step // VAL_EVERY - 1, step)
        epochs_run = len(hist)
        steps_run = step
    elif protocol == "fixed-epoch":
        for ep in range(MAX_EPOCHS):
            model.train()
            perm = rng.permutation(n_fit)
            for a in range(0, n_fit, BATCH):
                sel = perm[a:a + BATCH]
                ep_loss.append(_step(sel))
                step += 1
            _validate(ep, step)
            # halt rule unchanged from S2-C5 / S2-C6: overlap patience
            if ep - best["OV"]["epoch"] >= PATIENCE:
                break
        epochs_run = len(hist)
        steps_run = step
    else:
        raise KeyError(protocol)

    # ---- freeze each checkpoint, then measure it once ----------------------
    rec, scores, perimg = {}, {}, {}
    for sl in SELECTIONS:
        model.load_state_dict(best[sl]["state"])
        model.eval()
        s_test, test_m, test_per = eval_split(model, fs_test, orders, test_keys)
        _, train_m, _ = eval_split(model, fs_fit, orders,
                                   [keys[i] for i in fi])
        k = f"{tag}_{protocol}_n{n_fit}_{ARM}_s{seed}"
        for kk, v in zip(test_keys, s_test):
            scores[f"{k}__{sl}__{kk}"] = v.astype(np.float32)
        perimg[f"{sl}__honest"] = per_image_col(test_per)
        perimg[f"{sl}__full"] = np.stack(
            [[p[c] for c in PERIMAGE_COLS] for p in test_per]).astype(np.float32)
        torch.save({kk: v.cpu() for kk, v in best[sl]["state"].items()},
                   os.path.join(OUT, f"{k}__{sl}.pt"))
        rec[sl] = dict(selection=sl, criterion=SEL_KEY[sl],
                       best_point=int(best[sl]["epoch"]),
                       best_step=int(best[sl]["step"]),
                       n_val_points=len(hist),
                       val=best[sl]["val"], test=test_m, train=train_m)

    h8_stop = int(best["H8"]["epoch"]) + PATIENCE
    out = dict(protocol=protocol, arm=ARM, seed=seed, n_fit=n_fit,
               fit_rows=[int(i) for i in fi], n_params=np_,
               target=TARGET_ARM, layer=LAYER,
               init=dict(val=init_val, test=init_test),
               val_stats={"mu_mean": float(stats[0].mean()),
                          "sd_median": float(np.median(stats[1]))},
               train_seconds=float(time.time() - t_start),
               epochs_run=epochs_run, steps_run=steps_run,
               selections=rec,
               head_patience_halt_point=h8_stop,
               val_history=hist)
    return out, scores, perimg


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="fixed-step",
                    choices=["fixed-step", "fixed-epoch"])
    ap.add_argument("--ns", nargs="+", type=int, default=list(M1_NS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--tag", default="m1")
    ap.add_argument("--check-c6", action="store_true",
                    help="after training, assert the n=240 rung reproduces "
                         "s2c6_train_A.json's A240 arm and exit non-zero if not")
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    dev = "cuda"

    meta, plan, keys, rows_of = load_m1_plan()
    G = CombinedTeacher([os.path.join(OUT, TEACHER),
                         os.path.join(OUT, EXTRA_TEACHER)])
    orders = teacher_orders(G, keys)
    print(f"[split] combined={len(keys)} fit={len(rows_of['fit'])} "
          f"val={len(rows_of['val'])} test={len(rows_of['test'])} | "
          f"teacher keys={len(G.files)} | arm={ARM} target={TARGET_ARM}")
    print(f"[protocol] {args.protocol} ns={args.ns} seeds={args.seeds}")

    results, scores, perimg = {}, {}, {}
    for n in args.ns:
        fit_rows = ladder_fit_rows(rows_of, keys, n)
        t0 = time.time()
        stats = cached_stats_l4(n, fit_rows)
        print(f"[stats] n={n}: h4 mean/std over {n} fit rows "
              f"({time.time() - t0:.0f} s)", flush=True)
        targets = build_targets(TARGET_ARM, orders, keys)
        for seed in args.seeds:
            rec, sc, pi = run(args.protocol, seed, fit_rows, rows_of, keys,
                              orders, stats, targets, dev, tag=args.tag)
            k = f"{args.tag}_{args.protocol}_n{n}_{ARM}_s{seed}"
            results[k] = rec
            scores.update(sc)
            perimg.update({f"{k}__{kk}": v for kk, v in pi.items()})
            m = rec["selections"]
            print(f"[{k}] ep={m['H8']['best_point']}"
                  f"/{m['OV']['best_point']} of {rec['epochs_run']} "
                  f"({rec['steps_run']} steps) | "
                  f"init_h8={rec['init']['test']['head_recall8']:.4f} | "
                  + " ".join(f"{sl}: val={m[sl]['val'][COL]:.4f} "
                             f"test_h8={m[sl]['test']['head_recall8']:.4f} "
                             f"h16={m[sl]['test']['head_recall16']:.4f} "
                             f"ov={m[sl]['test']['overlap256']:.4f} "
                             f"train_h8={m[sl]['train']['head_recall8']:.4f}"
                             for sl in SELECTIONS)
                  + f" | {rec['train_seconds']:.0f}s", flush=True)

    np.savez_compressed(os.path.join(OUT, f"{args.tag}_scores_{args.protocol}.npz"),
                        **scores)
    np.savez_compressed(os.path.join(OUT, f"{args.tag}_perimage_{args.protocol}.npz"),
                        **perimg)
    dump(f"{args.tag}_train_{args.protocol}.json",
         dict(config=vars(args), results=results,
              metric_columns=list(PERIMAGE_COLS), layer=LAYER, arm=ARM,
              target=TARGET_ARM, keys=keys,
              test_keys=[keys[i] for i in rows_of["test"]],
              val_keys=[keys[i] for i in rows_of["val"]],
              split_counts={k: len(v) for k, v in rows_of.items()
                            if k != "extra"},
              seeds=list(args.seeds), margin=MARGIN, nboot=NBOOT,
              max_steps=MAX_STEPS, val_every=VAL_EVERY))

    # ---- gate G3, checked here so a failure stops the chain ----------------
    if args.check_c6:
        ref = json.load(open(os.path.join(OUT, "s2c6_train_A.json")))
        published = {0: 0.7992, 1: 0.8133, 2: 0.8017}
        worst, bad = 0.0, []
        for seed in args.seeds:
            r = ref["results"][f"A240__L4_s{seed}"]
            m = results[f"{args.tag}_fixed-step_n240_L4_s{seed}"]
            k = min(len(r["val_history"]), len(m["val_history"]))
            for i in range(k):
                for col in ("head_recall8", "head_recall16", "overlap256"):
                    worst = max(worst, abs(float(r["val_history"][i][col])
                                           - float(m["val_history"][i][col])))
            e_ok = r["selections"]["H8"]["best_epoch"] == \
                m["selections"]["H8"]["best_point"]
            h_m1 = m["selections"]["H8"]["test"]["head_recall8"]
            h_ok = abs(h_m1 - published[seed]) < 5e-5
            print(f"[gate G3 s{seed}] epochs compared={k} "
                  f"(C6 ran {r['epochs_run']}) | epoch_match={e_ok} "
                  f"| test R@8 m1={h_m1:.4f} published={published[seed]:.4f} "
                  f"match={h_ok}")
            if not (e_ok and h_ok):
                bad.append(seed)
        print(f"[gate G3] worst validation-curve |diff| = {worst:.3e}")
        dump(f"{args.tag}_g3.json",
             dict(worst_val_curve_abs_diff=worst, bad_seeds=bad,
                  published_per_seed=published,
                  n240_epochs_run={s: results[
                      f"{args.tag}_fixed-step_n240_L4_s{s}"]["epochs_run"]
                      for s in args.seeds},
                  c6_epochs_run={s: ref["results"][
                      f"A240__L4_s{s}"]["epochs_run"] for s in args.seeds}))
        if worst != 0.0 or bad:
            raise SystemExit(
                f"gate G3 FAILED (worst |diff|={worst:.3e}, seeds {bad}): the "
                f"fixed-step protocol is not a controlled re-parameterisation "
                f"of S2-C6's run. Stopping before the rest of the grid.")
        print("[gate G3] PASS -- fixed-step n=240 reproduces S2-C6 A240 exactly")


if __name__ == "__main__":
    main()
