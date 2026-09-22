"""
S2-C5 step 2: mechanism controls on the trained ladder.

Three questions, each with its own measurement.

1. Is a gain real, or seed noise?
   Every arm is trained with three seeds. For each seed the arm is compared to
   the *seed-matched* HEAD_RANK (so the comparison never mixes seed variance into
   the difference), and the seed-averaged arm is compared to the seed-averaged
   HEAD_RANK. Both are paired over the 150 held-out images with a 10 000-draw
   bootstrap CI. The seed spread of the arm itself is reported alongside, because
   a difference smaller than the spread is not a difference.

2. Is the context actually used?
   The honest pass and the wrong-image pass are compared three ways: as a
   headline metric difference, as a paired per-image bootstrap, and as the raw
   score displacement (max and mean absolute change, and the fraction of tokens
   whose score moves enough to cross the selection frontier). An arm whose wrong
   -image pass is indistinguishable from its honest pass on all three was not
   reading sample-specific configuration, whatever its headline number.

   The single seed used during training is not enough to make that call, so the
   derangement is redrawn K times here and the control is reported as a
   distribution.

3. If SET-CTX carries no gain, is it because the interaction term is switched
   off in practice?
   For SET-CTX the interaction contribution is measured directly: its norm
   relative to the residual stream it is added to, the normalised attention
   entropy, and the effective number of attended tokens exp(H). An interaction
   term that contributes ~0 % of the residual has collapsed to the LOCAL-MLP
   arm, and that is a mechanism finding, not a null result to be hidden.

Nothing here selects an arm. The verdict rule lives in ``s2c5_consolidate``.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                        # noqa: E402
from s2c1_train import FeatureSource, standardize_stats          # noqa: E402
from s2c3_common import BUDGET, LAYER, head_metrics, load_plan, teacher_orders  # noqa: E402
from s2c5_common import (BANKS, FROZEN_REF, bank_tag, dump, load_bank,  # noqa: E402
                         seed_mean_scores)
from s2c5_models import CONTEXTUAL, build                    # noqa: E402
from s2c5_train import context_bank, derangement, score_all   # noqa: E402

K_DERANGE = 5
COLS = ["head_recall8", "head_recall16", "head_recall32", "head_agree8",
        "head_agree16", "head_agree32", "overlap256"]


def bootstrap(d: np.ndarray, n_boot=10000, seed=0):
    """Paired bootstrap on the mean of per-image differences."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi),
                ci_excludes_zero=bool(lo > 0 or hi < 0), n=int(n))


def per_image(scores: dict, keys, orders, col="head_recall8"):
    return np.array([head_metrics(scores[k], orders[k])[col] for k in keys])


@torch.no_grad()
def attn_stats(model, h, h_other=None):
    """How much of the read-out actually comes from the token-token term.

    ``rel_norm`` is the interaction term's size relative to the residual it is
    added to. A term at ~0 % of the residual is a switched-off interaction, and
    the arm is LOCAL-MLP in disguise. ``self_vs_other`` compares the term's
    magnitude when the key/value set is the image's own against a different
    image's, which is the quantity the wrong-image control moves.
    """
    a = model.trunk(h)
    q, k = model.Q(a), model.K(a)
    att = torch.softmax(q @ k.transpose(-1, -2) * model.r ** -0.5, dim=-1)
    out = att @ model.Vv(a)
    rel = (out.norm(dim=-1) / (a.norm(dim=-1) + 1e-9))          # (B, N)
    ent = -(att.clamp_min(1e-12).log() * att).sum(-1)            # (B, N) nats
    top1 = att.amax(dim=-1)
    d = dict(
        residual_a_norm_mean=float(a.norm(dim=-1).mean()),
        rel_norm_mean=float(rel.mean()), rel_norm_median=float(rel.median()),
        entropy_nats_mean=float(ent.mean()),
        entropy_frac_of_max=float(ent.mean() / np.log(att.shape[-1])),
        eff_support_mean=float(ent.exp().mean()),
        top1_weight_mean=float(top1.mean()),
        n_tokens=int(att.shape[-1]),
    )
    # how much the interaction term moves across images (a collapse detector)
    o = out.norm(dim=-1).cpu().numpy()
    d["interaction_norm_across_images_std"] = float(o.std(axis=0).mean())
    if h_other is not None:
        a2 = model.trunk(h_other)
        out2 = model.attend(a, a2)
        d["self_vs_other_term_ratio"] = float(
            (out2.norm(dim=-1).mean() / (out.norm(dim=-1).mean() + 1e-9)))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c5")
    ap.add_argument("--arms", nargs="+",
                    default=["LOCAL-MLP", "GLOBAL-CTX", "SET-CTX",
                             "LOCAL-MLP-WIDE"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--derange", type=int, default=K_DERANGE)
    args = ap.parse_args()

    meta, plan, keys, rows_of = load_plan()
    G = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    orders = teacher_orders(G, keys)
    test_rows = rows_of["test"]
    test_keys = [keys[i] for i in test_rows]
    ds_test = [meta["plan"][i]["ds"] for i in test_rows]
    ds_all = [meta["plan"][i]["ds"] for i in range(len(keys))]

    Z = np.load(os.path.join(OUT, f"{args.tag}_scores.npz"))
    arm_scores = {}
    for arm in args.arms:
        for s in args.seeds:
            t = f"{arm}_s{s}"
            arm_scores[t] = {k.split("__", 1)[1]: Z[k].astype(np.float64)
                             for k in Z.files if k.startswith(t + "__")}

    # ---- frozen references --------------------------------------------------
    ref_per_seed = load_bank(FROZEN_REF)                     # {seed: {key: score}}
    # Two legitimate readings of "the reference", and they are not the same
    # number: the expected performance of one trained scorer (mean over seeds of
    # the per-image metric) and the performance of the seed-averaged scorer
    # (metric of the mean score vector). The arms are reported the first way --
    # mean over seeds of their per-image metric -- so every arm-vs-reference
    # comparison below uses ``ref_pi_mean``. ``ref_avg_pi`` is kept and reported
    # separately because S2-C4 quoted that reading.
    ref_pi_seed = np.stack([per_image(ref_per_seed[s], test_keys, orders)
                            for s in args.seeds])
    ref_pi_mean = ref_pi_seed.mean(0)
    ref_avg_pi = per_image(seed_mean_scores(ref_per_seed), test_keys, orders)
    lin_pi = per_image(seed_mean_scores(load_bank("LIN_L4")), test_keys, orders)

    # ---- standardisation + features for the context controls ----------------
    t0 = time.time()
    H = np.load(os.path.join(OUT, f"s2c1_feats_L{LAYER}.npy"), mmap_mode="r")
    mu, sd = standardize_stats(H, rows_of["fit"])
    del H
    print(f"[stats] {time.time() - t0:.0f} s")
    fs_test = FeatureSource(LAYER, test_rows, mu, sd, "cuda")
    fs_fit = FeatureSource(LAYER, rows_of["fit"], mu, sd, "cuda", cache=True)

    out = {"config": vars(args), "test_keys": test_keys, "ds_test": ds_test,
           "reference": FROZEN_REF, "arms": {}}

    for arm in args.arms:
        entry = {"per_seed": {}, "kind": None}
        pi_seeds = []
        for s in args.seeds:
            t = f"{arm}_s{s}"
            sc = arm_scores[t]
            pi = per_image(sc, test_keys, orders)
            pi_seeds.append(pi)
            ref_pi = per_image(ref_per_seed[s], test_keys, orders)
            d = {
                "head_recall8": float(pi.mean()),
                "head_recall16": float(per_image(sc, test_keys, orders,
                                                 "head_recall16").mean()),
                "head_recall32": float(per_image(sc, test_keys, orders,
                                                 "head_recall32").mean()),
                "head_agree8": float(per_image(sc, test_keys, orders,
                                               "head_agree8").mean()),
                "overlap256": float(per_image(sc, test_keys, orders,
                                              "overlap256").mean()),
            }
            entry["per_seed"][t] = {
                "test": d,
                "vs_HEAD_RANK_same_seed": {
                    "head_recall8": bootstrap(pi - ref_pi),
                },
            }
        pi_arr = np.stack(pi_seeds)                          # (seeds, n_test)
        pi_mean = pi_arr.mean(0)
        entry["seed_mean"] = {
            "head_recall8": float(pi_mean.mean()),
            "head_recall8_seed_sd": float(pi_arr.mean(1).std()),
            "per_seed_head_recall8": [float(x.mean()) for x in pi_arr],
        }
        entry["vs_HEAD_RANK_seedavg"] = {
            "head_recall8": bootstrap(pi_mean - ref_pi_mean),
            "head_recall8_abs_delta": float(pi_mean.mean() - ref_pi_mean.mean()),
        }
        entry["vs_HEAD_RANK_seedavg"]["per_seed_bootstrap"] = [
            bootstrap(pi_arr[i] - per_image(ref_per_seed[s], test_keys, orders))
            for i, s in enumerate(args.seeds)]

        # ---- per-benchmark -------------------------------------------------
        entry["by_benchmark"] = {}
        for d in ("TextVQA_VAL", "DocVQA_VAL", "OCRBench"):
            sel = [i for i, x in enumerate(ds_test) if x == d]
            entry["by_benchmark"][d] = {
                "n": len(sel),
                "head_recall8": float(pi_mean[sel].mean()),
                "ref_head_recall8": float(ref_pi_mean[sel].mean()),
                "delta": float(pi_mean[sel].mean() - ref_pi_mean[sel].mean()),
                "ci": bootstrap(pi_mean[sel] - ref_pi_mean[sel]),
            }

        # ---- contextual controls ------------------------------------------
        if arm in CONTEXTUAL:
            ctrls, attn = [], []
            for s in args.seeds:
                model = build(arm).to("cuda")
                model.load_state_dict(torch.load(
                    os.path.join(OUT, f"{args.tag}_{arm}_s{s}.pt"),
                    map_location="cuda", weights_only=False))
                model.eval()
                bank = context_bank(model, fs_test)
                honest = score_all(model, fs_test)
                draws = []
                for j in range(args.derange):
                    perm = derangement(len(test_keys), 1000 * s + j)
                    sw = score_all(model, fs_test, perm=perm, bank=bank)
                    pi_w = per_image(
                        {k: sw[i] for i, k in enumerate(test_keys)},
                        test_keys, orders)
                    draws.append({
                        "draw": j,
                        "head_recall8": float(pi_w.mean()),
                        "delta_vs_honest": float(pi_w.mean() - np.mean(
                            [head_metrics(honest[i], orders[k])["head_recall8"]
                             for i, k in enumerate(test_keys)])),
                        "paired_vs_honest": bootstrap(pi_w - per_image(
                            {k: honest[i] for i, k in enumerate(test_keys)},
                            test_keys, orders)),
                        "score_max_abs_diff": float(np.abs(sw - honest).max()),
                        "score_mean_abs_diff": float(np.abs(sw - honest).mean()),
                    })
                # does the substitution move tokens across the frontier at all?
                top_h = np.argsort(-honest, axis=1)[:, :BUDGET]
                top_w = np.argsort(-sw, axis=1)[:, :BUDGET]
                draws.append({
                    "top256_jaccard": float(np.mean(
                        [len(set(a) & set(b)) / len(set(a) | set(b))
                         for a, b in zip(top_h, top_w)]))})
                ctrls.append({"seed": s, "draws": draws,
                              "wrong_fit": _wrong_fit(arm, model, fs_test, fs_fit,
                                                      test_rows, test_keys, orders,
                                                      s)})
                if arm == "SET-CTX":
                    h1, _ = fs_test.batch(list(range(16)), False)
                    h2, _ = fs_test.batch(list(range(16, 32)), False)
                    attn.append(attn_stats(model, h1, h_other=h2))
                else:
                    attn.append(_global_ctx_stats(model, fs_test))
            entry["controls"] = ctrls
            entry["mechanism"] = attn
        out["arms"][arm] = entry
        print(f"[{arm}] seed-mean h8={entry['seed_mean']['head_recall8']:.4f} "
              f"(sd {entry['seed_mean']['head_recall8_seed_sd']:.4f}) "
              f"delta_vs_HEAD_RANK={entry['vs_HEAD_RANK_seedavg']['head_recall8_abs_delta']:+.4f} "
              f"CI=[{entry['vs_HEAD_RANK_seedavg']['head_recall8']['lo']:+.4f},"
              f"{entry['vs_HEAD_RANK_seedavg']['head_recall8']['hi']:+.4f}]",
              flush=True)

    out["reference_seed_mean_head_recall8"] = float(ref_pi_mean.mean())
    out["reference_seedavg_scorer_head_recall8"] = float(ref_avg_pi.mean())
    out["reference_per_seed_head_recall8"] = [
        float(per_image(ref_per_seed[s], test_keys, orders).mean())
        for s in args.seeds]
    out["LIN_L4_seed_mean_head_recall8"] = float(lin_pi.mean())
    dump(f"{args.tag}_controls.json", out)

    # ------------------------------------------------------------------ print -
    print("\n=== seed-averaged held-out head_recall@8, reference = HEAD_RANK ===")
    print(f"  {'arm':<16} {'h8':>7} {'seed_sd':>8} {'delta':>8} {'CI':>20}")
    print(f"  {'HEAD_RANK(ref)':<16} {ref_pi_mean.mean():7.4f} "
          f"{np.std(out['reference_per_seed_head_recall8']):8.4f} "
          f"{0.0:+8.4f} {'-':>20}")
    print(f"  {'LIN_L4':<16} {out['LIN_L4_seed_mean_head_recall8']:7.4f} "
          f"{'-':>8} {out['LIN_L4_seed_mean_head_recall8'] - ref_pi_mean.mean():+8.4f}")
    for arm in args.arms:
        e = out["arms"][arm]
        b = e["vs_HEAD_RANK_seedavg"]["head_recall8"]
        print(f"  {arm:<16} {e['seed_mean']['head_recall8']:7.4f} "
              f"{e['seed_mean']['head_recall8_seed_sd']:8.4f} "
              f"{b['mean']:+8.4f} [{b['lo']:+.4f},{b['hi']:+.4f}]")

    print("\n=== wrong-image context control (per arm, per seed, K draws) ===")
    for arm in args.arms:
        e = out["arms"][arm]
        if "controls" not in e:
            continue
        for c in e["controls"]:
            hon = e["per_seed"][f"{arm}_s{c['seed']}"]["test"]["head_recall8"]
            ds = [d for d in c["draws"] if "head_recall8" in d]
            print(f"  {arm:<14} s{c['seed']} honest={hon:.4f} "
                  f"wrong={np.mean([d['head_recall8'] for d in ds]):.4f} "
                  f"(delta {np.mean([d['delta_vs_honest'] for d in ds]):+.4f}, "
                  f"max|Δscore|={np.mean([d['score_max_abs_diff'] for d in ds]):.2e}, "
                  f"top256 Jaccard="
                  f"{np.mean([d['top256_jaccard'] for d in c['draws'] if 'top256_jaccard' in d]):.3f}) "
                  f"wrong-fit h8={c['wrong_fit']['head_recall8']:.4f}")

    print("\n=== mechanism read-out ===")
    for arm in args.arms:
        e = out["arms"][arm]
        if "mechanism" in e:
            print(f"  {arm}: {json.dumps(e['mechanism'][0], indent=None)}")


def _wrong_fit(arm, model, fs_test, fs_fit, test_rows, test_keys, orders, seed):
    """Substitute context drawn from fit images the scorer never evaluated on."""
    bank_fit = context_bank(model, fs_fit)
    perm = np.random.default_rng(seed + 991).integers(0, len(bank_fit),
                                                      size=len(test_keys))
    sw = score_all(model, fs_test, perm=perm, bank=bank_fit)
    return {c: float(np.mean([head_metrics(sw[i], orders[k])[c]
                              for i, k in enumerate(test_keys)]))
            for c in COLS}


def _global_ctx_stats(model, fs):
    """How much the context term actually varies from image to image.

    The decisive quantity is not how large the context term is, but how large its
    *image-to-image variation* is relative to the residual stream it is added to.
    If V(c) is nearly the same vector for every image, then GLOBAL-CTX differs
    from LOCAL-MLP only by a shared constant -- which LOCAL-MLP already carries in
    its own learnable bias -- and the arm has collapsed onto the local family
    however its headline number lands.
    """
    bank = context_bank(model, fs)
    with torch.no_grad():
        t = torch.from_numpy(bank).to("cuda")
        vc = model.V(t)                                     # (n, d_hid)
        h, _ = fs.batch(list(range(min(64, len(fs.rows)))), False)
        a = model.trunk(h)
    vc_np = vc.cpu().numpy()
    a_norm = float(a.norm(dim=-1).mean())
    # per-dimension spread of the context term across images
    vc_spread = float(vc_np.std(axis=0).mean())
    return {
        "ctx_dim": int(bank.shape[-1]),
        "residual_a_norm_mean": a_norm,
        "ctx_term_norm_mean": float(np.linalg.norm(vc_np, axis=1).mean()),
        "ctx_term_norm_std_across_images":
            float(np.linalg.norm(vc_np, axis=1).std()),
        "ctx_term_dim_spread_across_images": vc_spread,
        "ctx_variation_over_residual": vc_spread / (a_norm + 1e-9),
        "frac_ctx_dims_with_var": float((vc_np.std(axis=0) > 1e-6).mean()),
    }


if __name__ == "__main__":
    main()
