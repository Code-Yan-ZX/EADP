"""
S2-C5A step 1: the aggregation audit that explains 0.7783 (S2-C4) vs 0.7656 (S2-C5).

Both documents are the *same three S2-C3 HEAD_RANK checkpoints*, scored on the
same held-out 150 images with the same metric. Only the order of two operations
differs:

  PROTOCOL A  metric-then-mean   ("seed-specific metric, then average")
              for each seed s: m_s(i) = head_recall@8(score_s[i], teacher_i)
              report            mean_i mean_s m_s(i)
              -> the S2-C5 reference (and S2-C4's `trained_per_seed` means)

  PROTOCOL B  mean-then-metric   ("seed-averaged weight/direction, then re-score")
              score(i) = mean_s score_s(i)
              report   mean_i head_recall@8(score[i], teacher_i)
              -> the S2-C4 `HEAD_RANK_seedavg` arm (0.7783)

For a *linear* scorer the two differ only in where the mean is taken, and that is
not an identity: head_recall@8 is a nonlinear (rank/selection) functional, so
mean-of-metrics != metric-of-mean in general. This script measures the gap and
states it as the apples-to-apples number under each protocol.

PROTOCOL B is itself two protocols, and the difference between them is the
second half of the story. Writing the S2-C3 scorer as score_s = w_s . h + b_s
with w_s = ||w_s|| * w_hat_s:

  B_raw     mean of the raw weight vectors        (mean_s w_s) . h + mean_s b_s
            -> what ``s2c5_common.seed_mean_scores`` computes, since the cached
               per-seed score vectors *are* w_s . h + b_s
  B_unit    mean of the unit directions           (mean_s w_hat_s) . h
            -> what S2-C4's ``HEAD_RANK_seedavg`` arm computes

These are not the same vector, because ||w_s|| differs across seeds, so the raw
mean is a norm-weighted average of the three directions while the unit mean is
an unweighted one. A per-seed direction is only defined up to scale, so B_unit
is the aggregation that treats the three seeds symmetrically; S2-C4 is therefore
the protocol-correct one *for a direction comparison*, and B_raw is what the
cached score vectors silently implement.

The per-seed rankings are identical under every pathway (verified below:
reconstructing score_s from the checkpoint reproduces the cached vector exactly),
so PROTOCOL A is unaffected by this choice -- only the mean-then-metric number
moves.

No training happens here. Nothing is selected. The held-out 150 is used only to
measure, never to choose.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                          # noqa: E402
from s2c3_common import (BUDGET, LAYER, SEEDS, TEACHER,           # noqa: E402
                         head_metrics, load_plan, teacher_orders)
from s2c4_common import load_trained_direction                    # noqa: E402
from s2c5_common import dump                                      # noqa: E402

ARM = "HEAD_RANK"
COLS = ("head_recall8", "head_recall16", "head_recall32",
        "head_agree8", "overlap256")


def boot(d, n_boot=10000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c5a_aggregation")
    args = ap.parse_args()

    meta, plan, keys, rows_of = load_plan()
    G = np.load(os.path.join(OUT, TEACHER))
    orders = teacher_orders(G, keys)
    rows = rows_of["test"]
    test_keys = [keys[i] for i in rows]

    # ---- the three cached per-seed score vectors ----------------------------
    Z = np.load(os.path.join(OUT, "s2c3_scores.npz"))
    per_seed = {}
    for s in SEEDS:
        tag = f"{ARM}_s{s}"
        per_seed[s] = np.stack([Z[f"{tag}__{k}"].astype(np.float64)
                                for k in test_keys])          # (150, 1024)

    # ---- PROTOCOL A: per-seed metric, then average over seeds ---------------
    def per_image_metric(S: np.ndarray, col: str) -> np.ndarray:
        return np.array([head_metrics(S[i], orders[test_keys[i]])[col]
                         for i in range(len(test_keys))])

    A = {c: np.stack([per_image_metric(per_seed[s], c) for s in SEEDS])
         for c in COLS}                                        # (n_seeds, 150)
    A_mean = {c: A[c].mean(0) for c in COLS}                   # (150,)
    A_report = {c: dict(seed_mean=float(A_mean[c].mean()),
                        per_seed=[float(x) for x in A[c].mean(1)],
                        seed_sd=float(A[c].mean(1).std()))
                for c in COLS}

    # ---- PROTOCOL B (scores): average the score vectors, then score ---------
    S_mean = np.mean([per_seed[s] for s in SEEDS], axis=0)     # (150, 1024)
    B_scores = {c: per_image_metric(S_mean, c) for c in COLS}
    B_scores_report = {c: float(B_scores[c].mean()) for c in COLS}

    # ---- the two mean-then-metric routes ------------------------------------
    import torch

    H = np.load(os.path.join(OUT, f"s2c1_feats_L{LAYER}.npy"), mmap_mode="r")
    # reuse the exact S2-C1 preprocessing: fit-split mean/std over all tokens
    s1 = np.zeros(H.shape[-1], dtype=np.float64)
    s2 = np.zeros(H.shape[-1], dtype=np.float64)
    n = 0
    for j in rows_of["fit"]:
        x = H[j].astype(np.float32)
        s1 += x.sum(axis=0, dtype=np.float64)
        s2 += np.square(x, dtype=np.float64).sum(axis=0)
        n += x.shape[0]
    mu = (s1 / n).astype(np.float32)
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)).astype(np.float32)

    W_raw, B_raw_vec = [], []
    for s in SEEDS:
        ck = torch.load(os.path.join(OUT, f"s2c3_{ARM}_s{s}.pt"),
                        map_location="cpu", weights_only=False)
        W_raw.append(ck["out.weight"].detach().cpu().numpy()
                     .reshape(-1).astype(np.float64))
        B_raw_vec.append(ck["out.bias"].detach().cpu().numpy()
                         .reshape(-1).astype(np.float64))
    W_raw = np.stack(W_raw)                                    # (n_seeds, 4096)
    B_raw_vec = np.stack(B_raw_vec)                            # (n_seeds, 1)
    W_unit = W_raw / np.linalg.norm(W_raw, axis=1, keepdims=True)

    Hte = np.stack([(H[j].astype(np.float32) - mu) / sd for j in rows]).astype(
        np.float64)                                            # (150, 1024, 4096)

    def route(Wm: np.ndarray, b: np.ndarray | None) -> np.ndarray:
        S = np.einsum("ntd,d->nt", Hte, Wm)
        return S if b is None else S + b.reshape(-1, 1)

    B_unit = {c: per_image_metric(route(W_unit.mean(0), None), c) for c in COLS}
    B_unit_report = {c: float(B_unit[c].mean()) for c in COLS}
    B_rawdir = {c: per_image_metric(route(W_raw.mean(0), B_raw_vec.mean(0)), c)
                for c in COLS}
    B_rawdir_report = {c: float(B_rawdir[c].mean()) for c in COLS}

    # every per-seed reconstruction must reproduce the cached score vector
    max_recon = 0.0
    for i, s in enumerate(SEEDS):
        S_s = route(W_raw[i], B_raw_vec[i])
        max_recon = max(max_recon, float(np.abs(S_s - per_seed[s]).max()))

    # ---- the gap, per column, as a paired difference over images -----------
    gap = {}
    for c in COLS:
        gap[c] = dict(
            A_metric_then_mean=A_report[c]["seed_mean"],
            B_raw_mean_of_weights=B_scores_report[c],
            B_rawdir_mean_of_weights_rescored=B_rawdir_report[c],
            B_unit_mean_of_unit_directions=B_unit_report[c],
            delta_A_minus_B_raw=float((A_mean[c] - B_scores[c]).mean()),
            paired_A_vs_B_raw=boot(A_mean[c] - B_scores[c]),
            delta_B_raw_minus_B_unit=float((B_scores[c] - B_unit[c]).mean()),
            paired_B_raw_vs_B_unit=boot(B_scores[c] - B_unit[c]),
            delta_A_minus_B_unit=float((A_mean[c] - B_unit[c]).mean()),
            paired_A_vs_B_unit=boot(A_mean[c] - B_unit[c]))

    # ---- what S2-C4 reported vs what S2-C3 reported ------------------------
    C4 = json.load(open(os.path.join(OUT, "s2c4_direction.json")))
    C5 = json.load(open(os.path.join(OUT, "s2c5_factorization.json")))
    C5c = json.load(open(os.path.join(OUT, "s2c5_controls.json")))

    # S2-C5's own controls file kept both readings and named them. It used the
    # metric-then-mean one for every arm-vs-reference comparison, so S2-C5 never
    # mixed protocols -- assert that rather than trust the docstring.
    c5_controls_refs = dict(
        metric_then_mean=float(C5c["reference_seed_mean_head_recall8"]),
        mean_of_raw_score_vectors=float(
            C5c["reference_seedavg_scorer_head_recall8"]),
        per_seed=[float(x) for x in C5c["reference_per_seed_head_recall8"]])
    assert abs(c5_controls_refs["metric_then_mean"]
               - A_report["head_recall8"]["seed_mean"]) < 1e-9
    assert abs(c5_controls_refs["mean_of_raw_score_vectors"]
               - B_scores_report["head_recall8"]) < 1e-9

    # ---- cross-check the cached per-seed values against both documents -----
    c4_per_seed = {r["arm"]: r["head_recall8"] for r in C4["trained_per_seed"]}
    assert max(abs(c4_per_seed[f"{ARM}_s{s}"] - A_report["head_recall8"]["per_seed"]
                   [i]) for i, s in enumerate(SEEDS)) < 1e-9, \
        "s2c4 trained_per_seed disagrees with the cached score vectors"
    assert abs(C5["references"][ARM] - A_report["head_recall8"]["seed_mean"]) < 1e-9, \
        "s2c5 reference disagrees with the cached score vectors"

    out = dict(
        stage="S2-C5A",
        question="are S2-C4's HEAD_RANK = 0.7783 and S2-C5's HEAD_RANK = 0.7656 "
                 "the same checkpoints under different aggregation?",
        answer="yes -- same three S2-C3 HEAD_RANK checkpoints, same held-out 150, "
               "same metric; they differ only in whether the seed average is taken "
               "before or after the rank metric",
        arm=ARM, n_seeds=len(SEEDS), n_test=len(test_keys), metric="head_recall8",
        protocol_A=dict(
            name="metric-then-mean",
            definition="mean_i mean_s head_recall@8(score_{s,i}, teacher_i)",
            where_used="S2-C5 reference and rule_inputs; S2-C4 trained_per_seed means",
            report=A_report),
        protocol_B=dict(
            name="mean-then-metric",
            definition="mean_i head_recall@8(mean_s score_{s,i}, teacher_i)",
            where_used="S2-C4 BEST_TRAINED / arms table (HEAD_RANK_seedavg)",
            B_raw_mean_of_weights=B_scores_report,
            B_rawdir_mean_of_weights_rescored=B_rawdir_report,
            B_unit_mean_of_unit_directions=B_unit_report,
            weight_norms=[float(x) for x in np.linalg.norm(W_raw, axis=1)],
            max_abs_recon_error_per_seed=max_recon,
            note="B_raw == B_rawdir to float precision (the cached score vectors "
                 "are exactly w_s.h + b_s, verified by reconstruction); B_unit is "
                 "the different, scale-symmetric aggregation S2-C4 actually used"),
        gap=gap,
        headline=dict(
            s2c4_quoted=float(C4["rule_inputs"]["BEST_TRAINED_head_recall8"]),
            s2c4_quoted_protocol="B_unit (metric on the mean of the unit "
                                 "directions)",
            s2c5_quoted=float(C5["references"][ARM]),
            s2c5_quoted_protocol="A (metric-then-mean)",
            s2c3_quoted=float(C4["references"]["s2c3_best_shared_head_recall8"]),
            reproduced_A=float(A_report["head_recall8"]["seed_mean"]),
            reproduced_B_unit=float(B_unit_report["head_recall8"]),
            reproduced_B_raw=float(B_scores_report["head_recall8"]),
            total_gap_A_minus_B_unit=float(
                A_report["head_recall8"]["seed_mean"]
                - B_unit_report["head_recall8"]),
            gap_part_metric_order=float(
                A_report["head_recall8"]["seed_mean"]
                - B_scores_report["head_recall8"]),
            gap_part_unit_normalisation=float(
                B_scores_report["head_recall8"]
                - B_unit_report["head_recall8"]),
        ),
        protocol_mixing=dict(
            s2c5_used="metric-then-mean for arms AND for the reference, "
                      "consistently -- verified against s2c5_controls.json, "
                      "which recorded both readings and compared on this one",
            s2c5_controls_reported=c5_controls_refs,
            s2c4_used="metric of the 3-seed mean unit direction",
            mixed=False,
            note="the two stages are each internally consistent; they simply "
                 "answer different questions and their headline HEAD_RANK "
                 "numbers are therefore not comparable to each other"),
        sources=dict(scores="s2c3_scores.npz",
                     refs=["s2c4_direction.json", "s2c5_factorization.json",
                           "s2c5_controls.json"]),
    )
    dump(f"{args.tag}.json", out)

    print(f"\nHEAD_RANK, held-out 150, {len(SEEDS)} seeds -- "
          f"same checkpoints, two aggregations")
    print(f"  per-seed head_recall@8 : "
          f"{[round(x, 4) for x in A_report['head_recall8']['per_seed']]}")
    print(f"  weight norms per seed  : "
          f"{[round(float(x), 4) for x in np.linalg.norm(W_raw, axis=1)]}")
    print(f"  per-seed reconstruction of the cached vectors: max abs err "
          f"{max_recon:.2e}")
    print(f"\n  A     metric-then-mean        = "
          f"{A_report['head_recall8']['seed_mean']:.4f}"
          f"   (S2-C5 quotes {C5['references'][ARM]:.4f})")
    print(f"  B_raw mean of raw weights     = "
          f"{B_scores_report['head_recall8']:.4f}")
    print(f"        (rescored: same thing)  = "
          f"{B_rawdir_report['head_recall8']:.4f}")
    print(f"  B_unit mean of unit dirs      = "
          f"{B_unit_report['head_recall8']:.4f}"
          f"   (S2-C4 quotes "
          f"{C4['rule_inputs']['BEST_TRAINED_head_recall8']:.4f})")
    g = gap["head_recall8"]
    print(f"\n  decomposition of the gap:")
    print(f"    metric order  A - B_raw        = {g['delta_A_minus_B_raw']:+.4f}  "
          f"CI [{g['paired_A_vs_B_raw']['lo']:+.4f},"
          f" {g['paired_A_vs_B_raw']['hi']:+.4f}]")
    print(f"    unit norming  B_raw - B_unit   = "
          f"{g['delta_B_raw_minus_B_unit']:+.4f}  "
          f"CI [{g['paired_B_raw_vs_B_unit']['lo']:+.4f},"
          f" {g['paired_B_raw_vs_B_unit']['hi']:+.4f}]")
    print(f"    total         A - B_unit       = {g['delta_A_minus_B_unit']:+.4f}  "
          f"CI [{g['paired_A_vs_B_unit']['lo']:+.4f},"
          f" {g['paired_A_vs_B_unit']['hi']:+.4f}]")
    print(f"\n  all three are protocol-correct for their own question; they are "
          f"not\n  interchangeable, and no aggregation here mixes them.")


if __name__ == "__main__":
    main()
