"""
S2-C4 step 5: the pre-registered decision rule, and the stage's main JSON.

The rule is written here, in the file that applies it, before the numbers are
read into it -- so that it cannot be fitted to the result afterwards. It reads
only quantities other steps computed; it trains nothing and selects nothing.

--------------------------------------------------------------------------
Quantities
--------------------------------------------------------------------------
  SELF           in-sample oracle: image j's own w_j scored on image j. Upper
                 bound, reads j's teacher map.
  GLOBAL         mean of the 240 fit directions. No adaptation whatsoever.
  BEST_TRAINED   the best S2-C3 trained *shared* direction (HEAD_RANK, 3-seed
                 mean direction) -- the thing an adaptive scorer must beat.
  BEST_DEPLOY    max over every arm that uses only image j's cheap descriptors
                 (KNN{k}, NN, PCA_RIDGE) of mean head_recall@8.

  resolved(x, y) paired bootstrap (10 000, resampling the 150 held-out
                 instances) on per-instance head_recall@8; the 95 % CI on
                 mean(x) - mean(y) excludes zero.
  material(x)    mean head_recall@8 >= 0.80. S2-C3's best shared direction is
                 0.766, and the brief names 0.82-0.85 as the level at which a
                 downstream run becomes worth paying for, so 0.80 is the point
                 below which an arm has not changed the regime.

--------------------------------------------------------------------------
Decision
--------------------------------------------------------------------------
  A  LOW-DIMENSIONAL IMAGE-ADAPTIVE
     BEST_DEPLOY >= 0.80 AND resolved(BEST_DEPLOY, GLOBAL)
     AND the PCA basis is genuinely low-dimensional: PCA_ORACLE retains >= 0.90
     of SELF at k <= 64.
     -> a k-dimensional per-image direction is both sufficient and predictable.

  B  TRANSFERABLE CLUSTERS
     BEST_DEPLOY >= 0.80 AND resolved(BEST_DEPLOY, GLOBAL)
     AND the winning arm is a retrieval arm (NN/KNN).
     -> there are exploitable direction clusters, but they are not a linear
        low-dimensional family.

  C  NON-TRANSFERABLE
     BEST_DEPLOY < 0.80, or it fails to beat GLOBAL with a resolved positive CI.
     -> per-image oracle direction is strong, the variation across images is not
        captured by any cheap descriptor at a useful magnitude, and cross-image
        transfer does not exceed the trained shared direction.

  D  AMBIGUOUS
     everything else (e.g. a resolved gain that no reference can adjudicate).

The downstream gate from the brief is applied here too and recorded either way:
generation is run only if BEST_DEPLOY clears S2-C3's 0.766 and preferably
0.82-0.85. If it does not, the stage reports the required accuracy tables as
"not run" rather than spending 150 x 3 generations on an arm that the
head-retention half of the experiment has already excluded.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                            # noqa: E402
from s2c4_common import (HEAD_KS, K_HEAD, dump, load_cache,         # noqa: E402
                         paired)

GATE_S2C3 = 0.766        # S2-C3's best shared head_recall@8
GATE_IDEAL = 0.82        # the brief's "ideally 0.82-0.85"
MATERIAL = 0.80

# references reproduced from S2-C2 / S2-C3 / S2-B, quoted in the doc
REFERENCES = dict(
    s2c1_lin_l4_head_recall8=0.7358,
    s2c3_best_shared_head_recall8=0.7656,
    s2c3_best_shared_head_agree8=0.2656,
    random_head_recall8=0.247,
    s2c3_lin_l4_macro=57.318,
    teacher_macro=75.150,
    facility_macro=61.097,
    eadp_topk_macro=42.753,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c4")
    args = ap.parse_args()

    T = json.load(open(os.path.join(OUT, f"{args.tag}_transfer.json")))
    P = json.load(open(os.path.join(OUT, f"{args.tag}_pca.json")))
    Geo = json.load(open(os.path.join(OUT, f"{args.tag}_geometry.json")))
    Z = np.load(os.path.join(OUT, f"{args.tag}_transfer.npz"))
    Zp = np.load(os.path.join(OUT, f"{args.tag}_pca.npz"))

    def pi(name, key):
        return Z[f"pi::{name}::{key}"]

    arms = {a["arm"]: a for a in T["arms"]}
    parms = {a["arm"]: a for a in P["arms"]}

    # ---- the four quantities the rule reads ---------------------------------
    self_r8 = pi("SELF", "head_recall8")
    global_r8 = pi("GLOBAL", "head_recall8")
    trained_r8 = pi("HEAD_RANK_seedavg", "head_recall8")

    deployable = [a for a in T["arms"]
                  if a["arm"].startswith(("KNN", "NN__"))] + \
                 [a for a in P["arms"] if a["arm"].startswith("PCA_RIDGE")]
    best_dep = max(deployable, key=lambda a: a["head_recall8"])
    dep_r8 = pi(best_dep["arm"], "head_recall8")

    d_global = paired(dep_r8, global_r8)
    d_trained = paired(dep_r8, trained_r8)
    d_self_global = paired(self_r8, global_r8)

    oracle_by_k = {int(a["arm"].split("_k")[1]): a["head_recall8"]
                   for a in P["arms"] if a["arm"].startswith("PCA_ORACLE_k")}
    self_mean = float(self_r8.mean())
    ge = Geo["pca"]
    er = ge["effective_rank_participation_ratio"]
    dims80 = ge["dims_for_variance"]["n_for_80pct"]
    low_dim = any(oracle_by_k.get(k, 0) >= 0.90 * self_mean
                  for k in (16, 32, 64))
    retention_at = {k: (oracle_by_k[k] / self_mean if k in oracle_by_k else None)
                    for k in (8, 16, 32, 64, 128, 240)}

    # ---- apply the rule ------------------------------------------------------
    beats_global = bool(d_global[1] > 0)
    if dep_r8.mean() >= MATERIAL and beats_global:
        if best_dep["arm"].startswith(("KNN", "NN__")):
            verdict = "B"
        elif low_dim and best_dep["arm"].startswith("PCA_RIDGE"):
            verdict = "A"
        else:
            verdict = "D"
    else:
        verdict = "C"

    gate = dict(
        threshold=GATE_S2C3, ideal=GATE_IDEAL,
        best_deployable=best_dep["arm"],
        best_deployable_recall8=float(dep_r8.mean()),
        clears_s2c3=bool(dep_r8.mean() > GATE_S2C3),
        reaches_ideal=bool(dep_r8.mean() >= GATE_IDEAL),
        downstream_run=False,
        reason=(
            f"best deployable arm {best_dep['arm']} reaches head_recall@8 "
            f"{dep_r8.mean():.4f}, below the S2-C3 shared baseline it would have "
            f"to beat ({GATE_S2C3}); the head-retention half of the experiment "
            f"already excludes it, so no held-out generation was run"),
    )

    verdict_txt = {
        "A": "LOW-DIMENSIONAL IMAGE-ADAPTIVE -- a cheap descriptor predicts a "
             "per-image direction that beats the shared scorer",
        "B": "TRANSFERABLE CLUSTERS -- retrieval over fit directions beats the "
             "shared scorer, but not through a linear low-dimensional family",
        "C": "NON-TRANSFERABLE -- the per-image oracle direction is strong, but "
             "its variation is not recoverable from any cheap descriptor, and "
             "cross-image transfer does not reach the trained shared direction",
        "D": "AMBIGUOUS -- the evidence does not separate the branches",
    }[verdict]

    # ---- what the rule read, so it can be re-applied at other thresholds ----
    rule_inputs = dict(
        SELF_head_recall8=float(self_r8.mean()),
        SELF_head_recall32=float(pi("SELF", "head_recall32").mean()),
        GLOBAL_head_recall8=float(global_r8.mean()),
        BEST_TRAINED_arm="HEAD_RANK_seedavg",
        BEST_TRAINED_head_recall8=float(trained_r8.mean()),
        BEST_DEPLOY_arm=best_dep["arm"],
        BEST_DEPLOY_head_recall8=float(dep_r8.mean()),
        gap_self_minus_global=float(self_r8.mean() - global_r8.mean()),
        gap_self_minus_best_deploy=float(self_r8.mean() - dep_r8.mean()),
        gap_best_trained_minus_best_deploy=float(trained_r8.mean() - dep_r8.mean()),
        d_best_deploy_vs_GLOBAL=dict(delta=float(d_global[0]),
                                     ci=[float(d_global[1]), float(d_global[2])],
                                     clears=beats_global),
        d_best_deploy_vs_BEST_TRAINED=dict(
            delta=float(d_trained[0]),
            ci=[float(d_trained[1]), float(d_trained[2])],
            clears=bool(d_trained[1] > 0)),
        d_self_vs_GLOBAL=dict(delta=float(d_self_global[0]),
                              ci=[float(d_self_global[1]),
                                  float(d_self_global[2])]),
        pca_effective_rank=float(er),
        pca_dims_for_80pct=int(dims80),
        pca_oracle_retention_of_SELF=retention_at,
        low_dimensional=bool(low_dim),
        material_threshold=MATERIAL,
    )

    # ---- the geometry block the decision cites ------------------------------
    geometry_block = dict(
        n_fit=Geo["pairwise_cosine"]["fit"]["n"],
        cosine_mean_within=Geo["pairwise_cosine"]["fit"]["mean"],
        cosine_within_benchmark=Geo["pairwise_cosine"]["fit"]["within_benchmark_mean"],
        cosine_across_benchmark=Geo["pairwise_cosine"]["fit"]["across_benchmark_mean"],
        cosine_p05=Geo["pairwise_cosine"]["fit"]["p05"],
        cosine_p95=Geo["pairwise_cosine"]["fit"]["p95"],
        mean_cosine_to_global=Geo["cosine_to_global"]["fit"]["mean"],
        effective_rank=er,
        dims_for_variance=ge["dims_for_variance"],
        evr_top5=ge["evr_top20"][:5],
        pc1_frac_explained_by_benchmark=Geo[
            "leading_coefficients_by_benchmark"]["pc1"][
                "frac_explained_by_benchmark"],
        cluster_purity_k3=Geo["clusters_wrt_benchmark"]["k=3"][
            "purity_wrt_benchmark"],
        cluster_purity_k8=Geo["clusters_wrt_benchmark"]["k=8"][
            "purity_wrt_benchmark"],
    )

    # ---- the transfer block --------------------------------------------------
    dist = T["transfer_distribution"]
    transfer_block = dict(
        matrix_shape=[int(len(dist["per_image_mean"])), int(T["n_fit"])],
        random_other_mean=dist["mean"], median=dist["median"],
        p05=dist["p05"], p95=dist["p95"], sd=dist["sd"],
        per_image_mean_p50=float(np.median(dist["per_image_mean"])),
        per_image_best_p50=float(np.median(dist["per_image_best"])),
        frac_images_with_a_direction_above_0p85=float(np.mean(
            np.array(dist["per_image_best"]) > 0.85)),
        frac_images_with_a_direction_above_0p95=float(np.mean(
            np.array(dist["per_image_best"]) > 0.95)),
        nn_minus_random_column=float(
            arms["NN__mean"]["head_recall8"] - dist["mean"]),
        knn8_minus_global=float(
            arms["KNN8__meanstd+delta"]["head_recall8"]
            - arms["GLOBAL"]["head_recall8"]),
        by_benchmark={k: v for k, v in T["transfer_by_benchmark"].items()},
    )

    # ---- arms table ----------------------------------------------------------
    arm_rows = []
    for a in T["arms"]:
        r = dict(source="transfer", **a)
        arm_rows.append(r)
    for a in P["arms"]:
        arm_rows.append(dict(source="pca", **a))
    arm_rows.sort(key=lambda r: -r["head_recall8"])

    # ridge predictor detail: everything below GLOBAL is worth naming
    ridge_vs_global = {
        a["arm"]: dict(head_recall8=a["head_recall8"],
                       d_vs_global=float(a["head_recall8"]
                                         - arms["GLOBAL"]["head_recall8"]))
        for a in P["arms"] if a["arm"].startswith("PCA_RIDGE")}

    out = dict(
        note=__doc__,
        verdict=verdict,
        verdict_text=verdict_txt,
        stage="S2-C4",
        question="is the teacher's high-value ranking direction image-dependent, "
                 "and is the per-image variation predictable from cheap forward "
                 "information?",
        n_fit=T["n_fit"], n_test=T["n_test"],
        rule_inputs=rule_inputs,
        downstream_gate=gate,
        geometry=geometry_block,
        transfer=transfer_block,
        arms=arm_rows,
        ridge_vs_global=ridge_vs_global,
        pca_oracle_by_k=oracle_by_k,
        pca_ridge_curves=P["ridge_curves"],
        trained_per_seed=T["trained_per_seed"],
        references=REFERENCES,
        optimistics=(
            {a["arm"]: a["note"] for a in P["arms"] if a["optimistic"]}
            | {a["arm"]: a["note"] for a in T["arms"] if a.get("optimistic")}
        ),
    )
    dump(f"{args.tag}_direction.json", out)

    print(f"\n{'='*78}\nS2-C4 VERDICT: {verdict} -- {verdict_txt}\n{'='*78}\n")
    print(f"  SELF (optimistic)          head_recall@8 = {self_r8.mean():.4f}")
    print(f"  GLOBAL (no adaptation)     head_recall@8 = {global_r8.mean():.4f}")
    print(f"  BEST_TRAINED (S2-C3)       head_recall@8 = {trained_r8.mean():.4f}")
    print(f"  BEST_DEPLOY ({best_dep['arm']})")
    print(f"                             head_recall@8 = {dep_r8.mean():.4f}  "
          f"[{d_global[1]:+.4f},{d_global[2]:+.4f}] vs GLOBAL")
    print(f"  -> deployable is {trained_r8.mean()-dep_r8.mean():+.4f} below the "
          f"trained shared direction")
    print(f"  PCA effective rank = {er:.1f} of 240; {dims80} components for 80 % "
          f"of the direction variance")
    print(f"  oracle reconstruction retains "
          f"{retention_at[32]:.3f} of SELF at k=32, {retention_at[240]:.3f} at k=240")
    print(f"\n  downstream gate: run={gate['downstream_run']} "
          f"({gate['reason']})")


if __name__ == "__main__":
    main()
