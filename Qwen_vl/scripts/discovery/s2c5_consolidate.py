"""
S2-C5 step 3: apply the pre-registered decision rule and write the main artifact.

The rule below was fixed before the trained ladder was read: at the time it was
written only C5-0 and a 2-epoch plumbing smoke test had been seen. It is stated
in absolute points of held-out teacher Top-8 recall@256 and it uses per-image
arrays throughout, so every comparison is paired over the same 150 held-out
images.

  reference      HEAD_RANK, the S2-C3 arm, three seeds, per-image metrics read
                 from s2c3_headmetrics.npz (aligned to this stage's key order)
  MARGIN         0.01 of head_recall@8. One point. S2-C3's whole target move was
                 0.021, so a smaller difference is not an architecture result.
  SEEDS          a comparison must also be reproducible: all three seed-matched
                 deltas (arm_s - reference_s) must be positive. A mean delta that
                 rides on one lucky seed is not a mechanism.

  CLEARS(arm)    seed-averaged delta >= MARGIN, paired-bootstrap 95% CI lower
                 bound > 0, and all three seed-matched deltas > 0
  USES_CTX(arm)  contextual arms only: swapping this image's context for another
                 image's must cost at least MARGIN, with the paired CI of
                 (honest - wrong) excluding zero. A gain that survives its own
                 context being replaced was never reading sample-specific
                 configuration.

Cascade (first match wins):

  UNRESOLVED        neither the matched arms nor the wide local ceiling CLEAR
                    the reference. Nothing in this ladder moved the number, so
                    no mechanism is supported.
  NONLINEAR-LOCAL   LOCAL-MLP CLEARS, and no contextual arm beats it by MARGIN
                    reproducibly. A shared nonlinear token-local function is
                    sufficient; context is not the missing factor.
  SET-DEPENDENT     SET-CTX beats max(LOCAL-MLP, GLOBAL-CTX) by MARGIN
                    reproducibly, AND USES_CTX(SET-CTX).
  IMAGE-GLOBAL      GLOBAL-CTX beats LOCAL-MLP by MARGIN reproducibly, and
                    SET-CTX does not beat GLOBAL-CTX by MARGIN.
  UNRESOLVED        anything else: a contextual arm gains but the gain does not
                    survive the wrong-image substitution, or the arms disagree
                    across seeds, or the two contextual arms do not separate.
                    In all of those the honest report is that this stage has not
                    identified the factor.

The downstream gate is evaluated but not acted on: it requires a *deployable*
(capacity-matched) arm to reach 0.82 head_recall@8 with a positive paired CI. No
generation is run from this stage.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                       # noqa: E402
from s2c3_common import load_plan                              # noqa: E402
from s2c5_common import dump                                   # noqa: E402

MARGIN = 0.01
DEPLOY_TARGET = 0.82
DEPLOY_ASPIRATIONAL = 0.85
MATCHED = ("LOCAL-MLP", "GLOBAL-CTX", "SET-CTX")
WIDE = "LOCAL-MLP-WIDE"
CONTEXTUAL = ("GLOBAL-CTX", "SET-CTX")
SEEDS = (0, 1, 2)
COL = "head_recall8"


def boot(d: np.ndarray, n_boot=10000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(d))}


def compare(A: np.ndarray, B: np.ndarray) -> dict:
    """A - B, with a paired CI over images and a per-seed reproducibility check.

    ``A`` and ``B`` are (n_seeds, n_images). The CI is taken on the seed-averaged
    difference (the estimator actually reported); the per-seed row means are
    recorded so a gain carried by one seed is visible rather than averaged away.
    """
    D = A - B
    per_seed = [float(D[i].mean()) for i in range(D.shape[0])]
    out = boot(D.mean(0))
    out["per_seed"] = per_seed
    out["all_seeds_positive"] = bool(all(x > 0 for x in per_seed))
    out["seed_min"], out["seed_max"] = float(min(per_seed)), float(max(per_seed))
    out["passes"] = bool(out["mean"] >= MARGIN and out["lo"] > 0
                         and out["all_seeds_positive"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c5")
    args = ap.parse_args()

    T = json.load(open(os.path.join(OUT, f"{args.tag}_train.json")))
    C = json.load(open(os.path.join(OUT, f"{args.tag}_controls.json")))
    Z = np.load(os.path.join(OUT, f"{args.tag}_perimage.npz"))
    H = np.load(os.path.join(OUT, "s2c3_headmetrics.npz"))
    D0 = json.load(open(os.path.join(OUT, f"{args.tag}_c0_diagnosis.json")))

    cols = T["metric_columns"]
    ci = cols.index(COL)
    test_keys = T["test_keys"]
    ds_test = C["ds_test"]

    # s2c3_headmetrics stores a (tag, image) matrix with separate tag/key axes,
    # and its key order is not this stage's; align on keys, never on position.
    hm_keys = [str(k) for k in H["keys"]]
    hm_tags = [str(t) for t in H["tags"]]
    assert set(hm_keys) == set(test_keys), "held-out key sets differ"
    hm_order = [hm_keys.index(k) for k in test_keys]

    def hm_row(tag):
        return H[COL][hm_tags.index(tag)][hm_order]

    ref = np.stack([hm_row(f"HEAD_RANK_s{s}") for s in SEEDS])           # (3,150)
    lin = hm_row("S2C1_LIN_L4")
    arm = {a: np.stack([Z[f"{a}_s{s}"][:, ci] for s in SEEDS])            # (3,150)
           for a in list(MATCHED) + [WIDE]}

    res = {}
    for a in list(MATCHED) + [WIDE]:
        e = C["arms"][a]
        r = {"seed_mean_head_recall8": float(arm[a].mean()),
             "seed_sd": float(arm[a].mean(1).std()),
             "per_seed_head_recall8": [float(x) for x in arm[a].mean(1)],
             "n_params": T["results"][f"{a}_s0"]["n_params"],
             "access": T["results"][f"{a}_s0"]["access"],
             "vs_reference": compare(arm[a], ref),
             "by_benchmark": e["by_benchmark"]}
        r["uses_ctx"] = uses_ctx(e) if a in CONTEXTUAL else {"available": False}
        if a == WIDE:
            r["note"] = ("capacity ceiling for the token-local hypothesis, "
                         "deliberately not capacity-matched (2.36 M)")
        res[a] = r

    pairwise = {
        "GLOBAL-CTX_minus_LOCAL-MLP": compare(arm["GLOBAL-CTX"], arm["LOCAL-MLP"]),
        "SET-CTX_minus_LOCAL-MLP": compare(arm["SET-CTX"], arm["LOCAL-MLP"]),
        "SET-CTX_minus_GLOBAL-CTX": compare(arm["SET-CTX"], arm["GLOBAL-CTX"]),
        "WIDE_minus_LOCAL-MLP": compare(arm[WIDE], arm["LOCAL-MLP"]),
    }

    # ------------------------------------------------------------- cascade --
    ok = {a: res[a]["vs_reference"]["passes"] for a in MATCHED}
    ctx_beats_local = {"GLOBAL-CTX": pairwise["GLOBAL-CTX_minus_LOCAL-MLP"]["passes"],
                       "SET-CTX": pairwise["SET-CTX_minus_LOCAL-MLP"]["passes"]}
    set_over_global = pairwise["SET-CTX_minus_GLOBAL-CTX"]["passes"]

    if not any(ok.values()) and not res[WIDE]["vs_reference"]["passes"]:
        verdict, reason = "UNRESOLVED", (
            "no arm CLEARS the linear reference by >= 1 pt of held-out Top-8 "
            "recall reproducibly, including the 4x-wide token-local ceiling")
    elif ok["LOCAL-MLP"] and not any(ctx_beats_local.values()):
        verdict, reason = "NONLINEAR-LOCAL", (
            f"LOCAL-MLP CLEARS the reference by "
            f"{res['LOCAL-MLP']['vs_reference']['mean']:+.3f} reproducibly, and no "
            "contextual arm beats it by >= 1 pt reproducibly")
    elif (ctx_beats_local["SET-CTX"] and set_over_global
          and res["SET-CTX"]["uses_ctx"].get("passes")):
        verdict, reason = "SET-DEPENDENT", (
            "SET-CTX beats both parameter-matched arms reproducibly and swapping "
            "its context for another image's costs >= 1 pt")
    elif ctx_beats_local["GLOBAL-CTX"] and not set_over_global:
        verdict, reason = "IMAGE-GLOBAL", (
            "GLOBAL-CTX beats parameter-matched LOCAL-MLP reproducibly while "
            "SET-CTX adds nothing on top of it")
    else:
        verdict, reason = "UNRESOLVED", (
            "the arms do not separate under the pre-registered rule -- either no "
            "matched arm clears the reference reproducibly, or a contextual arm "
            "gains without surviving the wrong-image substitution, or the seed "
            "pattern is not consistent")

    gate_arms = [a for a in MATCHED
                 if res[a]["seed_mean_head_recall8"] >= DEPLOY_TARGET
                 and res[a]["vs_reference"]["lo"] > 0]
    gate = {"rule": f"capacity-matched arm with head_recall@8 >= {DEPLOY_TARGET} "
                    f"and a positive paired CI vs HEAD_RANK "
                    f"({res['LOCAL-MLP']['vs_reference']['mean'] * 0 + ref.mean():.3f}); "
                    f"aspirational {DEPLOY_ASPIRATIONAL}",
            "arms_clearing": gate_arms, "downstream_run": bool(gate_arms),
            "note": "no generation is run in S2-C5"}

    obj = {
        "stage": "S2-C5",
        "question": "is the teacher's high-value ranking-head signal a shared "
                    "nonlinear token-local function, or does it require the "
                    "image's other visual tokens?",
        "verdict": verdict,
        "verdict_reason": reason,
        "rule": {"margin": MARGIN, "primary": "held-out teacher Top-8 recall@256",
                 "reference": "HEAD_RANK (S2-C3), seed-averaged, per-image metrics "
                              "from s2c3_headmetrics.npz aligned by key",
                 "reference_head_recall8": float(ref.mean()),
                 "reference_per_seed": [float(x) for x in ref.mean(1)],
                 "reference_seed_sd": float(ref.mean(1).std()),
                 "requires_all_seed_matched_deltas_positive": True,
                 "paired_bootstrap_draws": 10000,
                 "frozen_before_reading_ladder": True},
        "references": {"HEAD_RANK": float(ref.mean()),
                       "LIN_L4": float(lin.mean()),
                       "HEAD_RANK_per_seed": [float(x) for x in ref.mean(1)],
                       "oracle_SELF_s2c4": 0.95,
                       "oracle_SELF_TOP8DIR_s2c4": 0.9817},
        "arms": res, "pairwise": pairwise, "deployable_gate": gate,
        "c5_0_diagnosis_summary": summarise_c0(D0),
        "sources": {k: f"{args.tag}_{k}.json" for k in
                    ("train", "controls", "c0_diagnosis")},
    }
    dump(f"{args.tag}_factorization.json", obj)

    # ----------------------------------------------------------------- print -
    print(f"\nreference HEAD_RANK head_recall@8 = {ref.mean():.4f} "
          f"(per-seed {[round(float(x), 4) for x in ref.mean(1)]}, "
          f"sd {ref.mean(1).std():.4f})")
    print(f"LIN_L4 (S2-C1)                    = {lin.mean():.4f}")
    print(f"\n{'arm':<18} {'h8':>7} {'sd':>7} {'delta':>8} {'CI_lo':>8} "
          f"{'seedΔ':>22} {'params':>10} clears")
    for a in list(MATCHED) + [WIDE]:
        r, b = res[a], res[a]["vs_reference"]
        print(f"{a:<18} {r['seed_mean_head_recall8']:7.4f} {r['seed_sd']:7.4f} "
              f"{b['mean']:+8.4f} {b['lo']:+8.4f} "
              f"{str([round(x, 3) for x in b['per_seed']]):>22} "
              f"{r['n_params']:10,} {b['passes']}")
    print("\npairwise (paired over 150 held-out images, seed-averaged):")
    for k, v in pairwise.items():
        print(f"  {k:<34} delta={v['mean']:+.4f} "
              f"CI=[{v['lo']:+.4f},{v['hi']:+.4f}] "
              f"per-seed={[round(x, 3) for x in v['per_seed']]} passes={v['passes']}")
    print("\nwrong-image context control (positive = context carries "
          "sample-specific information):")
    for a in CONTEXTUAL:
        u = res[a]["uses_ctx"]
        if u.get("available"):
            print(f"  {a:<12} honest-wrong={u['delta']:+.4f} passes={u['passes']} "
                  f"per-seed={[round(p['honest_minus_wrong'], 4) for p in u['per_seed']]}")
            for p in u["per_seed"]:
                print(f"      seed {p['seed']}: honest-wrong={p['honest_minus_wrong']:+.4f} "
                      f"max|Δscore|={p['score_max_abs_diff']:.2e} "
                      f"top256 Jaccard={p['top256_jaccard']:.4f} "
                      f"wrong-fit h8={p['wrong_fit_head_recall8']:.4f}")
    print(f"\nVERDICT: {verdict}\n  {reason}")
    print(f"\ndeployable gate ({DEPLOY_TARGET}): {gate['arms_clearing'] or 'none'}")


def uses_ctx(entry: dict) -> dict:
    """honest - wrong-image, per seed, averaged over the derangement draws."""
    per_seed, all_lo, all_hi = [], [], []
    for c in entry.get("controls", []):
        ds = [d for d in c["draws"] if "paired_vs_honest" in d]
        if not ds:
            continue
        # honest - wrong, so positive means "the context was read"
        m = float(np.mean([-d["paired_vs_honest"]["mean"] for d in ds]))
        per_seed.append({
            "seed": c["seed"], "honest_minus_wrong": m,
            "paired_lo": float(np.mean([-d["paired_vs_honest"]["hi"] for d in ds])),
            "paired_hi": float(np.mean([-d["paired_vs_honest"]["lo"] for d in ds])),
            "score_max_abs_diff": float(np.mean([d["score_max_abs_diff"] for d in ds])),
            "top256_jaccard": float(np.mean([d["top256_jaccard"] for d in c["draws"]
                                             if "top256_jaccard" in d])),
            "wrong_fit_head_recall8": c["wrong_fit"]["head_recall8"]})
        all_lo.append(per_seed[-1]["paired_lo"])
        all_hi.append(per_seed[-1]["paired_hi"])
    if not per_seed:
        return {"available": False}
    delta = float(np.mean([p["honest_minus_wrong"] for p in per_seed]))
    return {"available": True, "delta": delta, "per_seed": per_seed,
            "ci": [float(np.mean(all_lo)), float(np.mean(all_hi))],
            "passes": bool(delta >= MARGIN and np.mean(all_lo) > 0),
            "note": "delta = honest - wrong-image head_recall@8; positive means "
                    "the context carries sample-specific information"}


def summarise_c0(D0: dict) -> dict:
    out = {}
    for key in D0["arms"]:
        a = D0["arms"][key]
        k8 = a["by_k"]["8"]
        out[key] = {"head_recall8": a["aggregate"]["head_recall8"],
                    "missed_token_rank": k8["missed_token_rank"],
                    "missed_rank_buckets": k8["missed_rank_buckets"],
                    "per_image_worst_missed_rank": k8["per_image_worst_missed_rank"],
                    "candidate_oracle_coverage": k8["candidate_oracle_coverage"],
                    "by_benchmark_k8": {d: {"head_recall": a["by_benchmark"][d]["8"]["head_recall"],
                                            "missed_token_rank": a["by_benchmark"][d]["8"]["missed_token_rank"]}
                                        for d in a["by_benchmark"]}}
    return out


if __name__ == "__main__":
    main()
