"""
S2-C5 step 0: Candidate Accessibility Diagnosis. No model is trained here.

The question this file answers is a *geometry* question about the frozen
baselines, and it decides the architecture of the C5 ladder:

  When HEAD_RANK (or LIN_L4) keeps a teacher-head token out of the selected 256,
  where does that token actually sit in the student's ranking?

  (a) If the misses pile up just past the frontier, then a contextual scorer only
      ever has to look at a small candidate pool (top-384/512) -- the expensive
      token-token interaction can be confined to a frontier set, and the
      candidate oracle coverage (recall@C) is the ceiling to report.
  (b) If the misses are spread deep into the ranking, no frontier-only
      architecture can recover them and C5 must say so rather than build one.

Reported per arm and per teacher head size k in {8, 16, 32}:

  * pooled rank distribution of missed head tokens: median / mean / p75 / p90 / p95
  * fraction with student rank <= 256 / 320 / 384 / 512 / 768 / 1024
  * excess-rank buckets (how many places past the 256 frontier)
  * per-image worst missed-head rank distribution, and per-image mean missed rank
  * miss rate broken down by the token's *teacher* rank position (0..k-1) --
    whether the student loses the teacher's very best tokens or only its marginal
    head members
  * normalized score margin to the selection threshold (score of the missed token
    minus the score of the 256th-ranked token, divided by the within-image score
    standard deviation) -- a scale-free "how close was it" measure
  * the recall@C candidate-oracle curve for C in 256..1024 -- the ceiling of any
    scorer restricted to a top-C pool

All numbers come from the cached score banks (s2c1_scores.npz, s2c3_scores.npz)
and the cached P1-G2 teacher maps; nothing is recomputed and the GPU is untouched.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2c5_common import (BUDGET, DS_ALL, FROZEN_REF, LOCAL_LIN,  # noqa: E402
                         bank_tag, dump, excess_buckets, head_metrics,
                         image_ranks, load_bank, missed_rank_stats,
                         rank_buckets, recall_curve, seed_mean_scores,
                         student_ranks, test_rows)

KS = (8, 16, 32)
CS = (256, 288, 320, 384, 448, 512, 640, 768, 896, 1024)
PCTS = (50, 75, 90, 95, 100)


def dist(v: np.ndarray) -> dict:
    """Summary of a per-image distribution."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return {}
    q = np.percentile(v, PCTS)
    return {"mean": float(v.mean()), "median": float(q[0]), "p75": float(q[1]),
            "p90": float(q[2]), "p95": float(q[3]), "max": float(q[4])}


def analyse_arm(name: str, scores: dict, rows, keys, ds, orders) -> dict:
    """Everything the brief asks for, for one frozen arm over the held-out 150."""
    out = {"arm": name, "note": ""}

    # aggregate metrics, to tie these rows back to the S2-C3 table
    per_img = [head_metrics(scores[k], orders[k]) for k in keys]
    out["aggregate"] = {m: float(np.mean([r[m] for r in per_img]))
                        for m in per_img[0] if m in
                        ("overlap256", "head_recall8", "head_recall16",
                         "head_recall32", "head_agree8")}

    ranks_all = {k: student_ranks(scores[kk]) for k, kk in zip(keys, keys)}
    by_k = {}
    for k in KS:
        # ---- pooled over (image, missed token) pairs ------------------------
        miss_ranks, thr_margin, img_nmiss, img_mean, img_worst, img_hr = (
            [], [], [], [], [], [])
        pos_miss = np.zeros(k, dtype=np.int64)          # miss count per teacher rank
        pos_tot = np.zeros(k, dtype=np.int64)
        for key in keys:
            torder = orders[key]
            head, r_all, hr, missed = image_ranks(scores[key], torder, k)
            pos_tot += 1
            pos_miss += missed.astype(np.int64)
            img_nmiss.append(float(missed.sum()))
            img_hr.append(hr.astype(np.float64))
            if missed.any():
                mr = hr[missed].astype(np.float64)
                miss_ranks.append(mr)
                img_mean.append(float(mr.mean()))
                img_worst.append(float(mr.max()))
                # margin to the selection threshold, in within-image score sd
                s = np.asarray(scores[key], dtype=np.float64)
                thr = np.sort(s)[::-1][BUDGET - 1]
                sd = float(s.std()) or 1.0
                thr_margin.append((s[head[missed]] - thr) / sd)
            else:
                img_mean.append(0.0)
                img_worst.append(float(BUDGET))         # no miss -> at the frontier

        mr = np.concatenate(miss_ranks) if miss_ranks else np.array([])
        tm = np.concatenate(thr_margin) if thr_margin else np.array([])
        n_missing_images = int(np.sum(np.asarray(img_nmiss) > 0))

        # ---- candidate-oracle curve ----------------------------------------
        curve = [recall_curve(scores[key], orders[key], ks=(k,), cs=CS)
                 for key in keys]
        cov = {f"recall{k}@{c}": float(np.mean([r[f"recall{k}@{c}"] for r in curve]))
               for c in CS}

        by_k[k] = {
            "missed_token_rank": missed_rank_stats(mr),
            "missed_rank_buckets": rank_buckets(mr),
            "missed_excess_buckets": excess_buckets(mr),
            "threshold_margin_sd": (missed_rank_stats(tm) if len(tm) else {"n": 0}),
            # sanity check, not evidence: a "missed" token is by definition one
            # whose score is at or below the 256th-ranked score, so this is 1.0
            # by construction. It is recorded only to prove the miss definition
            # is the selection definition; the informative frontier-distance
            # measures are ``missed_rank_buckets`` and ``missed_excess_buckets``.
            "frac_missed_at_or_below_threshold_score": (
                float((tm <= 0).mean()) if len(tm) else float("nan")),
            "frac_missed_at_or_below_threshold_score_note":
                "1.0 by construction (definitional, not a finding)",
            "per_image_worst_missed_rank": dist(img_worst),
            "per_image_worst_missed_rank_buckets": rank_buckets(
                np.asarray(img_worst)),
            "per_image_mean_missed_rank": dist([v for v in img_mean if v > 0]),
            "per_image_n_missed": dist(img_nmiss),
            "n_images_with_any_miss": n_missing_images,
            "frac_images_with_any_miss": n_missing_images / len(keys),
            "miss_rate_by_teacher_rank": (pos_miss / pos_tot).tolist(),
            "candidate_oracle_coverage": cov,
            "recall_gain_from_pool": {
                f"pool{c}_minus_256": float(cov[f"recall{k}@{c}"]
                                            - cov[f"recall{k}@256"])
                for c in CS},
        }
    out["by_k"] = by_k

    # ---- per-benchmark decomposition ---------------------------------------
    per_ds = {}
    for d in DS_ALL:
        sel = [i for i, x in enumerate(ds) if x == d]
        dkeys = [keys[i] for i in sel]
        dout = {}
        for k in KS:
            mr, worst, nmiss = [], [], []
            for key in dkeys:
                head, r_all, hr, missed = image_ranks(scores[key], orders[key], k)
                nmiss.append(float(missed.sum()))
                if missed.any():
                    mr.append(hr[missed].astype(np.float64))
                    worst.append(float(hr[missed].max()))
            mrc = np.concatenate(mr) if mr else np.array([])
            dout[k] = {
                "n_images": len(dkeys),
                "head_recall": float(np.mean(
                    [head_metrics(scores[key], orders[key])[f"head_recall{k}"]
                     for key in dkeys])),
                "missed_token_rank": missed_rank_stats(mrc),
                "missed_rank_buckets": rank_buckets(mrc),
                "per_image_worst_missed_rank": dist(worst),
                "per_image_worst_missed_rank_buckets": rank_buckets(
                    np.asarray(worst)),
                "per_image_n_missed": dist(nmiss),
            }
        per_ds[d] = dout
    out["by_benchmark"] = per_ds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=[FROZEN_REF, LOCAL_LIN, "HEAD_BIN", "BASE"])
    ap.add_argument("--tag", default="s2c5_c0")
    args = ap.parse_args()

    rows, keys, ds, orders, meta, plan, all_keys, rows_of = test_rows()
    print(f"[split] held-out n={len(keys)} "
          f"({', '.join(f'{d}:{ds.count(d)}' for d in DS_ALL)})")

    banks = {a: load_bank(a) for a in args.arms}
    results, seed_views = {}, {}
    for arm in args.arms:
        per_seed = banks[arm]
        seeds = list(per_seed)
        # per-seed
        for s in seeds:
            name = bank_tag(arm, s)
            results[name] = analyse_arm(name, per_seed[s], rows, keys, ds, orders)
        # seed-mean: the headline reading for the multi-seed frozen arms
        if len(seeds) > 1:
            sm = seed_mean_scores(per_seed)
            results[f"{arm}_seedavg"] = analyse_arm(f"{arm}_seedavg", sm, rows,
                                                    keys, ds, orders)
        seed_views[arm] = {
            bank_tag(arm, s): {
                "head_recall8": float(np.mean(
                    [head_metrics(per_seed[s][k], orders[k])["head_recall8"]
                     for k in keys]))
            } for s in seeds}

    obj = {
        "note": __doc__,
        "config": vars(args),
        "split": {"fit": len(rows_of["fit"]), "val": len(rows_of["val"]),
                  "test": len(keys), "budget": BUDGET, "layer": 4},
        "seed_head_recall8_spread": seed_views,
        "arms": results,
    }
    dump(f"{args.tag}_diagnosis.json", obj)

    # ------------------------------------------------------------- console --
    for arm in args.arms:
        for tag in ([f"{arm}_seedavg"] if len(banks[arm]) > 1 else [arm]):
            r = results[tag]
            print(f"\n=== {tag}  h8={r['aggregate']['head_recall8']:.3f} "
                  f"h16={r['aggregate']['head_recall16']:.3f} "
                  f"h32={r['aggregate']['head_recall32']:.3f} "
                  f"ov={r['aggregate']['overlap256']:.3f}")
            for k in KS:
                b = r["by_k"][k]
                m = b["missed_token_rank"]
                w = b["per_image_worst_missed_rank"]
                print(f"  k={k:<2} missed={m['n']:<5} "
                      f"rank med={m['median']:.0f} mean={m['mean']:.0f} "
                      f"p75={m['p75']:.0f} p90={m['p90']:.0f} p95={m['p95']:.0f} "
                      f"| <=320 {b['missed_rank_buckets']['frac_le_320']:.2f} "
                      f"<=384 {b['missed_rank_buckets']['frac_le_384']:.2f} "
                      f"<=512 {b['missed_rank_buckets']['frac_le_512']:.2f} "
                      f"<=768 {b['missed_rank_buckets']['frac_le_768']:.2f}")
                print(f"        worst/img med={w['median']:.0f} "
                      f"p90={w['p90']:.0f} p95={w['p95']:.0f} max={w['max']:.0f} "
                      f"| imgs w/ miss {b['frac_images_with_any_miss']:.2f} "
                      f"| mask{margin_str(b)} "
                      f"| cov@256={b['candidate_oracle_coverage'][f'recall{k}@256']:.3f} "
                      f"@384={b['candidate_oracle_coverage'][f'recall{k}@384']:.3f} "
                      f"@512={b['candidate_oracle_coverage'][f'recall{k}@512']:.3f} "
                      f"@768={b['candidate_oracle_coverage'][f'recall{k}@768']:.3f}")
                print(f"        miss rate by teacher rank {[round(x,2) for x in b['miss_rate_by_teacher_rank']]}")

    print("\n=== per-benchmark, HEAD_RANK_seedavg, k=8 ===")
    r = results.get(f"{FROZEN_REF}_seedavg") or results[FROZEN_REF]
    for d in DS_ALL:
        b = r["by_benchmark"][d][8]
        m = b["missed_token_rank"]
        print(f"  {d:<12} h8={b['head_recall']:.3f} missed={m['n']:<4} "
              f"median={m['median']:.0f} p90={m['p90']:.0f} "
              f"<=384 {b['missed_rank_buckets']['frac_le_384']:.2f} "
              f"<=512 {b['missed_rank_buckets']['frac_le_512']:.2f} "
              f"| worst med={b['per_image_worst_missed_rank']['median']:.0f}")


def margin_str(b: dict) -> str:
    t = b["threshold_margin_sd"]
    if not t.get("n"):
        return "n/a"
    return f"med={t['median']:.2f}sd <=0:{b['frac_missed_at_or_below_threshold_score']:.2f}"


if __name__ == "__main__":
    main()
