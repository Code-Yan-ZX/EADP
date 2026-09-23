"""
M2 — consolidation: apply the pre-registered criteria and pick one conclusion.

Reads m2_accuracy.json, m2_perf.json, m2_correctness.json and writes
m2_system_audit.json plus m2_tables.md. Nothing is measured here; every number
comes from the three runners, and every threshold comes from the pre-registration
(§5), frozen before the first measurement.

Usage
    python scripts/discovery/m2_consolidate.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR                                        # noqa: E402
from m2_gdep import dump_json                                        # noqa: E402

DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
SEEDS = (0, 1, 2)
NBOOT = 10000

# ---- pre-registered thresholds (prereg §5). Frozen before measurement. -------
CRIT = dict(
    primary=dict(ci_lower_pts=-1.0, ttft_reduction=0.15),
    compression=dict(acc_drop_pts=2.0, ttft_reduction=0.25, mem_reduction=0.20),
    accuracy_led=dict(ci_lower_pts=0.0, ttft_ratio=1.10),
)

CANDIDATES = ("C0", "C1", "C2", "C3")
CONTROLS = ("C1-R", "C1-SHUF")

# ---- offline teacher-label cost (M1 §2.4 / S2-B §5). NEVER a deployment cost. -
TEACHER = dict(
    per_image_ms_cuda=512.3,          # 216.1 fwd + 296.2 bwd, measured
    per_image_ms_wall=912.0,          # 657 s / 720 maps, measured wall clock
    n_maps_n960=1170,                 # 450 frozen + 720 extension
    n_fit_n960=960,
    note=("offline gradient-teacher labels; amortised training cost, excluded "
          "from every deployment latency figure (prereg §6.3)"),
)


def load(name):
    p = os.path.join(OUTPUT_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def boot_macro(hits_a, hits_b, nb=NBOOT, seed=0):
    """Paired stratified bootstrap of the macro difference, over 3 benchmarks."""
    rng = np.random.default_rng(seed)
    draws = np.zeros(nb)
    for ds in DS_ORDER:
        a, b = np.asarray(hits_a[ds], float), np.asarray(hits_b[ds], float)
        n = len(a)
        idx = rng.integers(0, n, size=(nb, n))
        draws += (a[idx].mean(1) - b[idx].mean(1)) / len(DS_ORDER)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(mean=float(draws.mean()), lo=float(lo), hi=float(hi),
                ci_excludes_zero=bool(lo > 0 or hi < 0), n_boot=nb)


def hits_of(rec):
    return {ds: rec["per_benchmark"][ds]["hits"] for ds in DS_ORDER}


def macro_of(rec):
    return float(np.mean([np.mean(rec["per_benchmark"][ds]["hits"])
                          for ds in DS_ORDER]))


def transitions(a_hits, b_hits, cut=0.5):
    """a vs b on the same instances: rescued / still-wrong / newly-broken."""
    A = np.concatenate([np.asarray(a_hits[ds], float) for ds in DS_ORDER])
    B = np.concatenate([np.asarray(b_hits[ds], float) for ds in DS_ORDER])
    ac, bc = A >= cut, B >= cut
    return dict(rescued=int((~bc & ac).sum()), still_wrong=int((~bc & ~ac).sum()),
                newly_broken=int((bc & ~ac).sum()), both_correct=int((bc & ac).sum()),
                n=int(len(A)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="m2_accuracy.json")
    ap.add_argument("--perf", default="m2_perf.json")
    ap.add_argument("--corr", default="m2_correctness.json")
    ap.add_argument("--out", default="m2_system_audit.json")
    args = ap.parse_args()

    acc, perf, corr = load(args.acc), load(args.perf), load(args.corr)
    if acc is None or perf is None:
        raise SystemExit("accuracy and perf records are both required")
    arms = acc["arms"]

    audit = dict(stage="M2", method="GDEP", prereg="docs/scoring_search_m2_system_prereg.md",
                 thresholds=CRIT, teacher_offline_cost=TEACHER,
                 correctness=(corr or {}).get("all_passed"),
                 gate_F=acc.get("gate_F_harness_identity"),
                 accuracy_grid=dict(arms=acc["config"]["arms"],
                                    seeds=acc["config"]["seeds"],
                                    n_instances=acc["n_instances"]))

    # ---------------- seed means -------------------------------------------
    def seed_arm(arm):
        rs = [arms.get(f"{arm}|s{s}") for s in SEEDS]
        if any(r is None or "error" in r for r in rs):
            return None
        return rs

    table = {}
    for key, rec in arms.items():
        if "error" in rec:
            continue
        table[key] = dict(arm=rec["arm"], seed=rec["seed"],
                          macro_pct=rec["macro_pct"],
                          per_benchmark={ds: rec["per_benchmark"][ds]["acc_pct"]
                                         for ds in DS_ORDER},
                          n_kept_mean=rec.get("n_kept_mean"))
    audit["accuracy_table"] = table
    audit["seed_means"] = acc.get("seed_means")
    audit["summary_vs_B1"] = acc.get("summary_vs_B1")

    if "B1" not in arms or "error" in arms["B1"]:
        dump_json(args.out, audit)
        raise SystemExit("B1 did not run; nothing can be evaluated")

    b1, b0 = arms["B1"], arms.get("B0")
    b1_hits = hits_of(b1)
    b0_hits = hits_of(b0) if b0 and "error" not in b0 else None
    b1_macro = macro_of(b1)

    # ---------------- error transitions ------------------------------------
    trans = {}
    for key, rec in arms.items():
        if "error" in rec or rec["arm"] in ("B1",):
            continue
        t = dict(vs_B1=transitions(hits_of(rec), b1_hits))
        if b0_hits is not None:
            t["vs_B0"] = transitions(hits_of(rec), b0_hits)
        trans[key] = t
    audit["error_transitions"] = trans

    # ---------------- efficiency table -------------------------------------
    eff = {}
    for tag, r in perf["arms"].items():
        if "prefill" not in r:
            continue
        eff[tag] = dict(
            label=r.get("label"),
            ttft_wall_median_ms=r["prefill"]["wall"]["median"],
            ttft_wall_mean_ms=r["prefill"]["wall"]["mean"],
            ttft_wall_p90_ms=r["prefill"]["wall"]["p90"],
            ttft_wall_std_ms=r["prefill"]["wall"]["std"],
            prefill_event_median_ms=r["prefill"]["event"]["median"],
            prefill_event_mean_ms=r["prefill"]["event"]["mean"],
            prefill_event_p90_ms=r["prefill"]["event"]["p90"],
            prefill_event_std_ms=r["prefill"]["event"]["std"],
            stages={k: v["mean"] for k, v in r["prefill"]["stages"].items()},
            n_kept=r["prefill"]["n_kept"], context_len=r["prefill"]["context_len"],
            seq_full=r["prefill"]["seq_full"],
            kv_bytes_early=r["prefill"]["kv_bytes_early"],
            kv_bytes_late=r["prefill"]["kv_bytes_late"],
            kv_bytes_total=r["prefill"]["kv_bytes_total"],
            kv_total_if_no_prune=r["prefill"]["kv_total_if_no_prune"],
            peak_allocated_mb=r["peak_allocated_mb"],
            peak_delta_over_load_mb=r["peak_delta_over_load_mb"],
            decode32_median_ms=r["decode"]["total_cuda"]["median"],
            decode32_mean_ms=r["decode"]["total_cuda"]["mean"],
            tpot_ms=r["decode"]["tpot_ms"],
            throughput_rps=r["throughput_rps"],
            cache_policy=r["prefill"]["cache_policy"],
            pos_policy=r["prefill"]["pos_policy"],
            scorer_params=r["prefill"]["scorer_params"],
            extra_forward_or_recompute=r["extra_forward_or_recompute"],
            cfg_hash=r["prefill"]["cfg_hash"],
            speedup_vs=r.get("speedup_vs"),
        )
    audit["efficiency_table"] = eff
    audit["perf_environment"] = perf.get("config")
    audit["batch4"] = perf.get("batch4")

    # ---------------- pre-registered criteria ------------------------------
    b1_ttft = eff.get("B1", {}).get("ttft_wall_median_ms")
    b1_peak = eff.get("B1", {}).get("peak_allocated_mb")
    b1_peak_delta = eff.get("B1", {}).get("peak_delta_over_load_mb")
    crit = {}
    for cand in CANDIDATES:
        rs = seed_arm(cand)
        if rs is None:
            crit[cand] = dict(evaluated=False, reason="arm incomplete")
            continue
        # seed-averaged per-image hits, paired against B1
        A = {ds: np.mean([np.asarray(r["per_benchmark"][ds]["hits"], float)
                          for r in rs], axis=0) for ds in DS_ORDER}
        ci = boot_macro(A, b1_hits)
        obs = float(np.mean([A[ds].mean() for ds in DS_ORDER])
                    - np.mean([np.asarray(b1_hits[ds], float).mean()
                               for ds in DS_ORDER]))
        d_pts = obs * 100
        lo_pts, hi_pts = ci["lo"] * 100, ci["hi"] * 100
        macro_pts = float(np.mean([A[ds].mean() for ds in DS_ORDER]) * 100)
        e = eff.get(cand, {})
        ttft = e.get("ttft_wall_median_ms")
        ttft_red = (1 - ttft / b1_ttft) if (ttft and b1_ttft) else None
        mem_red = (1 - e.get("peak_allocated_mb", 0) / b1_peak) if b1_peak else None
        mem_red_delta = ((1 - e.get("peak_delta_over_load_mb", 0) / b1_peak_delta)
                         if b1_peak_delta else None)
        c1_ = bool(lo_pts > CRIT["primary"]["ci_lower_pts"]
                   and ttft_red is not None
                   and ttft_red >= CRIT["primary"]["ttft_reduction"])
        c2_ = ((b1_macro * 100 - macro_pts) <= CRIT["compression"]["acc_drop_pts"]
               and ((ttft_red is not None and ttft_red >= CRIT["compression"]["ttft_reduction"])
                    or (mem_red is not None and mem_red >= CRIT["compression"]["mem_reduction"])))
        c3_ = (lo_pts > CRIT["accuracy_led"]["ci_lower_pts"]
               and ttft is not None and b1_ttft is not None
               and ttft <= CRIT["accuracy_led"]["ttft_ratio"] * b1_ttft)
        crit[cand] = dict(
            evaluated=True, macro_pct=macro_pts, macro_seed_range=[
                min(r["macro_pct"] for r in rs), max(r["macro_pct"] for r in rs)],
            vs_B1_macro_pts=d_pts, vs_B1_ci_pts=[lo_pts, hi_pts],
            ttft_median_ms=ttft, ttft_reduction_vs_B1=ttft_red,
            peak_mb=e.get("peak_allocated_mb"), peak_reduction_vs_B1=mem_red,
            peak_delta_mb=e.get("peak_delta_over_load_mb"),
            peak_delta_reduction_vs_B1=mem_red_delta,
            vs_B0_macro_pts=(macro_pts - macro_of(b0) * 100)
            if b0 and "error" not in b0 else None,
            primary_success=bool(c1_), strong_compression_success=bool(c2_),
            accuracy_led_success=bool(c3_))
    audit["criteria"] = crit

    # ---- matched-seed deltas: n=960 vs n=240 (C1 - C0), same seed ---------
    ms = {}
    for a, b, label in (("C1", "C0", "n960_minus_n240"),
                        ("C2", "C1", "block8_minus_topk"),
                        ("C3", "C1", "facility_minus_topk"),
                        ("C1-R", "C1", "renumber_minus_preserve"),
                        ("C1-SHUF", "C1", "shuffled_minus_honest")):
        ra, rb = seed_arm(a), seed_arm(b)
        if ra is None or rb is None:
            continue
        per_seed = []
        for xa, xb in zip(ra, rb):
            per_seed.append(float(np.mean([
                np.mean(xa["per_benchmark"][ds]["hits"])
                - np.mean(xb["per_benchmark"][ds]["hits"]) for ds in DS_ORDER]) * 100))
        A = {ds: np.mean([np.asarray(r["per_benchmark"][ds]["hits"], float)
                          for r in ra], axis=0) for ds in DS_ORDER}
        B = {ds: np.mean([np.asarray(r["per_benchmark"][ds]["hits"], float)
                          for r in rb], axis=0) for ds in DS_ORDER}
        ci = boot_macro(A, B)
        ms[label] = dict(a=a, b=b, per_seed_pts=per_seed,
                         mean_pts=float(np.mean(per_seed)),
                         ci_pts=[ci["lo"] * 100, ci["hi"] * 100],
                         all_seeds_positive=bool(all(x > 0 for x in per_seed)))
    audit["matched_seed_deltas"] = ms

    # ---------------- the conclusion (prereg §8) ---------------------------
    audit["conclusion"] = _conclude(audit, crit, eff, b1_ttft, arms)

    dump_json(args.out, audit)
    _tables(audit)
    print(f"\n[conclusion] {audit['conclusion']['code']}: "
          f"{audit['conclusion']['statement']}")


def _conclude(audit, crit, eff, b1_ttft, arms):
    """P1..P5, first match wins, exactly the pre-registered readings."""
    any_primary = [k for k, v in crit.items() if v.get("primary_success")]
    any_strong = [k for k, v in crit.items() if v.get("strong_compression_success")]
    any_accled = [k for k, v in crit.items() if v.get("accuracy_led_success")]

    # The content-free control.  `shuffled_minus_honest` is C1-SHUF - C1, so a
    # scorer that carries image-specific signal makes it NEGATIVE and negative
    # on every seed.  A control that does not separate leaves it at zero.
    shuf = audit["matched_seed_deltas"].get("shuffled_minus_honest")
    shuf_gap = (-shuf["mean_pts"]) if shuf else None
    score_carries_signal = bool(shuf and shuf_gap >= 1.0
                                and all(x < 0 for x in shuf["per_seed_pts"]))

    # P4: is the L4 full-token forward + compaction structurally too expensive?
    c1 = eff.get("C1", {})
    st = c1.get("stages", {})
    gdep_prefix = sum(st.get(k, 0.0) for k in
                      ("L0_L4_ms", "scorer_ms", "selector_ms", "token_compaction_ms"))
    b1 = eff.get("B1", {})
    saved = (b1.get("ttft_wall_median_ms", 0.0)
             - c1.get("ttft_wall_median_ms", 0.0))
    l4_too_late = bool(c1 and b1 and saved <= 0)

    if any_accled or any_primary or any_strong:
        cleared = sorted(set(any_accled + any_primary + any_strong))
        statement = ("GDEP is a new Pareto point: " + ", ".join(cleared)
                     + (" each clear" if len(cleared) > 1 else " clears")
                     + " the pre-registered bar.")
        detail = dict(accuracy_led=any_accled, primary=any_primary,
                      strong_compression=any_strong,
                      score_specific=score_carries_signal,
                      shuffled_minus_honest_pts=(shuf["mean_pts"] if shuf else None))
        # P1's pre-registered antecedent is "a 5.2 or 5.3 bar holds AND the gain
        # survives C1-SHUF".  When the first half holds and the second does not,
        # that is reported rather than rounded away.
        detail["prereg_P1_antecedent_met"] = bool(score_carries_signal)
        if not score_carries_signal:
            statement += (" The Pareto point exists, but P1's second antecedent "
                          "does not: the content-free shuffled-score control does "
                          f"not separate from the honest arm (gap "
                          f"{shuf_gap if shuf_gap is not None else float('nan'):+.2f} "
                          "macro points, below the 1.0-point attribution margin), "
                          "so the position on the frontier is a property of "
                          "pruning to 256 tokens on a map with these statistics, "
                          "not of the learned score.")
        return dict(code="P1", statement=statement, detail=detail)

    # no Pareto bar cleared: is it accuracy that failed, or the mechanism?
    best_acc = max((v.get("vs_B1_macro_pts", -1e9) for v in crit.values()),
                   default=float("-inf"))
    min_drop = min((-(v.get("vs_B1_macro_pts") or 0.0) for v in crit.values()),
                   default=float("inf"))
    if l4_too_late:
        return dict(code="P4", statement=(
            "The L4 pruning point is too late: the L0-L4 full-token forward plus "
            f"compaction ({gdep_prefix:.1f} ms of the prefill) consumes the whole "
            "saving, so end-to-end TTFT does not fall. The mechanism cannot pay "
            "for itself at this depth; the distillation has to move into the "
            "vision encoder / pre-LLM stage."),
            detail=dict(gdep_prefix_ms=gdep_prefix,
                        ttft_saved_vs_B1_ms=saved))
    if best_acc < 0 and min_drop > CRIT["compression"]["acc_drop_pts"]:
        return dict(code="P3", statement=(
            "GDEP is faster but the accuracy loss is unacceptable: the best "
            f"candidate trails B1 by {min_drop:.2f} macro points, above the "
            "pre-registered 2.0-point bar."),
            detail=dict(best_vs_B1_pts=best_acc))
    if best_acc >= 0:
        return dict(code="P2", statement=(
            "GDEP has better accuracy but the performance cost is too large: "
            "accuracy is at or above B1 but no pre-registered efficiency bar is "
            "met."), detail=dict(best_vs_B1_pts=best_acc))
    if score_carries_signal:
        return dict(code="P5", statement=(
            "The learned score works but the selector / system implementation is "
            "the bottleneck: the honest arm beats the content-free shuffled "
            f"control by {shuf_gap:.2f} macro points on every seed, and no "
            "candidate clears a Pareto bar."),
            detail=dict(shuffled_gap_pts=shuf_gap,
                        shuffled_minus_honest_pts=shuf["mean_pts"]))
    return dict(code="P3", statement=(
        "No candidate clears a Pareto bar and the shuffled control does not "
        "separate from the honest arm; the configuration as implemented is not "
        "better than the incumbent on either axis."),
        detail=dict(best_vs_B1_pts=best_acc, shuffled_gap_pts=shuf_gap))


def _tables(audit):
    L = []
    A = L.append
    A("# M2 — GDEP system Pareto: machine-generated tables\n")
    A(f"Correctness gates all passed: **{audit['correctness']}**\n")
    A("## Accuracy (macro over the three benchmarks)\n")
    A("| arm | seed | TextVQA | DocVQA | OCRBench | macro |")
    A("|---|---|---|---|---|---|")
    for k, v in sorted(audit["accuracy_table"].items()):
        A(f"| {v['arm']} | {v['seed']} | {v['per_benchmark']['TextVQA_VAL']:.3f} "
          f"| {v['per_benchmark']['DocVQA_VAL']:.3f} "
          f"| {v['per_benchmark']['OCRBench']:.3f} | {v['macro_pct']:.3f} |")
    A("\n## Efficiency\n")
    A("| arm | TTFT median (ms) | TTFT p90 | prefill event median | peak MB | "
      "KV early (MB) | KV late (MB) | TPOT (ms) | rps |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(audit["efficiency_table"].items()):
        A(f"| {k} | {v['ttft_wall_median_ms']:.2f} | {v['ttft_wall_p90_ms']:.2f} "
          f"| {v['prefill_event_median_ms']:.2f} | {v['peak_allocated_mb']:.0f} "
          f"| {v['kv_bytes_early']/1e6:.2f} | {v['kv_bytes_late']/1e6:.2f} "
          f"| {v['tpot_ms']:.3f} | {v['throughput_rps']:.2f} |")
    A("\n## Stage breakdown (mean ms)\n")
    keys = ["image_preprocess_ms", "vision_encoder_ms", "eadp_scoring_ms",
            "selector_ms", "L0_L4_ms", "scorer_ms", "token_compaction_ms",
            "llm_forward_ms"]
    A("| arm | " + " | ".join(k.replace("_ms", "") for k in keys) + " |")
    A("|" + "---|" * (len(keys) + 1))
    for k, v in sorted(audit["efficiency_table"].items()):
        A(f"| {k} | " + " | ".join(f"{v['stages'].get(x, 0.0):.3f}" for x in keys) + " |")
    A("\n## Pre-registered criteria\n")
    A("| arm | vs B1 (pts) | 95 % CI | TTFT red. | peak red. | primary | "
      "compression | accuracy-led |")
    A("|---|---|---|---|---|---|---|---|")
    for k, v in sorted(audit["criteria"].items()):
        if not v.get("evaluated"):
            A(f"| {k} | — | — | — | — | not evaluated | | |")
            continue
        A(f"| {k} | {v['vs_B1_macro_pts']:+.2f} | "
          f"[{v['vs_B1_ci_pts'][0]:+.2f}, {v['vs_B1_ci_pts'][1]:+.2f}] | "
          f"{100*v['ttft_reduction_vs_B1']:.1f} % | "
          f"{100*v['peak_reduction_vs_B1']:.1f} % | "
          f"{v['primary_success']} | {v['strong_compression_success']} | "
          f"{v['accuracy_led_success']} |")
    A("\n## Matched-seed deltas (macro points)\n")
    A("| contrast | mean | 95 % CI | per seed | all positive |")
    A("|---|---|---|---|---|")
    for k, v in audit["matched_seed_deltas"].items():
        A(f"| {v['a']} − {v['b']} | {v['mean_pts']:+.3f} | "
          f"[{v['ci_pts'][0]:+.3f}, {v['ci_pts'][1]:+.3f}] | "
          f"{', '.join(f'{x:+.2f}' for x in v['per_seed_pts'])} | "
          f"{v['all_seeds_positive']} |")
    A(f"\n## Conclusion\n\n```\n{audit['conclusion']['code']}: "
      f"{audit['conclusion']['statement']}\n```\n")
    with open(os.path.join(OUTPUT_DIR, "m2_tables.md"), "w") as f:
        f.write("\n".join(L))
    print("[saved] m2_tables.md")


if __name__ == "__main__":
    main()
