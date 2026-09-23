"""
M2 interim snapshot — read-only export of everything the *completed* artefacts
already contain, plus whatever the read-only diagnostic produced.

Writes:
  Qwen_vl/outputs/discovery/m2_system_interim.json     (machine-readable)
  docs/scoring_search_m2_system_interim.md             (human-readable)

This carries **no verdict**, changes no arm, re-selects no seed and does not
touch the grid. It exists to decide one question: is the C0/C1 degradation
explained by the method, by an objective mismatch, or by an implementation
defect in the generation/cache path?

Usage
    python scripts/discovery/m2_interim.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR                                     # noqa: E402
from m2_gdep import dump_json                                     # noqa: E402

DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
SEEDS = (0, 1, 2)
NBOOT = 10000

ARM_DEFS = {
    "B0": dict(
        mode="full", pruning="none", selector="none", scorer="none",
        cache="36 layers x 1058 tokens, unpruned",
        path="engine `full` mode: manual 36-layer loop over the whole sequence"),
    "B1": dict(
        mode="prellm", pruning="none inside the LLM (prune happens before it)",
        selector="facility (official, `_greed_select_impl`)",
        scorer="official EADP importance (alpha 0.5, beta 2.0)",
        cache="36 layers x 290 tokens; L0-L4 cache does not exist",
        path="incumbent: pruner over the vision embeddings, then 36 layers"),
    "B2": dict(
        mode="prellm", pruning="none inside the LLM",
        selector="block8 (D1: same objective, T/8 super-steps)",
        scorer="official EADP importance",
        cache="36 layers x 290 tokens",
        path="incumbent with the cheap selector"),
    "C0": dict(
        mode="gdep", pruning="layer 4 (0-based), mid-stack",
        selector="topk", scorer="M1 LOCAL-MLP, n=240, seed 0, 524 673 params",
        cache="L0-L4 1058 -> compacted to 290; L5-L35 290",
        path="GDEP: 5 full-length layers, online score, select, compact, 31 layers"),
    "C1": dict(
        mode="gdep", pruning="layer 4", selector="topk",
        scorer="M1 LOCAL-MLP, n=960, seeds 0/1/2",
        cache="L0-L4 1058 -> 290; L5-L35 290", path="GDEP primary candidate"),
    "C2": dict(
        mode="gdep", pruning="layer 4", selector="block8",
        scorer="M1 LOCAL-MLP, n=960, seeds 0/1/2",
        cache="L0-L4 1058 -> 290; L5-L35 290", path="GDEP secondary candidate"),
    "C3": dict(
        mode="gdep", pruning="layer 4", selector="facility",
        scorer="M1 LOCAL-MLP, n=960, seeds 0/1/2",
        cache="L0-L4 1058 -> 290; L5-L35 290", path="GDEP secondary candidate"),
    "C1-R": dict(
        mode="gdep", pruning="layer 4", selector="topk",
        scorer="M1 LOCAL-MLP, n=960, seeds 0/1/2",
        cache="L0-L4 1058 -> 290; L5-L35 290",
        path="C1 with RENUMBER: compacted positions reassigned arange(0,290) "
             "instead of the original ids -- the one declared control that "
             "changes only the position policy"),
    "C1-SHUF": dict(
        mode="gdep", pruning="layer 4", selector="topk",
        scorer="M1 LOCAL-MLP n=960, but the score map rotated onto another "
               "image (derangement by +7 instances)",
        cache="L0-L4 1058 -> 290; L5-L35 290",
        path="content-free control: identical statistics, no image-specific "
             "signal. If it matches C1, the learned score is doing no work"),
}


def load(name):
    p = os.path.join(OUTPUT_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def boot_macro(a_hits, b_hits, nb=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    draws = np.zeros(nb)
    for ds in DS_ORDER:
        a, b = np.asarray(a_hits[ds], float), np.asarray(b_hits[ds], float)
        n = len(a)
        idx = rng.integers(0, n, size=(nb, n))
        draws += (a[idx].mean(1) - b[idx].mean(1)) / len(DS_ORDER)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    obs = float(np.mean([np.asarray(a_hits[ds], float).mean()
                         - np.asarray(b_hits[ds], float).mean()
                         for ds in DS_ORDER]))
    return dict(mean_pts=obs * 100, lo_pts=float(lo) * 100,
                hi_pts=float(hi) * 100, n_boot=nb)


def hit_stats(hits):
    h = np.asarray(hits, float)
    return dict(n=int(len(h)), correct=int((h >= 0.5).sum()),
                wrong=int((h < 0.5).sum()), frac=float((h >= 0.5).mean()),
                mean_hit=float(h.mean()))


def transitions(a, b, cut=0.5):
    A = np.concatenate([np.asarray(a[ds], float) for ds in DS_ORDER])
    B = np.concatenate([np.asarray(b[ds], float) for ds in DS_ORDER])
    ac, bc = A >= cut, B >= cut
    return dict(rescued=int((~bc & ac).sum()),
                newly_broken=int((bc & ~ac).sum()),
                both_correct=int((bc & ac).sum()),
                both_wrong=int((~bc & ~ac).sum()), n=int(len(A)))


# ---------------------------------------------------------------------------
def perf_table(perf):
    rows = {}
    for tag, r in perf["arms"].items():
        if "prefill" not in r:
            continue
        p = r["prefill"]
        st = {k: v["mean"] for k, v in p["stages"].items()}
        sp = r.get("speedup_vs", {})
        rows[tag] = dict(
            label=r.get("label"),
            model_prefill_event_median_ms=p["event"]["median"],
            model_prefill_event_p90_ms=p["event"]["p90"],
            ttft_wall_median_ms=p["wall"]["median"],
            ttft_wall_p90_ms=p["wall"]["p90"],
            ttft_wall_std_ms=p["wall"]["std"],
            vision_encoder_ms=st.get("vision_encoder_ms", 0.0),
            image_preprocess_ms=st.get("image_preprocess_ms", 0.0),
            eadp_scoring_ms=st.get("eadp_scoring_ms", 0.0),
            selector_ms=st.get("selector_ms", 0.0),
            L0_L4_full_token_ms=st.get("L0_L4_ms", 0.0),
            scorer_ms=st.get("scorer_ms", 0.0),
            token_kv_compaction_ms=st.get("token_compaction_ms", 0.0),
            L5_plus_ms=st.get("llm_forward_ms", 0.0),
            decode32_ms=r["decode"]["total_cuda"]["median"],
            tpot_ms=r["decode"]["tpot_ms"],
            peak_mem_mb=r["peak_allocated_mb"],
            kv_bytes_early=r["prefill"]["kv_bytes_early"],
            kv_bytes_late=r["prefill"]["kv_bytes_late"],
            kv_bytes_total=r["prefill"]["kv_bytes_total"],
            kv_total_if_no_prune=r["prefill"]["kv_total_if_no_prune"],
            context_len=p["context_len"], n_kept=p["n_kept"],
            cache_policy=p["cache_policy"], pos_policy=p["pos_policy"],
            throughput_rps=r["throughput_rps"],
            extra_forward_or_recompute=r["extra_forward_or_recompute"],
            speedup_vs_B0=sp.get("B0"), speedup_vs_B1=sp.get("B1"),
            speedup_vs_B2=sp.get("B2"))
    return rows


def bucket(xs, edges=(50, 200, 800, 2000)):
    xs = np.asarray(xs, float)
    out = {}
    prev = 0
    for e in edges:
        out[f"{prev}-{e}"] = int(((xs >= prev) & (xs < e)).sum())
        prev = e
    out[f"{prev}+"] = int((xs >= prev).sum())
    return out


def len_stats(preds):
    L = np.array([len(p) for p in preds], float)
    capped = L > 2000
    return dict(mean_chars=float(L.mean()), median_chars=float(np.median(L)),
                p90_chars=float(np.percentile(L, 90)), max_chars=int(L.max()),
                microlen_buckets=bucket(L),
                n_capped=int(capped.sum()), frac_capped=float(capped.mean()),
                n_terminated=int((~capped).sum()),
                frac_terminated=float((~capped).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="m2_system_interim.json")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    perf = load("m2_perf.json")
    acc = load("m2_accuracy.json")
    corr = load("m2_correctness.json")
    diag = load("m2_interim_diag.json")
    d5 = load("m2_interim_d5.json")

    snap = dict(
        stage="M2", kind="INTERIM READ-ONLY SNAPSHOT",
        note=("No verdict. Does not modify the pre-registration, the arm set, "
              "the seed set or the running grid. Produced while the accuracy "
              "grid was still executing."),
        generated=__import__("time").strftime("%F %T"),
        arm_definitions=ARM_DEFS,
        completed_accuracy_runs=sorted(acc["arms"]) if acc else [],
        missing_accuracy_runs=None,
        correctness_all_passed=(corr or {}).get("all_passed"),
        environment=(corr or {}).get("environment"),
    )

    # expected vs done
    expected = ["B0", "B1", "B2"] + [
        f"{a}|s{s}" for a in ("C0", "C1", "C2", "C3", "C1-R", "C1-SHUF")
        for s in SEEDS]
    done = set(acc["arms"]) if acc else set()
    snap["missing_accuracy_runs"] = [k for k in expected if k not in done]

    # ------------------------------------------------------------------ perf --
    if perf:
        snap["performance"] = perf_table(perf)
        b1 = snap["performance"].get("B1", {})
        b2 = snap["performance"].get("B2", {})
        b0 = snap["performance"].get("B0", {})
        for tag, r in snap["performance"].items():
            r["ttft_reduction_vs_B1"] = (1 - r["ttft_wall_median_ms"]
                                         / b1["ttft_wall_median_ms"]
                                         if b1 else None)
            r["model_prefill_reduction_vs_B1"] = (
                1 - r["model_prefill_event_median_ms"]
                / b1["model_prefill_event_median_ms"] if b1 else None)

    # -------------------------------------------------------------- accuracy --
    if acc and "B1" in acc["arms"]:
        A = acc["arms"]
        rows, deltas = {}, {}
        for key, r in sorted(A.items()):
            if "error" in r:
                continue
            ent = dict(arm=r["arm"], seed=r["seed"], macro_pct=r["macro_pct"],
                       per_benchmark={ds: r["per_benchmark"][ds]["acc_pct"]
                                      for ds in DS_ORDER},
                       per_benchmark_hits={ds: hit_stats(
                           r["per_benchmark"][ds]["hits"]) for ds in DS_ORDER},
                       all_hits=hit_stats(np.concatenate(
                           [np.asarray(r["per_benchmark"][ds]["hits"], float)
                            for ds in DS_ORDER])),
                       gen_len=len_stats(r["predictions"]),
                       n_kept_mean=r.get("n_kept_mean"))
            ent["gen_len_per_benchmark"] = {}
            for ds in DS_ORDER:
                sel = [i for i, m in enumerate(r["per_image_meta"])
                       if m["ds"] == ds]
                ent["gen_len_per_benchmark"][ds] = len_stats(
                    [r["predictions"][i] for i in sel])
            rows[key] = ent
            if key != "B1":
                d = {ds: float(r["per_benchmark"][ds]["acc_pct"]
                               - A["B1"]["per_benchmark"][ds]["acc_pct"])
                     for ds in DS_ORDER}
                ci = boot_macro(hits_of(r), hits_of(A["B1"]))
                deltas[key] = dict(
                    arm=r["arm"], seed=r["seed"],
                    vs_B1_macro_pts=float(np.mean(list(d.values()))),
                    vs_B1_per_benchmark_pts=d, vs_B1_ci_pts=[ci["lo_pts"], ci["hi_pts"]],
                    vs_B1_ci_excludes_zero=bool(ci["lo_pts"] > 0 or ci["hi_pts"] < 0),
                    transitions_vs_B1=transitions(hits_of(r), hits_of(A["B1"])),
                    transitions_vs_B0=(transitions(hits_of(r), hits_of(A["B0"]))
                                       if "B0" in A else None))
        snap["accuracy"] = rows
        snap["accuracy_vs_B1"] = deltas

        # seed means per arm
        sm = {}
        for arm in ("C0", "C1", "C2", "C3", "C1-R", "C1-SHUF"):
            rs = [rows.get(f"{arm}|s{s}") for s in SEEDS]
            have = [r for r in rs if r]
            if not have:
                continue
            sm[arm] = dict(n_seeds=len(have),
                           seeds_present=[r["seed"] for r in have],
                           macro_seed_mean=float(np.mean([r["macro_pct"]
                                                          for r in have])),
                           macro_values=[r["macro_pct"] for r in have],
                           macro_range=[float(min(r["macro_pct"] for r in have)),
                                        float(max(r["macro_pct"] for r in have))],
                           vs_B1=[deltas[f"{arm}|s{s}"]["vs_B1_macro_pts"]
                                  for s in SEEDS if f"{arm}|s{s}" in deltas],
                           capped_rate=[r["gen_len"]["frac_capped"] for r in have])
            sm[arm]["complete"] = len(have) == 3
        snap["seed_means"] = sm

        # runaway instance list (10+)
        run = {}
        for key, r in sorted(A.items()):
            if "error" in r:
                continue
            capped = [i for i, p in enumerate(r["predictions"]) if len(p) > 2000]
            if not capped:
                continue
            entries = []
            for i in capped[:12]:
                meta = r["per_image_meta"][i]
                txt = r["predictions"][i]
                entries.append(dict(
                    key=acc["keys"][i], benchmark=meta["ds"],
                    context_len=meta.get("context_len"),
                    n_kept=meta.get("n_kept"),
                    kv_seq_len=meta.get("kv_seq_len"),
                    first_tokens=txt[:60].replace("\n", " "),
                    last_tokens=txt[-60:],
                    chars=len(txt),
                    looping=bool(len(txt) > 200 and
                                 (txt[-40:] in txt[:-40] or
                                  len(set(txt[-200:])) < 20))))
            run[key] = dict(arm=r["arm"], seed=r["seed"],
                            n_capped=len(capped), frac=len(capped) / 150.0,
                            keys=[acc["keys"][i] for i in capped],
                            examples=entries)
        snap["runaway_instances"] = run

    if diag:
        snap["diagnostics"] = diag
    if d5:
        snap["diagnostic_D5_position_isolation"] = d5

    # ---- the coverage audit the brief asks for -----------------------------
    snap["gate_coverage"] = dict(
        note=("Does the A-E gate set cover the generation-path properties the "
              "interim audit needs? Answered item by item; the two gaps were "
              "closed by the read-only diagnostics D1/D2/D5 rather than by "
              "changing the grid."),
        items={
            "1_K1024_full_multistep_generation_vs_stock": dict(
                covered_by="D1 (not G-A..G-E)",
                result="engine and HF generate agree token-for-token on the "
                       "identity path (D1)"),
            "2_first_decode_step_logits": dict(
                covered_by="D1 prefill argmax + G-C prefill logits",
                result="prefill argmax == HF's first generated token; prefill "
                       "logits identical (G-C, 0.000e+00)"),
            "3_first_8_decode_steps": dict(
                covered_by="D1 (16 steps requested; runs to EOS)",
                result="per-step token equality recorded"),
            "4_eos_and_stopping_criteria": dict(
                covered_by="D1 (both paths) + code inspection",
                result="generation_config.eos_token_id = [151645, 151643]; the "
                       "engine uses that same set; same stop behaviour observed "
                       "in D1"),
            "5_cache_position_after_compaction": dict(
                covered_by="D2 (not A-E)",
                result="decode advances cache_position 287,288,289 while the "
                       "cache length is 287 pre-decode; GAP FOUND: the PRESERVE "
                       "branch freezes the RoPE position instead of advancing it"),
            "6_mrope_position_ids_and_rope_deltas": dict(
                covered_by="D2 + D5 (not A-E)",
                result="engine passes explicit (3,1,S) ids; rope_deltas is not "
                       "used in inputs_embeds mode. GAP FOUND: PRESERVE did not "
                       "advance its ids across decode steps"),
            "7_kv_lengths_L0_L4_and_L5_plus": dict(
                covered_by="G-D + D2",
                result="L0-L4 compacted 1058->290; L5-L35 = 290; D2 confirms "
                       "shapes [1,8,287,128] -> [1,8,290,128] after 3 steps"),
            "8_generation_configuration_identical": dict(
                covered_by="code inspection + G-F",
                result="every arm greedy do_sample=False, same max_new_tokens; "
                       "B1 reproduces the published numbers exactly, which is "
                       "the strongest available check"),
        },
        gaps_found=["5 (PRESERVE position advance)", "6 (same, from the mRoPE "
                    "side)"])

    dump_json(args.out_json, snap)
    print(f"[saved] {args.out_json}")


def hits_of(rec):
    return {ds: rec["per_benchmark"][ds]["hits"] for ds in DS_ORDER}


if __name__ == "__main__":
    main()