"""Consolidate all S2-A artifacts into the single deliverable JSON."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import BLOCK, MAPS, OUT
from scoring_search_s1 import evaluate

METHODS = ["A1_G1", "A1_G2", "A1_G1L1", "P1_G1", "P1_G2", "P2_G1", "P2_G2"]
ALL = ["official", "s1_best"] + METHODS
TIERS = {"loc64": lambda n: n == 64, "loc256": lambda n: n <= 256, "all": lambda n: True}


def main():
    D = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))
    recs, scores = D["records"], D["scores"]
    keys = sorted(recs)
    Z = np.load(os.path.join(OUT, "s2a_null_control.npz"))
    N = json.load(open(os.path.join(OUT, "s2a_null_control.json")))
    norms = {k: v.astype(np.float64)
             for k, v in np.load(os.path.join(OUT, "s2a_input_norm.npz")).items()}
    rng = np.random.default_rng(0)

    def spear(a, b):
        return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

    cases = {}
    for k in keys:
        needed = np.array(recs[k]["needed"])
        c = dict(ds=recs[k]["ds"], idx=recs[k]["idx"], needed=[int(x) for x in needed],
                 nnec=int(len(needed) * BLOCK * BLOCK),
                 answer=recs[k]["answer"], n_answer_tokens=recs[k]["n_answer_tokens"],
                 n_answer_tokens_raw=recs[k]["n_answer_tokens_raw"],
                 A1_nll=recs[k].get("A1_nll"), P_top1_token=recs[k].get("P_top1_token"),
                 P_margin=recs[k].get("P_margin"))
        sc = {m: np.asarray(scores[k][m], dtype=np.float64) for m in METHODS}
        sc["official"] = np.load(os.path.join(MAPS, f"probe_{k}.npz"))["importance"].astype(np.float64)
        sc["s1_best"] = np.load(os.path.join(MAPS, f"{k}_b256.npz"))["global_sim"].reshape(-1).astype(np.float64)
        sc["ctrl_input_norm"] = norms[k]
        sc["ctrl_random"] = rng.random(1024)
        sc["null_mismatched_ans_G1"] = Z[f"{k}__G1"].astype(np.float64)
        sc["null_mismatched_ans_G2"] = Z[f"{k}__G2"].astype(np.float64)
        c["metrics"] = {m: evaluate(dict(needed=needed), s) for m, s in sc.items()}
        c["matched_vs_mismatched_spearman_G2"] = spear(sc["A1_G2"], sc["null_mismatched_ans_G2"])
        c["mismatched_answer"] = N[k]["mismatched_answer"]
        c["mismatched_nll"] = N[k]["nll"]
        cases[k] = c

    def agg(method, fn, field):
        v = np.array([cases[k]["metrics"][method][field] for k in keys if fn(cases[k]["nnec"])])
        o = np.array([cases[k]["metrics"]["official"][field] for k in keys if fn(cases[k]["nnec"])])
        d = v - o
        return dict(n=len(v), mean=float(np.nanmean(v)), official=float(np.nanmean(o)),
                    delta=float(np.nanmean(d)),
                    improved=int(np.sum(d < -1e-9)), worsened=int(np.sum(d > 1e-9)))

    aggregate = {}
    for tk, fn in TIERS.items():
        aggregate[tk] = {m: {f: agg(m, fn, f) for f in
                             ("mean_rank_pct", "median_rank_pct", "block_auroc", "block_ap",
                              "recall@128", "recall@256", "recall@512")}
                         for m in ALL + ["ctrl_input_norm", "ctrl_random",
                                         "null_mismatched_ans_G1", "null_mismatched_ans_G2"]}

    # gates
    gates = {"A": {}, "B": {}}
    for m in ["A1_G1", "A1_G2", "A1_G1L1"]:
        a, au = aggregate["loc256"][m]["mean_rank_pct"], aggregate["loc256"][m]["block_auroc"]
        gates["A"][m] = dict(delta=-a["delta"], auroc=au["mean"], improved=a["improved"],
                             passed=bool(-a["delta"] >= 0.10 and au["mean"] >= 0.65
                                         and a["improved"] >= 6))
    gates["A_passed"] = any(v["passed"] for v in gates["A"].values())
    for m in ["P1_G1", "P1_G2", "P2_G1", "P2_G2"]:
        a, au = aggregate["loc256"][m]["mean_rank_pct"], aggregate["loc256"][m]["block_auroc"]
        gates["B"][m] = dict(delta=-a["delta"], auroc=au["mean"], improved=a["improved"],
                             passed=bool((au["mean"] >= 0.60 or -a["delta"] >= 0.08)
                                         and a["improved"] >= 6))
    gates["B_passed"] = gates["A_passed"] and any(v["passed"] for v in gates["B"].values())

    eff = {f: float(np.nanmean([recs[k].get(f, np.nan) for k in keys]))
           for f in ("t_A2_fwd_ms", "t_P1_bwd_ms", "t_A1_fwd_ms", "t_A1_bwd_ms",
                     "P1_peak_mem_mb")}

    json.dump({
        "meta": {"n_cases": len(keys), "methods": METHODS,
                 "objective_A1": "J = sum_t log p(a_t | image, question, a_<t); teacher forcing on the "
                                 "unpruned model's own correct answer (ref_prediction), capped at 64 tokens",
                 "objective_P1": "J = logit[argmax] at the first answer position of the prompt-only forward",
                 "objective_P2": "J = logit[top1] - logit[top2] at the same position",
                 "saliency_G1": "||grad_i||_2", "saliency_G1L1": "||grad_i||_1",
                 "saliency_G2": "sum_d |grad_i,d * v_i,d|",
                 "answer_token_cap": 64,
                 "autograd_note": "importing vlmeval.config calls torch.set_grad_enabled(False) globally; "
                                  "the probe re-enables it explicitly"},
        "hook_sanity": json.load(open(os.path.join(OUT, "s2a_hook_sanity.json"))),
        "cases": cases, "aggregate": aggregate, "gates": gates, "efficiency": eff,
    }, open(os.path.join(OUT, "s2a_gradient_viability.json"), "w"), indent=1)

    print("consolidated ->", os.path.join(OUT, "s2a_gradient_viability.json"))
    print(f"\nGATE A passed: {gates['A_passed']}   GATE B passed: {gates['B_passed']}")
    sp = [cases[k]["matched_vs_mismatched_spearman_G2"] for k in keys]
    print(f"matched-vs-mismatched saliency Spearman: mean {np.mean(sp):.3f}")
    print(f"legacy hook_sanity file kept as s2a_hook_sanity.json")


if __name__ == "__main__":
    main()
