"""Consolidate every S2-B artifact into outputs/discovery/s2b_accuracy_translation.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT

DS = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
# Stage-1 reference timings (ms) from docs/eadp_method_discovery.md §4.2/§4.4
STAGE1 = {"eadp256_e2e_ms": 244.8, "unpruned_e2e_ms": 522.1,
          "pruned_prefill_ms": 77.99, "unpruned_prefill_ms": 228.43,
          "eadp_prune_overhead_ms": 41.96, "facility_selector_ms": 40.15,
          "eadp_scoring_ms": 0.92}


def load(p):
    q = os.path.join(OUT, p)
    return json.load(open(q)) if os.path.exists(q) else None


def main():
    full = json.load(open(os.path.join(OUT, "s2b_full.json")))["runs"]
    ana = json.load(open(os.path.join(OUT, "s2b_full_analysis.json")))
    trans = json.load(open(os.path.join(OUT, "s2b_full_transitions.json")))
    pilot = json.load(open(os.path.join(OUT, "s2b_pilot_analysis.json")))
    ident = json.load(open(os.path.join(OUT, "s2b_identity_analysis.json")))
    ctrl = json.load(open(os.path.join(OUT, "s2b_ctrl_analysis.json")))
    conc = json.load(open(os.path.join(OUT, "s2b_concentration.json")))
    gmeta = json.load(open(os.path.join(OUT, "s2b_gradient_scores_meta.json")))

    # timing
    fwd = float(np.mean([v["fwd_ms"] for v in gmeta.values()]))
    bwd = float(np.mean([v["bwd_ms"] for v in gmeta.values()]))
    peak = float(np.mean([v["peak_mb"] for v in gmeta.values()]))
    # per-selector, not averaged across selectors (topk is ~0.7 ms, facility ~42 ms)
    sel_by = {s: float(np.mean([v["select_ms_mean"] for v in full.values()
                                if v["selector"] == s]))
              for s in {v["selector"] for v in full.values()}}
    prune_by = {s: float(np.mean([v["prune_ms_mean"] for v in full.values()
                                  if v["selector"] == s]))
                for s in {v["selector"] for v in full.values()}}
    sel_ms, prune_ms = sel_by["facility"], prune_by["facility"]
    timing = {
        "gradient_forward_ms": fwd, "gradient_backward_ms": bwd,
        "gradient_peak_mem_mb": peak,
        "selection_ms": sel_ms, "prune_total_ms": prune_ms,
        "selection_ms_by_selector": sel_by, "prune_ms_by_selector": prune_by,
        "oracle_total_ms": fwd + bwd + STAGE1["eadp256_e2e_ms"],
        "oracle_note": "gradient fwd+bwd plus the full untouched EADP-256 e2e "
                       "(selection + pruned prefill + generation). Not deployable.",
        "stage1_reference": STAGE1,
        "oracle_vs_eadp256_e2e": (fwd + bwd + STAGE1["eadp256_e2e_ms"]) / STAGE1["eadp256_e2e_ms"],
        "oracle_vs_unpruned_e2e": (fwd + bwd + STAGE1["eadp256_e2e_ms"]) / STAGE1["unpruned_e2e_ms"],
    }

    out = {
        "setting": {
            "frozen_set": "Stage-1 Part 2 sample bank: sample_indices(len(data), 150, 0) "
                          "per benchmark, n=450, indices verified identical to "
                          "diag_selectors_b256.json",
            "budget": 256, "pilot": "every 3rd index of the frozen 150 (n=50/dataset)",
            "score": "P1 + gradient x input (G2): k=argmax(logits at first answer "
                     "position); J=logits[k]; s_i=sum_d |dJ/dv_i,d * v_i,d|",
            "baseline": "diag_selectors_b256.json facility @256, reused not recomputed",
        },
        "harness_identity_check": {
            "arm": "official scoring through this runner (override=None)",
            "per_dataset": {d: ident["summary"]["facility|official"]["per_dataset"][d]
                            for d in DS},
            "macro_delta": ident["summary"]["facility|official"]["macro_delta"],
            "win_tie_loss": [ident["summary"]["facility|official"][k]
                             for k in ("win", "tie", "loss")],
            "verdict": "exact reproduction of the Stage-1 numbers on all three benchmarks",
        },
        "shuffled_score_control": {
            "arm": "C2 calibration, score->instance assignment rotated by 7",
            "summary": ctrl["summary"]["facility|C2+shuf7"],
            "verdict": "macro delta +0.97, CI spans zero => the gain is content-specific",
        },
        "pilot": pilot["summary"],
        "full": ana["summary"],
        "full_diagnostics": ana["diagnostics"],
        "full_transitions": ana["transitions"],
        "transitions_by_category": trans,
        "spatial_concentration": conc,
        "timing": timing,
    }
    # flat accuracy table for convenience
    acc = {}
    for k, v in full.items():
        acc.setdefault(f"{v['selector']}|{v['calibration']}", {})[v["dataset"]] = v["acc_pct"]
    out["accuracy_pct"] = acc

    json.dump(out, open(os.path.join(OUT, "s2b_accuracy_translation.json"), "w"), indent=1)
    print("[saved]", os.path.join(OUT, "s2b_accuracy_translation.json"))
    print(f"  oracle total {timing['oracle_total_ms']:.0f} ms "
          f"= {timing['oracle_vs_eadp256_e2e']:.1f}x EADP-256 e2e, "
          f"{timing['oracle_vs_unpruned_e2e']:.2f}x unpruned e2e")


if __name__ == "__main__":
    main()
