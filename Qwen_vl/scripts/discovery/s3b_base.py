"""
S3-B step 1 (GPU): the base points, plus the stage gates.

A0 measures, for all 150 held-out instances and BOTH bases, the delivered
F(0) = S set: official accuracy under greedy generation, teacher-forced gold
NLL, and this instance's bf16 resolution floor. This is what makes "wrong" in
Part A mean *wrong in this regime* rather than wrong in some other run's regime
(the two disagree on ~15 % of instances), and it gives the honest denominator
for how often a small bundle rescues at all.

Gates (all must pass before the base loop; `--gates-only` runs just them):

  B1  recomputed bank-G top-256 == the live GDEP engine's select_idx (3 probes)
  B2  bank-L reuse: arms of the frozen S2-C2 rescue grid, re-generated with the
      S3-B harness, return the recorded prediction text and hit label
  B3  harness equivalence: bank-G F(0) at cap 64 reproduces the frozen
      s3a_nll.json base predictions exactly (6 probes on the shared instances)
  B4  determinism: the same arm scored twice is bit-identical
  B5  delivery: the spliced visual rows are exactly the cached vision-tower
      embeddings of the arm's set (max-abs-diff 0)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                    # noqa: E402
from s1_audit import OUT                                         # noqa: E402
import s3b_common as B                                           # noqa: E402
import s3b_gpu as G                                              # noqa: E402
from scoring import per_sample_hits                              # noqa: E402

B3_CAP = 64         # the S3-A harness cap, reproduced exactly in gate B3
B2_ARMS = 6
B3_PROBES = 6


# ---------------------------------------------------------------------------
def run_gates(h, model, datasets, banks, bundles_L):
    rep = {}

    # ---- B1: bank G S == live engine select_idx ---------------------------
    from m2_gdep import GDEPConfig, GDEPEngine, MODE_GDEP
    cfg = GDEPConfig(mode=MODE_GDEP, budget=B.BUDGET, selector="topk",
                     n_arm=B.STUDENT_N_ARM, seed=B.STUDENT_SEED, tag="S3B-B1")
    eng = GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=8)
    rows = []
    for rec in list(banks["G"].values())[:3]:
        ds_obj = datasets[rec["ds"]]
        model.set_dump_image(ds_obj.dump_image)
        msg = common.build_message(model, ds_obj, rec["ds"],
                                   ds_obj.data.iloc[rec["idx"]])
        st, info = eng.prefill(eng.prepare(msg, rec["ds"]))
        sel = sorted(int(t) for t in info["select_idx"])
        rows.append(dict(key=rec["key"], identical=bool(sel == sorted(rec["S"])),
                         symdiff=len(set(sel) ^ set(rec["S"]))))
        del st
        torch.cuda.empty_cache()
    rep["B1_engine_select"] = dict(rows=rows,
                                   passed=all(r["identical"] for r in rows))
    print(f"[B1] {rep['B1_engine_select']['passed']} {rows}")

    # ---- B2: bank-L arms reproduce S2-C2 ----------------------------------
    swap = json.load(open(os.path.join(OUT, "s2c2_rescue.json")))["runs"]
    cand = []
    for arm, r in sorted(swap.items()):
        if not arm.startswith("teacher:") or not r.get("keys"):
            continue
        k = int(arm.split(":")[1].split("|")[0])
        if k > B.KS_MAX:
            continue
        for key, pred, hit in zip(r["keys"], r["predictions"], r["hits"]):
            cand.append((k, key, pred, float(hit), arm))
    # prefer arms the frozen grid scored CORRECT: a reuse gate on a passing arm
    # proves the protocol delivers the same set, not just the same failure.
    picks, per_ds, used_k = [], {ds: 0 for ds in B.DS_ALL}, set()
    for c in sorted(cand, key=lambda x: (-x[3], x[0], x[1])):
        ds = c[1].rsplit("_", 1)[0]
        if per_ds[ds] >= B2_ARMS // 3 or c[0] in used_k:
            continue
        picks.append(c)
        per_ds[ds] += 1
        used_k.add(c[0])
        if len(picks) >= B2_ARMS:
            break
    rows = []
    for k, key, pred_ref, hit_ref, arm in picks:
        rec = banks["L"][key]
        ds_obj = datasets[rec["ds"]]
        prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
        adds, drops = B.prefix_arms(rec, k)
        ids, text = h.generate(prep, B.apply_arms(rec, adds, drops),
                               max_new=2048)
        pred = h.vlm._post_process_response(text)
        hit = float(per_sample_hits(rec["ds"], [row], [pred])[0])
        rows.append(dict(key=key, arm=arm, k=k, text_match=bool(pred == pred_ref),
                         hit_match=bool(abs(hit - hit_ref) < 1e-6),
                         pred=pred[:60], pred_ref=pred_ref[:60],
                         hit=hit, hit_ref=hit_ref))
        print(f"[B2] {key} {arm}: text={rows[-1]['text_match']} "
              f"hit={rows[-1]['hit_match']} ({hit:.2f} vs {hit_ref:.2f})")
    rep["B2_s2c2_reuse"] = dict(rows=rows, passed=all(
        r["text_match"] and r["hit_match"] for r in rows))

    # ---- B3: harness == frozen S3-A base preds (cap 64) -------------------
    s3a = json.load(open(os.path.join(OUT, "s3a_nll.json")))["preds"]
    rows, n = [], 0
    for rec in banks["G"].values():
        if rec["key"] not in s3a or n >= B3_PROBES:
            continue
        n += 1
        prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
        ids, text = h.generate(prep, rec["S"], max_new=B3_CAP)
        pred = h.vlm._post_process_response(text)
        rows.append(dict(key=rec["key"], text_match=bool(
            pred == s3a[rec["key"]]["pred"]),
            pred=pred[:50], pred_s3a=s3a[rec["key"]]["pred"][:50]))
        print(f"[B3] {rec['key']}: {rows[-1]['text_match']}")
    rep["B3_harness_equiv"] = dict(rows=rows,
                                   passed=all(r["text_match"] for r in rows))

    # ---- B4: determinism ---------------------------------------------------
    rec = list(banks["G"].values())[0]
    prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
    a = G.score_set(h, prep, ans, rec, [], [], row)
    b = G.score_set(h, prep, ans, rec, [], [], row)
    rep["B4_determinism"] = dict(L_equal=bool(a["L"] == b["L"]),
                                 text_equal=bool(a["pred"] == b["pred"]),
                                 passed=bool(a["L"] == b["L"]
                                             and a["pred"] == b["pred"]))
    print(f"[B4] {rep['B4_determinism']['passed']}")

    # ---- B5: the spliced rows ARE the cached vision embeddings ------------
    rows = []
    for rec in (list(banks["G"].values())[1], list(banks["L"].values())[2]):
        prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
        adds, drops = B.prefix_arms(rec, 8)
        sel = B.apply_arms(rec, adds, drops)
        got = h._prompt(prep, sel)
        n_pre = prep["n_prefix"]
        direct = prep["vis"].index_select(
            0, torch.as_tensor(sorted(sel), device=h.dev))
        mad = float((got[n_pre:n_pre + len(sel)] - direct).abs().max())
        rows.append(dict(key=rec["key"], max_abs_diff=mad,
                         n_visual=int(got.shape[0] - n_pre
                                      - prep["suffix"].shape[0])))
        print(f"[B5] {rec['key']}: mad={mad:.3e}")
    rep["B5_delivery"] = dict(rows=rows, passed=all(
        r["max_abs_diff"] == 0.0 and r["n_visual"] == B.BUDGET
        for r in rows))

    rep["all_passed"] = all(rep[x]["passed"] for x in
                            ("B1_engine_select", "B2_s2c2_reuse",
                             "B3_harness_equiv", "B4_determinism",
                             "B5_delivery"))
    return rep


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--tag", default="s3b_base")
    args = ap.parse_args()

    cases = json.load(open(os.path.join(OUT, B.CASES_JSON)))
    banks = {b: {r["key"]: r for r in cases["banks"][b]} for b in ("G", "L")}
    order = [r["key"] for r in cases["banks"]["G"]]
    bundles_L = cases["bundles_L"]

    out_path = os.path.join(OUT, f"{args.tag}.json")
    results = dict(config=cases["config"], done=[], recs={})
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev.get("config") != results["config"]:
            raise RuntimeError("existing s3b_base.json under a different config")
        results["done"], results["recs"] = prev.get("done", []), prev.get("recs", {})
        print(f"[resume] {len(results['done'])} instances done")

    datasets = {ds: common.build_dataset(ds) for ds in B.DS_ALL}
    model, h = G.load_stack()
    gate_path = os.path.join(OUT, "s3b_gates.json")
    if os.path.exists(gate_path) and not args.gates_only:
        rep = json.load(open(gate_path))
        print(f"[gates] reusing s3b_gates.json all_passed={rep['all_passed']}")
    else:
        rep = run_gates(h, model, datasets, banks, bundles_L)
        json.dump(rep, open(gate_path, "w"), indent=1)
        print(f"[gates] all_passed={rep['all_passed']}")
    if args.gates_only or not rep["all_passed"]:
        sys.exit(0 if args.gates_only and rep["all_passed"] else 1)

    todo = [k for k in order if k not in results["done"]]
    t0 = time.time()
    for n, key in enumerate(todo):
        rec = banks["G"][key]
        try:
            prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
            floor = G.measure_floor(h, prep, ans, rec)
            ent = dict(key=key, ds=rec["ds"], idx=rec["idx"], floor=floor,
                       n_golds=len(golds), banks={})
            for bank in ("G", "L"):
                r = banks[bank][key]
                m = G.score_set(h, prep, ans, r, [], [], row, kept=r["S"])
                ent["banks"][bank] = dict(hit=m["hit"], L=m["L"],
                                          pred=m["pred"], n_tokens=m["n_tokens"])
            results["recs"][key] = ent
            results["done"].append(key)
        except Exception:
            traceback.print_exc()
            print(f"[skip] {key}")
            continue
        if (n + 1) % 10 == 0 or n == len(todo) - 1:
            hg = ent["banks"]["G"]["hit"]
            hl = ent["banks"]["L"]["hit"]
            print(f"[{n + 1}/{len(todo)}] {key} hit_G={hg:.2f} hit_L={hl:.2f} "
                  f"{time.time() - t0:.0f}s")
        tmp = out_path + ".tmp"
        json.dump(results, open(tmp, "w"))
        os.replace(tmp, out_path)
    acc = {b: {ds: [] for ds in B.DS_ALL} for b in ("G", "L")}
    for e in results["recs"].values():
        for b in ("G", "L"):
            acc[b][e["ds"]].append(e["banks"][b]["hit"])
    print("[base] F(0) accuracy by bank/benchmark:")
    for b in ("G", "L"):
        print("   ", b, {ds: round(float(np.mean(v)) * 100, 2)
                         for ds, v in acc[b].items()})
    print(f"[saved] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
