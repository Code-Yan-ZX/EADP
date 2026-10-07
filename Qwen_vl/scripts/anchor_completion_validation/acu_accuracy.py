"""Anchor Completion Validation — generation + official scoring driver.

Full-split resumable shards per (panel, arm, dataset).  Scoring semantics
(frozen):
  * official VLMEvalKit evaluators;
  * S0 hard failure: empty official dict / evaluate() exception ->
    score_failures.jsonl + nonzero exit, NO partial _score.json;
  * per-question scores: TextVQA mean-of-matches; DocVQA ANLS-correct
    (hit = 0 if 1-minANLS < 0.5 else 1-minANLS); OCRBench verbatim official
    branch structure (D-4: math branch no lowercase, strip + \n->' ' both
    sides); generic datasets: eval_score / eval_match column of the official
    results file when present;
  * per-question mean MUST reproduce the official headline inside its
    rounding granularity (hard-fail beyond; never a ±0.5 slack).

Usage:
  python acu_accuracy.py --panel main --arms BASE,MAIN025 --mode both
  python acu_accuracy.py --panel main --arms BASE --mode gen --smoke 5
  python acu_accuracy.py --panel nonreg --arms BASE,MAIN025 --mode both
  python acu_accuracy.py --panel fresh --arms MAIN100,MAIN_SIM025 --mode both
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time

import torch
from statistics import mean as _mean

import acu_common as AU
from acu_common import common as C

FAIL_LOG = os.path.join(AU.OUT_DIR, "score_failures.jsonl")


def fail(arm, ds, kind, detail):
    entry = dict(arm=arm, ds=ds, kind=kind, detail=str(detail))
    with open(FAIL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[score FAIL] {arm} {ds}: {kind}: {detail}", flush=True)
    return entry


def panel_rows(panel: str, ds: str):
    """Row indices for a panel.  main/nonreg: ALL official rows.
    fresh: the frozen fresh-panel manifest rows."""
    if panel in ("main", "nonreg"):
        dataset = C.build_dataset(ds)
        return list(range(len(dataset.data))), dataset
    if panel == "fresh":
        man = json.load(open(os.path.join(AU.OUT_DIR,
                                          "fresh_panel_manifest.json")))
        d = man["datasets"][ds]
        if d["status"] != "ok":
            return [], None
        return d["rows"], C.build_dataset(ds)
    raise KeyError(panel)


def run_gen(args, eng):
    for ds in (args.ds.split(",") if args.ds else AU.DS_ALL):
        if args.panel == "main" and ds not in AU.DS_MAIN:
            continue
        if args.panel == "nonreg" and ds not in AU.DS_NONREG:
            continue
        if args.panel == "fresh" and ds not in AU.DS_MAIN:
            continue   # fresh panel covers main tasks only
        bank = AU.load_bank(ds)
        rows, dataset = panel_rows(args.panel, ds)
        if not rows:
            print(f"[skip] {args.panel}/{ds}: empty row list", flush=True)
            continue
        eng.vlm.set_dump_image(dataset.dump_image)
        t0 = time.time()
        for arm in args.arms.split(","):
            if args.panel == "fresh" and arm in AU.ARMS_FORMAL:
                # protocol §6.3: BASE/MAIN025 fresh-panel predictions are
                # TAKEN from the verified full-panel run (same config,
                # same rows) — never regenerated on the fresh panel.
                print(f"[skip] fresh/{arm} {ds}: formal-method predictions "
                      f"are reused from the main panel (frozen rule)",
                      flush=True)
                continue
            cfg = AU.arm_cfg(arm)
            path = AU.shard_path(args.panel, arm, ds)
            shard = AU.load_shard(path)
            if "meta" not in shard:
                shard["meta"] = AU.shard_meta(args.panel, arm, ds,
                                              AU.repo_commit(),
                                              args.max_new_tokens)
            for n, i in enumerate(rows):
                key = str(i)
                if key in shard["records"] or key not in bank:
                    continue
                rec_bank = bank[key]
                row = dataset.data.iloc[i]
                msg = C.build_message(eng.vlm, dataset, ds, row)
                timings = {}
                out = AU.run_one(eng, msg, ds, rec_bank, cfg,
                                 max_new_tokens=args.max_new_tokens,
                                 timings=timings)
                shard["records"][key] = dict(
                    prediction=out["text"],
                    truncated=len(out["gen_ids"]) >= args.max_new_tokens,
                    n_vis_kept=int(out["meta"]["n_vis_kept"]),
                    n_vis=int(rec_bank.get("n_vis", -1)),
                    ttft_ms=timings.get("ttft_ms"),
                    merge_ms=timings.get("merge_ms"),
                    vision_ms=timings.get("vision_ms"),
                    llm_prefill_ms=timings.get("llm_prefill_ms"),
                    layer_calls_ok=bool(out["meta"]["layer_calls_ok"]),
                    degeneracy=AU.degeneracy(out["text"]))
                if len(shard["records"]) % 20 == 0:
                    AU.save_shard(path, shard)
                if n % 20 == 0:
                    el = time.time() - t0
                    print(f"[{args.panel}/{arm} {ds}] {n}/{len(rows)} "
                          f"({el/max(1, n+1):.2f}s/q)", flush=True)
            AU.save_shard(path, shard)
            print(f"[done] {args.panel}/{arm} {ds}: "
                  f"{len(shard['records'])} records", flush=True)


def _headline(official: dict):
    for v in official.values():
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, (int, float)):
                    return float(vv)
        elif isinstance(v, (int, float)):
            return float(v)
    return None


def _per_q_ocerbench(shard, ds):
    from vlmeval.dataset import build_dataset as _bd
    dso = _bd(ds)
    per_q = {}
    for k, rec in shard["records"].items():
        row = dso.data.iloc[int(k)]
        predict = str(rec["prediction"]).strip()
        answers = ast.literal_eval(row["answer"])
        hit = 0.0
        if row["category"] == "Handwritten Mathematical Expression Recognition":
            p2 = predict.replace("\n", " ").replace(" ", "")
            if any(a.strip().replace("\n", " ").replace(" ", "") in p2
                   for a in answers):
                hit = 1.0
        else:
            pr = predict.lower().strip().replace("\n", " ")
            if any(a.lower().strip().replace("\n", " ") in pr
                   for a in answers):
                hit = 1.0
        per_q[str(k)] = hit
    return per_q


def _per_q_generic(tsv, ds, shard):
    """Extract per-row scores from the official results file when present.

    File naming differs per dataset (verified 2026-10-02 against live
    evaluate() runs):
      ChartQA/TextVQA/DocVQA : <stem>_results.xlsx  (eval_score / eval_match)
      MMStar/RealWorldQA/MMBench: <stem>_exact_matching_result.xlsx (hit)
      POPE                   : <stem>_auxmatch.xlsx (score bool per row)
    """
    import numpy as np
    import pandas as pd
    stem = tsv[:-4] if tsv.endswith(".tsv") else tsv
    candidates = [stem + "_results.xlsx", stem + "_results.tsv",
                  stem + "_exact_matching_result.xlsx",
                  stem + "_auxmatch.xlsx"]
    positions = sorted(int(k) for k in shard["records"])
    for det in candidates:
        if not os.path.exists(det):
            continue
        d = pd.read_excel(det) if det.endswith("xlsx") \
            else pd.read_csv(det, sep="\t")
        col_hit = ("hit" if "hit" in d.columns
                   else "score" if "score" in d.columns else None)

        def _int_set(col):
            # some official aux files (e.g. HallusionBench) use string
            # question ids in `index`; treat unparseable index as absent
            try:
                return set(col.astype(int))
            except (ValueError, TypeError):
                return None

        idx_int = _int_set(d["index"]) if "index" in d.columns else None
        # index-mapping when the file's index column IS the row position;
        # otherwise the results file preserves input order -> map by order
        idx_match = (idx_int is not None and idx_int == set(positions)
                     and len(d) == len(positions))
        if "eval_score" in d.columns and idx_int is not None \
                and set(positions) <= idx_int \
                and len(d) >= len(positions):
            m = dict(zip(d["index"].astype(int),
                         d["eval_score"].astype(float)))
            return {str(p): float(m[p]) for p in positions}
        if "eval_match" in d.columns and len(d) == len(positions):
            per_q = {}
            for pos, (_, r) in zip(positions, d.iterrows()):
                m = r["eval_match"]
                if isinstance(m, str):
                    m = ast.literal_eval(m)
                m = list(m)
                if ds == "DocVQA_VAL":
                    md = float(np.min(m))
                    hit = 0.0 if 1 - md < 0.5 else 1 - md
                else:
                    hit = float(np.mean(m))
                per_q[str(pos)] = hit
            return per_q
        if idx_match:
            m = dict(zip(d["index"].astype(int), d[col_hit].astype(float)))
            return {str(p): float(1.0 if v else 0.0) if col_hit == "score"
                    else float(v) for p, v in ((p, m[p]) for p in positions)}
        if len(d) == len(positions):
            # order-preserving mapping (e.g. MMBench 'index' is the dataset's
            # own question id, not the iloc row)
            vals = d[col_hit].astype(float).tolist()
            return {str(p): float(1.0 if v else 0.0) if col_hit == "score"
                    else float(v) for p, v in zip(positions, vals)}
    return None


def _headline100(official: dict, ds: str):
    """Headline on the 0-100 scale.  POPE official primary metric is
    accuracy ('acc'; the 'Overall' key is F1 — disclosed in the report);
    MMStar/RealWorldQA/MMBench return 0-1 fractions and are scaled."""
    if ds == "POPE" and "acc" in official:
        v = official["acc"]
        val = v[0] if isinstance(v, dict) else v
        return float(val)
    if "Overall" in official:
        v = official["Overall"]
        val = v[0] if isinstance(v, dict) else v
        if isinstance(val, (int, float)):
            h = float(val)
            if h <= 1.0000001:
                h *= 100.0
            return h
    h = _headline(official)
    if h is not None and h <= 1.0000001:
        h *= 100.0
    return h


def derive_fresh_formal(args, arm, ds):
    """Fresh-panel score entry for BASE/MAIN025: subset of the verified
    main-panel per-question scores over the frozen fresh rows (no new
    generation — provenance recorded)."""
    man = json.load(open(os.path.join(AU.OUT_DIR,
                                      "fresh_panel_manifest.json")))
    d = man["datasets"][ds]
    if d["status"] != "ok":
        return
    main_score_path = AU.shard_path("main", arm, ds).replace(
        ".json", "_score.json")
    if not os.path.exists(main_score_path):
        return
    main_score = json.load(open(main_score_path))
    want = [str(i) for i in d["rows"]]
    per_q = {k: main_score["per_question"][k] for k in want
             if k in main_score["per_question"]}
    if not per_q:
        return
    path = AU.shard_path("fresh", arm, ds)
    summary = dict(panel="fresh", arm=arm, ds=ds, n=len(per_q),
                   derived_from=dict(
                       panel="main", shard=os.path.relpath(
                           AU.shard_path("main", arm, ds), AU.QWEN_ROOT),
                       rule="same-config same-row reuse (protocol §6.3)",
                       fresh_manifest_sha256=man["sha256"]),
                   official={},
                   per_question=per_q,
                   perq_vs_headline_diff=None,
                   perq_reproduces_headline=None,
                   base_commit=AU.repo_commit())
    with open(path.replace(".json", "_score.json"), "w") as f:
        json.dump(summary, f, indent=1)
    acc = 100.0 * sum(float(v) for v in per_q.values()) / len(per_q)
    print(f"[fresh-derived] {arm} {ds}: n={len(per_q)} acc={acc:.3f}",
          flush=True)


def run_score(args):
    from vlmeval.dataset import build_dataset as vlmeval_build
    import pandas as pd

    failures = []
    if args.panel == "fresh":
        for ds in AU.DS_MAIN:
            for arm in args.arms.split(","):
                if arm in AU.ARMS_FORMAL:
                    derive_fresh_formal(args, arm, ds)
    for ds in (args.ds.split(",") if args.ds else AU.DS_ALL):
        if args.panel == "main" and ds not in AU.DS_MAIN:
            continue
        if args.panel == "nonreg" and ds not in AU.DS_NONREG:
            continue
        if args.panel == "fresh" and ds not in AU.DS_MAIN:
            continue
        for arm in args.arms.split(","):
            path = AU.shard_path(args.panel, arm, ds)
            shard = AU.load_shard(path)
            done = sorted(int(k) for k in shard["records"])
            if not done:
                continue
            dataset = vlmeval_build(ds)
            data = dataset.data
            sub = data.iloc[done].copy()
            for col in ("image",):
                if col in sub.columns:
                    sub = sub.drop(columns=[col])
            sub["prediction"] = [shard["records"][str(i)]["prediction"]
                                 for i in done]
            sub["truncated"] = [shard["records"][str(i)]["truncated"]
                                for i in done]
            tsv = path.replace(".json", "_pred.tsv")
            sub.to_csv(tsv, sep="\t", index=False)
            try:
                res = dataset.evaluate(tsv)
            except Exception as e:
                failures.append(fail(arm, ds, "evaluate_exception", e))
                continue
            if hasattr(res, "to_dict"):
                res = res.to_dict()
            if not res:
                failures.append(fail(arm, ds, "empty_official",
                                     f"evaluate returned nothing tsv={tsv}"))
                continue
            if ds == "OCRBench":
                per_q = _per_q_ocerbench(shard, ds)
            else:
                per_q = _per_q_generic(tsv, ds, shard)
            # strict headline reproduction check
            chk, diff = None, None
            if per_q:
                m = 100.0 * sum(float(v) for v in per_q.values()) / len(per_q)
                if ds == "OCRBench" and "Final Score" in res:
                    h = 100.0 * float(res["Final Score"]) / len(done)
                    diff = abs(m - h)
                    chk = diff <= 0.05
                elif ds == "POPE":
                    # official 'acc' = mean over the category-EXPLODED frame
                    # (a row in k categories counts k times) — reproduce it
                    # exactly; the per-question currency stays row-level.
                    cats = [str(dataset.data.iloc[int(k)]["category"])
                            .split(",") for k in done]
                    exploded = [float(per_q[str(k)]) for k, cs in
                                zip(done, cats) for _ in cs]
                    h = _headline100(res, ds)
                    diff = abs(100.0 * float(_mean(exploded)) - h)
                    chk = diff <= 1e-6
                    m = 100.0 * float(_mean([float(v) for v in
                                              per_q.values()]))
                else:
                    h = _headline100(res, ds)
                    if h is not None:
                        diff = abs(m - h)
                        # headline rounding granularity: fail beyond 0.05
                        chk = diff <= 0.05
            if chk is False:
                failures.append(fail(
                    arm, ds, "perq_headline_mismatch",
                    f"perq_mean={m:.6f} headline={h:.6f} diff={diff:.6f}"))
                continue
            summary = dict(panel=args.panel, arm=arm, ds=ds, n=len(done),
                           official={k: (float(v) if isinstance(v, (int, float))
                                         else str(v))
                                     for k, v in res.items()},
                           per_question=per_q,
                           perq_vs_headline_diff=diff,
                           perq_reproduces_headline=chk,
                           base_commit=AU.repo_commit())
            with open(path.replace(".json", "_score.json"), "w") as f:
                json.dump(summary, f, indent=1)
            print(f"[scored] {args.panel}/{arm} {ds}: n={len(done)} "
                  f"diff={None if diff is None else round(diff, 6)} "
                  f"official={{'main': {_headline(res)}}}", flush=True)
    if failures:
        raise SystemExit(f"run_score: {len(failures)} scoring failure(s) "
                         f"recorded in {FAIL_LOG}; affected _score.json "
                         f"NOT written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True,
                    choices=["main", "nonreg", "fresh"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    ap.add_argument("--ds", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--smoke", type=int, default=None,
                    help="limit each ds to first N bank-covered rows")
    args = ap.parse_args()

    if args.smoke:
        # smoke = first N rows per ds that already have a bank entry
        orig = panel_rows

        def limited(panel, ds):
            rows, dataset = orig(panel, ds)
            bank = AU.load_bank(ds)
            rows = [i for i in rows if str(i) in bank][:args.smoke]
            return rows, dataset
        globals()["panel_rows"] = limited

    if args.mode in ("gen", "both"):
        eng = AU.load_engine(max_new_tokens=args.max_new_tokens)
        run_gen(args, eng)
    if args.mode in ("score", "both"):
        run_score(args)


if __name__ == "__main__":
    main()
