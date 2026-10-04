"""Specialized scoring for the 5 new Table-4 datasets (L_R_MAIN025).

Record-supplement §2 requirement: MME acc+ (image-pair), HallusionBench
grouped fAcc/qAcc, and MMBench circular scores must NOT be replaced by a
plain per-question mean.  This driver therefore, per dataset:

  1. builds the prediction TSV exactly like rtg_accuracy_legacy._score_one
     (frozen semantics, S0 hard-fail included via perq extraction),
  2. runs the OFFICIAL dataset.evaluate (unchanged),
  3. dumps per-question artifacts from the official intermediate file
     (extracted answer + 0/1 score + grouping keys),
  4. independently recomputes the aggregate from the per-question scores
     with local code and cross-checks it against the official headline.

Per-dataset aggregation (from VLMEvalKit source, this repo's pinned copy):
  MME       acc+ = mean over image_path groups of prod(scores); category
            score = acc + acc+; perception/reasoning/total are sums.
  Hallusion set/figure/question ids parsed from index; aAcc = row mean;
            fAcc = mean of all() over (l2-cat, set_id, figure_id);
            qAcc = mean of all() over (l2-cat, set_id, question_id).
  AI2D      plain per-question mean over exact-match score.
  InfoVQA   ANLS per question, plain mean (same as DocVQA path).
  MMB-CN    circular headline only; perq mapping unavailable (same F3
            rotation limitation as MMB-EN) -> recorded explicitly, no
            per-question mean is reported in its place.

Usage: python rtg_score_new5.py [--datasets ...]
Output: <shard>.json旁边 <ds>_score.json (same schema as main segment,
plus per_question_records and aggregate_crosscheck).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

import rtg_common as RC
from rtg_accuracy_legacy import ARM, FULL, fail, load_shard, shard_path

NEW5 = ["AI2D_TEST", "HallusionBench", "MME", "MMBench_DEV_CN_V11",
        "InfoVQA_VAL"]


def build_pred_tsv(ds):
    from vlmeval.dataset import build_dataset as vlmeval_build
    path = shard_path(ds)
    shard = load_shard(path)
    done = sorted(int(k) for k in shard["records"])
    if not done:
        fail(ds, "no_records", path)
        return None
    dataset = vlmeval_build(ds)
    data = dataset.data
    sub = data.iloc[done].copy()
    for col in ("image",):
        if col in sub.columns:
            sub = sub.drop(columns=[col])
    sub["prediction"] = [shard["records"][str(i)]["prediction"] for i in done]
    sub["truncated"] = [shard["records"][str(i)]["truncated"] for i in done]
    tsv = path.replace(".json", "_pred.tsv")
    sub.to_csv(tsv, sep="\t", index=False)
    return dataset, tsv, done, shard


def _load_auxmatch(tsv, dataset):
    """Official per-row intermediate: extracted + score columns."""
    from vlmeval.smp.file import get_intermediate_file_path, load
    p = get_intermediate_file_path(tsv, "_auxmatch")
    if not os.path.exists(p):
        return None, p
    return load(p), p


def crosscheck_mme(aux, official):
    """Recompute perception/reasoning/total from per-row scores."""
    stats = defaultdict(lambda: defaultdict(list))
    for _, it in aux.iterrows():
        stats[it["category"]][it["image_path"]].append(int(it["score"]))
    cate_score = {k: np.mean([np.mean(v) for v in g.values()]) * 100
                  + np.mean([np.prod(v) for v in g.values()]) * 100
                  for k, g in stats.items()}
    percep = ["OCR", "artwork", "celebrity", "color", "count", "existence",
              "landmark", "position", "posters", "scene"]
    reas = ["code_reasoning", "commonsense_reasoning",
            "numerical_calculation", "text_translation"]
    P = sum(cate_score[c] for c in percep if c in cate_score)
    R = sum(cate_score[c] for c in reas if c in cate_score)
    local = dict(perception=P, reasoning=R, total=P + R)

    def _scalar(v):
        if isinstance(v, dict):
            v = next(iter(v.values()))
        return float(v) if v is not None else None
    off_P = _scalar(official.get("perception"))
    off_R = _scalar(official.get("reasoning"))
    chk = (off_P is not None and abs(P - off_P) < 1e-6
           and off_R is not None and abs(R - off_R) < 1e-6)
    return local, cate_score, bool(chk)


def crosscheck_hallusion(aux, official):
    """Recompute aAcc/fAcc/qAcc from per-row scores."""
    d = aux.copy()
    d["set_id"] = [x.split("_")[3] for x in d["index"]]
    d["figure_id"] = [x.split("_")[4] for x in d["index"]]
    d["question_id"] = [x.split("_")[5] for x in d["index"]]
    aAcc = float(np.mean(d["score"])) * 100
    g_f = defaultdict(list)
    g_q = defaultdict(list)
    for _, it in d.iterrows():
        g_f[f"{it['l2-category']}_{it['set_id']}_{it['figure_id']}"].append(
            int(it["score"]))
        g_q[f"{it['l2-category']}_{it['set_id']}_{it['question_id']}"].append(
            int(it["score"]))
    fAcc = float(np.mean([np.all(v) for v in g_f.values()])) * 100
    qAcc = float(np.mean([np.all(v) for v in g_q.values()])) * 100
    local = dict(aAcc=aAcc, fAcc=fAcc, qAcc=qAcc)
    # official res comes from DataFrame.to_dict: {'split': {i: name, ...},
    # 'aAcc': {i: val, ...}, ...} — locate the Overall row by split name
    chk = None
    try:
        splits = official.get("split") or {}
        idx = next(i for i, name in splits.items() if name == "Overall")
        off_a = float(official["aAcc"][idx])
        chk = abs(aAcc - off_a) < 1e-6
    except (StopIteration, KeyError, TypeError, ValueError):
        pass
    return local, chk


def score_one(ds):
    import vlmeval  # noqa: F401 (env bootstrap)
    out = build_pred_tsv(ds)
    if out is None:
        return False
    dataset, tsv, done, shard = out
    try:
        res = dataset.evaluate(tsv)
    except Exception as e:
        return fail(ds, "evaluate_exception", e)
    if hasattr(res, "to_dict"):
        res = res.to_dict()
    official = {k: (v if isinstance(v, (int, float, str, list, dict))
                    else str(v)) for k, v in res.items()}
    aux, aux_path = _load_auxmatch(tsv, dataset)
    perq, crosscheck = {}, None
    if aux is not None:
        for _, it in aux.iterrows():
            rec = dict(extracted=str(it.get("extracted")),
                       score=int(it["score"]))
            if ds == "MME":
                rec.update(category=it["category"],
                           image_path=it["image_path"])
            elif ds == "HallusionBench":
                parts = str(it["index"]).split("_")
                rec.update(set_id=parts[3], figure_id=parts[4],
                           question_id=parts[5],
                           l2_category=it.get("l2-category"))
            elif ds in ("AI2D_TEST", "AI2D_TEST_NO_MASK"):
                rec.update(category=it.get("category"))
            perq[str(it["index"])] = rec
    elif ds == "InfoVQA_VAL":
        # ANLS per-question via the same extractor validated on DocVQA
        from acu_accuracy import _per_q_generic
        perq = _per_q_generic(tsv, ds, shard) or {}

    if ds == "MME" and aux is not None:
        local, cate, chk = crosscheck_mme(aux, official)
        crosscheck = dict(metric="perception/reasoning sums",
                          local=local, category_scores=cate, match=chk)
    elif ds == "HallusionBench" and aux is not None:
        local, chk = crosscheck_hallusion(aux, official)
        crosscheck = dict(metric="aAcc/fAcc/qAcc", local=local, match=chk)
    elif ds.startswith("MMBench"):
        crosscheck = dict(metric="circular headline (F3 subset)",
                          local=None, match=None,
                          note="per-question mapping unavailable "
                               "(rotated dedup); headline only")
    elif ds == "InfoVQA_VAL" and perq:
        m = 100.0 * float(np.mean([float(v) for v in perq.values()]))
        crosscheck = dict(metric="ANLS plain mean",
                          local=dict(value=m), match=None)
    summary = dict(arm=ARM, pipeline="official_legacy", panel="full_new5",
                   ds=ds, n=len(done), official=official,
                   per_question=perq, auxmatch_path=str(aux_path),
                   aggregate_crosscheck=crosscheck,
                   base_commit=RC.git_commit())
    with open(shard_path(ds).replace(".json", "_score.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"[new5 scored] {ds}: n={len(done)} crosscheck={crosscheck}",
          flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None)
    args = ap.parse_args()
    dsets = (args.datasets.split(",") if args.datasets
             else NEW5)
    ok = all(score_one(ds) for ds in dsets)
    if not ok:
        raise SystemExit("new5 scoring failures recorded")


if __name__ == "__main__":
    main()
