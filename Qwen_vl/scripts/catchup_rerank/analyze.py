"""CPU-only paired exploratory QA analysis; official DocVQA ANLS is secondary.

Scoring extracts the exact pure functions from the checked-in official
VLMEvalKit source. No model, torch or VLMEvalKit package import is needed.
"""
from __future__ import annotations
import argparse
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import numpy as np

QWEN = Path(__file__).resolve().parents[2]
ARMS = ("full", "rres512", "baseline", "dense_full", "dense_2k", "ltc_2k")
SEED = 20261108
DRAWS = 10000
CONTRASTS = tuple(("baseline", arm) for arm in ARMS if arm != "baseline") + (
    ("dense_2k", "ltc_2k"), ("dense_full", "dense_2k"))


def pure_functions(path, names, namespace):
    content = Path(path).read_text()
    functions = [node for node in ast.parse(content).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise RuntimeError(f"Scoring functions missing from {path}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return dict(source=str(path), sha256=hashlib.sha256(content.encode()).hexdigest(),
                extracted_functions=sorted(names))


def scorers():
    official = QWEN / "VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py"
    namespace = {"np": np,
                 "listinstr": lambda patterns, value: any(pattern in value for pattern in patterns)}
    anls_meta = pure_functions(official, {"levenshtein_distance", "anls_compute", "hit_calculate"}, namespace)

    def score_anls(prediction, answers):
        distances = [namespace["anls_compute"](answer, prediction) for answer in answers]
        # anls_compute is NORMALIZED DISTANCE. The official hit function takes
        # its best-alias complement, then applies the official 0.5 threshold.
        return float(namespace["hit_calculate"]([{"match": distances}], "DocVQA")[0])

    primary_source = QWEN / "scripts/cf_mining/cf_mine.py"
    primary_ns = {"re": re, "Decimal": Decimal, "InvalidOperation": InvalidOperation}
    primary_meta = pure_functions(primary_source, {"normalize", "correct"}, primary_ns)

    def score_exact(prediction, answers):
        return any(primary_ns["correct"](prediction, answer) for answer in answers)

    return score_exact, score_anls, dict(
        primary_exact={**primary_meta,
                       "semantics": "Runner's Decimal numeric exact: strip outer quotes, lowercase, remove whitespace/currency/comma, canonicalize Unicode minus and Decimal; preserve percent suffix/sign; leading zeros are not preserved."},
        secondary_anls={**anls_meta,
                        "semantics": "Official original-string anls_compute distance followed by hit_calculate DocVQA similarity/threshold; no Decimal or currency normalization."},
        literal_strict_exact={"semantics": "Raw generated string equals at least one original answer alias exactly; no stripping, whitespace, case, currency, comma, percent or numeric normalization."},
    )


def intervals(values, draws):
    return np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()


def summarize(vectors, records, binary=True):
    n = len(records)
    if n == 0:
        return dict(n=0, arms={}, paired={}, note="No complete untruncated pairs; not estimable")
    # Each source parent contributes exactly one question; these indices are
    # document bootstrap draws, shared across arms and contrasts.
    draws = np.random.default_rng(SEED).integers(0, n, size=(DRAWS, n))
    arms = {}
    for arm in ARMS:
        vector = np.asarray(vectors[arm], dtype=float)
        entry = dict(n=n, mean=float(vector.mean()), bootstrap_95ci=intervals(vector, draws))
        if binary:
            entry.update(correct=int(vector.sum()), accuracy=float(vector.mean()))
        arms[arm] = entry
    paired = {}
    for before, after in CONTRASTS:
        delta = np.asarray(vectors[after], dtype=float) - np.asarray(vectors[before], dtype=float)
        entry = dict(before=before, after=after, n=n,
                     delta_mean=float(delta.mean()), bootstrap_95ci=intervals(delta, draws),
                     improved=int((delta > 0).sum()), worsened=int((delta < 0).sum()),
                     tied=int((delta == 0).sum()))
        if binary:
            entry.update(delta_accuracy=float(delta.mean()), rescue=int((delta == 1).sum()),
                         breakage=int((delta == -1).sum()))
        if bool((delta == 0).all()):
            entry["interval_note"] = "Identical observed outcomes; a degenerate interval does not establish algorithmic equivalence."
        paired[after + "_minus_" + before] = entry
    return dict(n=n, arms=arms, paired=paired)


def analyze(x, manifest=None):
    rows = x["records"]
    if not x.get("complete") or len(rows) != x["config"]["sample_count"] or len(rows) != 16:
        raise ValueError("Primary analysis requires all 16 frozen samples, including errors/truncations")
    ids = [r["parent_id"] for r in rows]
    item_ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)) or len(item_ids) != len(set(item_ids)):
        raise ValueError("Duplicate source parent or question")
    if manifest is not None:
        raw = Path(manifest).read_bytes()
        frozen = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != x["config"]["manifest_sha256"]:
            raise ValueError("Data manifest hash differs from runner's frozen manifest")
        by_id = {item["item_id"]: item for item in frozen["items"]}
        if set(item_ids) != set(by_id) or len(by_id) != 16:
            raise ValueError("Predictions do not cover exactly the 16 frozen questions")
        for row in rows:
            item = by_id[row["item_id"]]
            for field in ("parent_id", "question", "answers"):
                if row[field] != item[field]:
                    raise ValueError(f"Frozen annotation changed: {row['item_id']} {field}")

    exact, score_anls, metadata = scorers()
    scores = {metric: {arm: [] for arm in ARMS} for metric in ("exact", "anls", "literal")}
    detail = []
    for row in rows:
        answers = row["answers"]
        if not answers or not all(isinstance(answer, str) for answer in answers):
            raise ValueError("Original human answer aliases must be nonempty strings")
        item = dict(item_id=row["item_id"], parent_id=row["parent_id"],
                    question=row["question"], answers=answers, predictions={})
        for arm in ARMS:
            prediction = row["predictions"][arm]
            text = prediction["text"]
            if not isinstance(text, str):
                raise ValueError("Missing or non-string prediction")
            primary = bool(exact(text, answers))
            if "correct" in prediction and primary != bool(prediction["correct"]):
                raise ValueError(f"Runner/scorer primary exact disagreement: {row['item_id']} {arm}")
            literal = any(text == answer for answer in answers)
            anls = score_anls(text, answers)
            scores["exact"][arm].append(primary)
            scores["anls"][arm].append(anls)
            scores["literal"][arm].append(literal)
            item["predictions"][arm] = dict(text=text, exact=primary,
                                              secondary_anls=anls, literal_strict_exact=literal,
                                              numeric_literal_disagreement=primary != literal,
                                              truncated=bool(prediction["truncated"]),
                                              token_count=len(prediction.get("token_ids", [])))
        detail.append(item)

    primary = summarize(scores["exact"], rows)
    secondary = summarize(scores["anls"], rows, binary=False)
    literal = summarize(scores["literal"], rows)
    for arm in ARMS:
        primary["arms"][arm]["truncated"] = sum(bool(row["predictions"][arm]["truncated"]) for row in rows)
        primary["arms"][arm]["secondary_anls"] = secondary["arms"][arm]["mean"]
        primary["arms"][arm]["literal_strict_exact"] = literal["arms"][arm]["correct"]
        literal["arms"][arm]["numeric_literal_disagreements"] = sum(
            item["predictions"][arm]["numeric_literal_disagreement"] for item in detail)

    untruncated = [index for index, row in enumerate(rows)
                   if all(not row["predictions"][arm]["truncated"] for arm in ARMS)]
    sensitivity = {}
    for metric, binary in (("exact", True), ("anls", False), ("literal", True)):
        vectors = {arm: [scores[metric][arm][index] for index in untruncated] for arm in ARMS}
        sensitivity[metric] = summarize(vectors, [rows[index] for index in untruncated], binary)

    return dict(
        schema_version=2, n=len(rows), complete=True,
        denominator="All 16 preselected independent source documents; no teacher correctness or truncation filtering",
        arms=primary["arms"], paired=primary["paired"],
        secondary_anls=secondary, literal_strict_exact=literal,
        all_untruncated_sensitivity={"n": len(untruncated),
                                    "included_item_ids": [rows[index]["item_id"] for index in untruncated],
                                    "excluded_item_ids": [rows[index]["item_id"] for index in range(len(rows)) if index not in untruncated],
                                    **sensitivity},
        records=detail, scoring=metadata,
        bootstrap=dict(seed=SEED, draws=DRAWS, unit="source parent document (one question per parent)",
                       method="paired percentile bootstrap; same resampled parent indices across all arms and contrasts"),
        timing_note="No latency estimate is inferred from correctness or operation count; measured runtime artifacts must be reported separately.",
        limitations="Exploratory DocVQA training sources, compact numeric-QA-selected, N=16; positive, zero or negative estimates only motivate further study, and cannot establish benchmark generalization, equivalence or a general rejection of reranking.",
    )


def self_check():
    exact, anls, _ = scorers()
    assert anls("1997", ["1996"]) == .75
    assert not exact("1997", ["1996"])
    assert anls("ab", ["ac"]) == .5  # checked-in official boundary is inclusive
    assert anls("abcd", ["axyz"]) == 0.
    assert anls("incorrect", ["incorrect", "correct"]) == 1.
    assert not exact("12", ["12%"])
    assert exact("0012", ["12"]) and "0012" != "12"
    vectors = {arm: [0, 1] for arm in ARMS}
    vectors["ltc_2k"] = [1, 0]
    summary = summarize(vectors, [{"parent_id": "a"}, {"parent_id": "b"}])
    pair = summary["paired"]["ltc_2k_minus_dense_2k"]
    assert pair["rescue"] == pair["breakage"] == 1 and pair["delta_accuracy"] == 0
    assert "dense_2k_minus_dense_full" in summary["paired"]
    assert summarize(vectors, [{"parent_id": "a"}, {"parent_id": "b"}]) == summary
    return dict(pass_all=True, official_anls_single_digit=.75,
                exact_single_digit=False, official_anls_half_similarity=.5,
                paired_rescue_breakage=[1, 1], bootstrap_seed=SEED, gpu_used=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--manifest", default=str(QWEN / "outputs/catchup_rerank/data/manifest.json"))
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if not args.input or not args.output:
        p.error("--input and --output are required unless --self-check is used")
    raw = Path(args.input).read_bytes()
    result = analyze(json.loads(raw), args.manifest)
    result["input"] = str(Path(args.input).resolve())
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["analysis_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({key: result[key] for key in ("n", "complete", "arms", "paired", "limitations")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
