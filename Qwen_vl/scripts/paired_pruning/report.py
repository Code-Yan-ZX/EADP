#!/usr/bin/env python3
"""Audit full denominators and report paired source-document bootstrap results.

All metrics use the original TRAIN-heldout question roster. Missing outputs
never become a smaller successful test set. Test outcomes never select seeds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.paired_pruning"

from .evaluate import (BASELINES, LEARNED_ARMS, atomic_json, items_for_split,
                      load_manifest, official_anls, read_protocol, strict_match,
                      timing_key, validate_keep_indices, verify_freeze)


def paired_bootstrap(values_a, values_b, parents, draws=10000, seed=20261111, batch=256):
    """Question-weighted delta B-A, resampling parents with shared indices.

    A sampled parent brings all of its questions, including repeated draws of
    that parent. The varying per-parent question counts remain in denominators.
    """
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or len(parents) != len(a):
        raise ValueError("bootstrap requires nonempty equal-length paired data")
    if draws < 1 or batch < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid bootstrap inputs")
    names, inverse = np.unique(np.asarray(parents, dtype=str), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    sums = np.bincount(inverse, weights=b - a)
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws)
    for start in range(0, draws, batch):
        end = min(start + batch, draws)
        index = rng.integers(0, len(names), size=(end - start, len(names)))
        sampled[start:end] = sums[index].sum(1) / counts[index].sum(1)
    return {"delta": float(np.mean(b - a)),
            "ci95": [float(v) for v in np.percentile(sampled, [2.5, 97.5])],
            "n_questions": len(a), "n_parents": len(names), "bootstrap_draws": draws,
            "improved": int(np.sum(b > a)), "worsened": int(np.sum(b < a)),
            "tied": int(np.sum(b == a))}


def load_database(path, freeze_hash):
    path = Path(path).resolve()
    if not path.is_file():
        return [], [], []
    connection = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='freeze_sha256'").fetchone()
        if row is None or row[0] != freeze_hash:
            raise ValueError("database freeze identity mismatch")
        predictions = [json.loads(r[0]) for r in connection.execute("SELECT payload FROM predictions")]
        timings = [json.loads(r[0]) for r in connection.execute("SELECT payload FROM timing")]
        events = [json.loads(r[0]) for r in connection.execute("SELECT payload FROM events ORDER BY id")]
        return predictions, timings, events
    finally:
        connection.close()


def _native_audit(record):
    audit = record.get("audit", {})
    if (audit.get("native") is not True or audit.get("ds_streams") != 3
            or audit.get("positions_preserved") is not True):
        raise ValueError("record lacks passing native DeepStack and original-position audit")
    if "physical_visual_count" in audit and audit["physical_visual_count"] != record["actual_visual_count"]:
        raise ValueError("native physical count disagrees with output metadata")
    if record.get("ds_lengths") != [record["actual_visual_count"]] * 3:
        raise ValueError("DeepStack lengths disagree with selected tokens")
    positions = record.get("positions_sha256", "")
    if not isinstance(positions, str) or len(positions) != 64 or any(c not in "0123456789abcdef" for c in positions):
        raise ValueError("missing original-position trace hash")


def audit_predictions(records, manifest, frozen, protocol):
    items = {str(q["item_id"]): (page, q) for page, q in items_for_split(manifest)}
    specs = {spec["name"]: spec for spec in frozen["arms"]}
    expected = {(item_id, arm) for item_id in items for arm in specs}
    indexed = {}
    n, k = protocol.get("n", 1024), protocol.get("k", 256)
    for record in records:
        key = (str(record["item_id"]), record["arm"])
        if key not in expected or key in indexed:
            raise ValueError(f"unexpected or duplicate evaluation key: {key}")
        page, question = items[key[0]]
        spec = specs[key[1]]
        if record.get("freeze_sha256") != frozen["freeze_sha256"]:
            raise ValueError("prediction freeze identity mismatch")
        if record.get("parent_id") != str(page["parent_id"]) or record.get("page_id") != str(page["page_id"]):
            raise ValueError("prediction parent/page does not match original roster")
        if record.get("numeric") != question["numeric"] or record.get("question") != question["question"]:
            raise ValueError("question/subgroup changed in evaluation output")
        if record.get("base_arm") != spec["arm"] or record.get("seed") != spec["seed"]:
            raise ValueError("arm/seed metadata mismatch")
        count = n if spec["arm"] == "full" else k
        source_count = k if spec["arm"] == "rres512" else n
        validate_keep_indices(record.get("keep_idx"), count, source_count)
        if record.get("actual_visual_count") != count or record.get("source_visual_count") != source_count:
            raise ValueError("visual budget mismatch")
        _native_audit(record)
        if type(record.get("truncated")) is not bool:
            raise ValueError("missing truncation flag")
        cap = protocol.get("max_new_tokens", 128)
        if record.get("max_new_tokens") != cap or not isinstance(record.get("token_ids"), list):
            raise ValueError("missing generation cap/token trace")
        if len(record["token_ids"]) > cap:
            raise ValueError("generation exceeded frozen cap")
        for metric, recomputed in (("anls", official_anls(record["text"], question["answers"])),
                                   ("strict", strict_match(record["text"], question["answers"]))):
            if not np.isclose(record.get(metric, float("nan")), recomputed, atol=1e-12, rtol=0):
                raise ValueError(f"stored {metric} disagrees with independent original-answer scoring")
        indexed[key] = record
    missing = sorted(expected - set(indexed))
    return items, indexed, missing


def metric_summary(records, expected_count):
    complete = len(records) == expected_count and expected_count > 0
    observed_anls = float(np.mean([r["anls"] for r in records])) if records else None
    observed_strict = float(np.mean([r["strict"] for r in records])) if records else None
    return {"status": "complete" if complete else "incomplete",
            "n_expected": expected_count, "n_observed": len(records),
            "anls": observed_anls if complete else None,
            "strict": observed_strict if complete else None,
            "observed_anls": observed_anls, "observed_strict": observed_strict,
            "truncated": sum(r["truncated"] for r in records),
            "full_denominator_anls_bounds": (
                [sum(r["anls"] for r in records) / expected_count,
                 (sum(r["anls"] for r in records) + expected_count - len(records)) / expected_count]
                if expected_count else None)}


def timing_summary(timings, manifest, frozen, protocol):
    specs = {spec["name"]: spec for spec in frozen["arms"]}
    dev_items = {str(q["item_id"]): (page, q) for page, q in items_for_split(manifest, "dev")}
    config = frozen["timing"]
    expected = {timing_key(i, arm, cap, block) for i in dev_items for arm in specs
                for cap in config["caps"] for block in range(config["blocks"])}
    indexed = {}
    by_arm = defaultdict(list)
    n, k = protocol.get("n", 1024), protocol.get("k", 256)
    for record in timings:
        key = timing_key(record["item_id"], record["arm"], record["cap"], record["block"])
        if key not in expected or key in indexed:
            raise ValueError(f"unexpected or duplicate timing key: {key}")
        if record.get("freeze_sha256") != frozen["freeze_sha256"]:
            raise ValueError("timing freeze identity mismatch")
        page, _ = dev_items[str(record["item_id"])]
        if record.get("parent_id") != str(page["parent_id"]) or record.get("page_id") != str(page["page_id"]):
            raise ValueError("timing page/parent mismatch")
        spec = specs[record["arm"]]
        count = n if spec["arm"] == "full" else k
        validate_keep_indices(record.get("keep_idx"), count, k if spec["arm"] == "rres512" else n)
        if record.get("actual_visual_count") != count:
            raise ValueError("timing visual count mismatch")
        _native_audit(record)
        if record.get("generated_tokens") != record["cap"] or len(record.get("token_ids", [])) != record["cap"]:
            raise ValueError("timing must use the fixed generated length, ignoring EOS")
        if not np.isfinite(record["elapsed_ms"]) or record["elapsed_ms"] <= 0:
            raise ValueError("invalid elapsed time")
        indexed[key] = record
        by_arm[(record["arm"], record["cap"])].append(record["elapsed_ms"])
    summaries = {}
    expected_per_arm = len(dev_items) * config["blocks"]
    for arm in specs:
        summaries[arm] = {}
        for cap in config["caps"]:
            values = by_arm[(arm, cap)]
            summaries[arm][str(cap)] = {
                "n_expected": expected_per_arm, "n_observed": len(values),
                "status": "complete" if len(values) == expected_per_arm else "incomplete",
                "median_ms": float(np.median(values)) if len(values) == expected_per_arm else None,
                "p25_p75_ms": [float(v) for v in np.percentile(values, [25, 75])] if values else None}
    paired = {}
    for arm in specs:
        for reference in ("full", "b1", "rres512"):
            if arm == reference or reference not in specs:
                continue
            for cap in config["caps"]:
                pairs = [(indexed.get(timing_key(i, reference, cap, b)),
                          indexed.get(timing_key(i, arm, cap, b)))
                         for i in dev_items for b in range(config["blocks"])]
                if all(a is not None and b is not None for a, b in pairs):
                    paired[f"{arm}_vs_{reference}__cap_{cap}"] = {
                        "median_delta_ms": float(np.median([b["elapsed_ms"] - a["elapsed_ms"] for a, b in pairs])),
                        "median_speedup": float(np.median([a["elapsed_ms"] / b["elapsed_ms"] for a, b in pairs])),
                        "n_pairs": len(pairs)}
    return {"status": "complete" if len(indexed) == len(expected) else "incomplete",
            "n_expected": len(expected), "n_observed": len(indexed),
            "missing_keys": sorted(expected - set(indexed)), "arms": summaries,
            "paired": paired, "scope": config["scope"],
            "limitations": "Small fixed TRAIN-dev panel; not benchmark latency or serving throughput."}


def build_report(records, timings, manifest, frozen, protocol, draws=10000, seed=20261111):
    items, indexed, missing = audit_predictions(records, manifest, frozen, protocol)
    item_ids = list(items)
    arm_names = [spec["name"] for spec in frozen["arms"]]
    arms = {}
    groups = {"all": item_ids,
              "numeric": [i for i in item_ids if items[i][1]["numeric"]],
              "non_numeric": [i for i in item_ids if not items[i][1]["numeric"]]}
    for arm in arm_names:
        arms[arm] = {group: metric_summary([indexed[(i, arm)] for i in ids if (i, arm) in indexed], len(ids))
                     for group, ids in groups.items()}
    comparisons = {}

    def compare(candidate, reference):
        if candidate not in arm_names or reference not in arm_names:
            return
        key = f"{candidate}_vs_{reference}"
        comparisons[key] = {}
        for group, ids in groups.items():
            if not ids or not all((i, candidate) in indexed and (i, reference) in indexed for i in ids):
                comparisons[key][group] = {"status": "incomplete", "n_expected": len(ids)}
                continue
            parents = [items[i][0]["parent_id"] for i in ids]
            comparisons[key][group] = {"status": "complete", **{
                metric: paired_bootstrap([indexed[(i, reference)][metric] for i in ids],
                                         [indexed[(i, candidate)][metric] for i in ids],
                                         parents, draws=draws, seed=seed)
                for metric in ("anls", "strict")}}

    for spec in frozen["arms"]:
        if spec["seed"] is not None:
            for reference in ("b1", "b2", "rres512"):
                compare(spec["name"], reference)
    for training_seed in protocol.get("seeds", [11, 23, 37]):
        for candidate, reference in (("ordered_rank", "bag_rank"), ("ordered_pair", "ordered_rank"),
                                     ("ordered_pair", "bag_rank"), ("token_marginal", "canonical"),
                                     ("multimode", "canonical"), ("multimode", "token_marginal")):
            compare(f"{candidate}__seed_{training_seed}", f"{reference}__seed_{training_seed}")
    seed_summaries = {}
    for base_arm in LEARNED_ARMS:
        names = [s["name"] for s in frozen["arms"] if s["arm"] == base_arm]
        if not names:
            continue
        complete = all(arms[name]["all"]["status"] == "complete" for name in names)
        summary = {"status": "complete" if complete else "incomplete", "seeds": names}
        if complete:
            for metric in ("anls", "strict"):
                values = [arms[name]["all"][metric] for name in names]
                summary[metric] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
            summary["truncated_per_seed"] = {name: arms[name]["all"]["truncated"] for name in names}
        seed_summaries[base_arm] = summary
    timing = timing_summary(timings, manifest, frozen, protocol)
    qa_complete = not missing
    return {"schema": 1, "status": "complete" if qa_complete and timing["status"] == "complete" else "incomplete",
            "qa_status": "complete" if qa_complete else "incomplete", "freeze_sha256": frozen["freeze_sha256"],
            "dataset_scope": "DocVQA TRAIN heldout by source parent; not official test or official VAL results",
            "metric": "official DocVQA ANLS (0..1, >=0.5); literal strict alias equality",
            "bootstrap": {"draws": draws, "seed": seed, "unit": "source parent document",
                          "weighting": "question-weighted; shared parent resamples in each paired contrast"},
            "n_questions": len(items), "n_parents": len({str(p["parent_id"]) for p, q in items.values()}),
            "n_expected_outputs": len(items) * len(arm_names), "n_observed_outputs": len(indexed),
            "missing_outputs": [{"item_id": i, "arm": a} for i, a in missing],
            "subgroup_counts": {group: len(ids) for group, ids in groups.items()},
            "arms": arms, "seed_summary": seed_summaries, "paired_comparisons": comparisons,
            "timing": timing,
            "limitations": ["All questions, including truncations and full-model errors, stay in the denominator.",
                            "No test-based checkpoint, seed or question selection.",
                            "Confidence intervals are exploratory and not adjusted for multiple comparisons.",
                            "Document bootstrap covers sample uncertainty, not the full training-seed distribution.",
                            "A complete evaluation is not evidence that either proposed mechanism succeeds."]}


def research_context(run_dir, frozen):
    """Expose training fallback and mechanism availability without outcome selection."""
    run_dir = Path(run_dir).resolve()
    training = {}
    for experiment in ("query", "sets"):
        path = run_dir / f"checkpoints/{experiment}_summary.json"
        summary = json.loads(path.read_text())
        training[experiment] = {
            "status": summary.get("status"), "arms": summary.get("arms"), "seeds": summary.get("seeds"),
            "protocol_sha256": summary.get("protocol_sha256"),
            "cache_index_sha256": summary.get("cache_index_sha256"),
            "checkpoints": [{key: row.get(key) for key in (
                "arm", "seed", "sha256", "best_step", "val_loss", "updates", "fallback_exposures",
                "question_exposures", "fallback_exposure_fraction")}
                for row in summary.get("checkpoints", [])]}
    diagnostic_path = run_dir / "diagnostics/mechanisms.json"
    mechanisms = {"status": "not_available", "supports_method_claim": False,
                  "scope": "fit/validation mechanism diagnostics; not test-method performance"}
    if diagnostic_path.is_file():
        if str(diagnostic_path) not in frozen["files"]:
            mechanisms["status"] = "not_in_evaluation_freeze"
        else:
            data = json.loads(diagnostic_path.read_text())
            if data.get("protocol_sha256") != frozen["protocol_sha256"]:
                raise ValueError("mechanism protocol differs from evaluation")
            inventories = defaultdict(list)
            for row in data.get("bank_inventory", []):
                inventories[row["role"]].append(row)
            mechanisms.update({"status": "complete" if data.get("complete") else "incomplete",
                               "source_sha256": frozen["files"][str(diagnostic_path)],
                               "summary": data.get("summary", {}),
                               "bank_inventory": {}})
            for role, rows in inventories.items():
                no_success = sum(row["successful_masks"] == 0 for row in rows)
                mechanisms["bank_inventory"][role] = {
                    "questions": len(rows), "no_success_questions": no_success,
                    "no_success_fallback_fraction": no_success / len(rows),
                    "multi_success_questions": sum(row["successful_masks"] >= 2 for row in rows)}
            mechanisms["interpretation"] = (
                "Multiple successful masks alone do not prove incompatible modes. Crossed-set failures and "
                "heldout generation improvements must be examined separately; no-success examples use a "
                "fallback target and cannot support a multiple-sufficient-set claim.")
    return {"training": training, "mechanisms": mechanisms}


def _percent(value):
    return "—" if value is None else f"{100 * value:.3f}"


def markdown_report(report):
    lines = ["# Paired pruning evaluation", "", f"Status: **{report['status']}**; QA: **{report['qa_status']}**.", "",
             report["dataset_scope"] + ".", "",
             f"Original denominator: {report['n_questions']} questions / {report['n_parents']} source parents; "
             f"outputs {report['n_observed_outputs']}/{report['n_expected_outputs']}.", "",
             "Scores below are percentages. Incomplete arms have no primary score; see JSON for missing keys and bounds.", "",
             "| Arm | ANLS | Strict | Truncated | Outputs |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, groups in report["arms"].items():
        row = groups["all"]
        lines.append(f"| {name} | {_percent(row['anls'])} | {_percent(row['strict'])} | {row['truncated']} | "
                     f"{row['n_observed']}/{row['n_expected']} |")
    lines += ["", "## All training seeds", "", "No seed is selected by test performance.", "",
              "| Arm | Mean ANLS | Seed range |", "| --- | ---: | ---: |"]
    for name, row in report["seed_summary"].items():
        if row["status"] == "complete":
            metric = row["anls"]
            lines.append(f"| {name} | {_percent(metric['mean'])} | {_percent(metric['min'])}–{_percent(metric['max'])} |")
        else:
            lines.append(f"| {name} | incomplete | — |")
    lines += ["", "## Paired comparisons", "",
              f"Delta is candidate minus reference in percentage points; {report['bootstrap']['draws']} paired parent bootstrap draws.", "",
              "| Comparison | Δ ANLS | 95% CI | Improved / worsened |", "| --- | ---: | --- | ---: |"]
    for name, groups in report["paired_comparisons"].items():
        result = groups["all"]
        if result["status"] == "complete":
            anls = result["anls"]
            lines.append(f"| {name} | {_percent(anls['delta'])} | [{_percent(anls['ci95'][0])}, "
                         f"{_percent(anls['ci95'][1])}] | {anls['improved']} / {anls['worsened']} |")
        else:
            lines.append(f"| {name} | incomplete | — | — |")
    lines += ["", "## Numeric and non-numeric questions", "",
              "| Arm | Numeric ANLS (N) | Non-numeric ANLS (N) |", "| --- | ---: | ---: |"]
    for name, groups in report["arms"].items():
        numeric, non_numeric = groups["numeric"], groups["non_numeric"]
        lines.append(f"| {name} | {_percent(numeric['anls'])} ({numeric['n_expected']}) | "
                     f"{_percent(non_numeric['anls'])} ({non_numeric['n_expected']}) |")
    timing = report["timing"]
    lines += ["", "## Fresh end-to-end timing on fixed DEV questions", "",
              f"Status: {timing['status']}; {timing['n_observed']}/{timing['n_expected']} measured invocations.", "",
              "Each invocation includes image preparation, vision encoding, selection and generation; model loading is excluded. "
              "TTFT generates one token; the 16-token measurement disables EOS only for timing. QA retains native EOS. "
              "These small dev-page timings are not benchmark-wide latency claims.", "",
              "| Arm | Median TTFT ms | Median 16-token total ms |", "| --- | ---: | ---: |"]
    for name, values in timing["arms"].items():
        one, sixteen = values["1"]["median_ms"], values["16"]["median_ms"]
        lines.append(f"| {name} | {'—' if one is None else f'{one:.3f}'} | {'—' if sixteen is None else f'{sixteen:.3f}'} |")
    if "research_context" in report:
        context = report["research_context"]
        mechanism = context["mechanisms"]
        lines += ["", "## Mechanism and training context", "",
                  f"Mechanism diagnostics: **{mechanism['status']}**. These fit/validation checks do not establish test-method effectiveness.", ""]
        if mechanism.get("summary"):
            summary = mechanism["summary"]
            lines += [f"Query interventions: {summary.get('query_questions', 0)} validation questions; "
                      f"own minus swapped-mask ANLS: {summary.get('query_own_minus_swapped_anls', 'unavailable')} percentage points.", "",
                      f"Distinct successful pairs: {summary.get('distinct_success_pairs', 0)}; "
                      f"crossed-set failures: {summary.get('crossed_failures', 0)}.", ""]
        for role, inventory in mechanism.get("bank_inventory", {}).items():
            lines += [f"{role}: {inventory['multi_success_questions']}/{inventory['questions']} questions have at least two successful masks; "
                      f"{inventory['no_success_questions']}/{inventory['questions']} have no successful mask "
                      f"(fallback fraction {_percent(inventory['no_success_fallback_fraction'])}%).", ""]
        lines += ["No-success fallbacks must not be counted as successful-set supervision. "
                  "Multiple successful sets alone do not show that mixing their members fails.", "",
                  "| Checkpoint | Updates | Validation-selected step | Fallback exposures / question exposures |",
                  "| --- | ---: | ---: | ---: |"]
        for training in context["training"].values():
            for row in training["checkpoints"]:
                lines.append(f"| {row['arm']}__seed_{row['seed']} | {row['updates']} | {row['best_step']} | "
                             f"{row['fallback_exposures']} / {row['question_exposures']} |")
    lines += ["", "## Interpretation limits", ""] + [f"- {note}" for note in report["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261111)
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("bootstrap draws must be positive")
    frozen = verify_freeze(args.run_dir)
    protocol = read_protocol(args.run_dir)
    manifest = load_manifest(args.run_dir)
    records, timings, events = load_database(args.run_dir / "evaluation/predictions.sqlite3", frozen["freeze_sha256"])
    report = build_report(records, timings, manifest, frozen, protocol,
                          draws=args.bootstrap, seed=args.bootstrap_seed)
    report["research_context"] = research_context(args.run_dir, frozen)
    report["event_count"] = len(events)
    report["events"] = events
    atomic_json(args.run_dir / "evaluation/report.json", report)
    (args.run_dir / "evaluation/report.md").write_text(markdown_report(report))
    print(json.dumps({"status": report["status"], "qa_status": report["qa_status"],
                      "outputs": report["n_observed_outputs"], "expected": report["n_expected_outputs"]}))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
