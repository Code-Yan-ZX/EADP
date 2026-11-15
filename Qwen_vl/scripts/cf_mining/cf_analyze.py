"""CPU-only, denominator-explicit analysis of the semantic-difference pilot.

Read actual persisted predictions; never infer a missing experiment result.
JSON and the compact figure may describe an explicitly incomplete snapshot.
The prose report requires completed mining and training artifacts and is only
written when --write-report is supplied.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "Qwen_vl" / "outputs" / "cf_mining"


def read_json(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def normalize(value):
    value = str(value).strip().lower().strip("'\"`")
    value = re.sub(r"[\s,$£€]", "", value).replace("−", "-")
    suffix = "%" if value.endswith("%") else ""
    number = value[:-1] if suffix else value
    try:
        return str(Decimal(number).normalize()) + suffix
    except InvalidOperation:
        return value


@lru_cache(maxsize=1)
def official_anls_functions():
    """Load only the official pure functions, without vlmeval/model imports."""
    import numpy as np
    utility = ROOT / "Qwen_vl/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py"
    misc = ROOT / "Qwen_vl/VLMEvalKit/vlmeval/smp/misc.py"
    selected = []
    for source, names in ((misc, {"listinstr"}),
                          (utility, {"levenshtein_distance", "anls_compute", "hit_calculate"})):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in nodes} != names:
            raise RuntimeError(f"official ANLS helper definitions missing: {source}")
        selected.extend(nodes)
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(utility), "exec"), namespace)
    distance, hit = namespace["anls_compute"], namespace["hit_calculate"]
    # Regression for the historical distance-versus-score inversion.
    assert distance("1997", "1996") == .25
    assert float(hit([{"match": [.25]}], "DocVQA_VAL")[0]) == .75
    assert normalize("1997") != normalize("1996")
    return distance, hit, dict(source=str(utility), source_sha256=hashlib.sha256(utility.read_bytes()).hexdigest(),
                              loading="AST-extracted official pure functions; no package/model imports", threshold=.5,
                              threshold_semantics="official hit_calculate zeroes similarity below 0.5 and retains equality",
                              sanity=dict(ground_truth="1997", prediction="1996", normalized_distance=.25,
                                          secondary_anls=.75, primary_exact=False))


def official_anls(answers, prediction):
    distance, hit, _ = official_anls_functions()
    distances = [distance(str(answer), str(prediction)) for answer in answers]
    if not distances:
        return None
    return float(hit([{"match": distances}], "DocVQA_VAL")[0])


def answer_aliases(row, lookup, side=None):
    original = lookup.get(str(row.get("pair_id")), {})
    answers = row.get("answers")
    options = [answers.get(side) if isinstance(answers, dict) else None]
    if side:
        options.extend([row.get(side + "_answers"), original.get(side + "_answers")])
    if side in (None, "original"):
        options.extend([answers if isinstance(answers, list) else None, original.get("answers")])
    options.extend([row.get((side + "_answer") if side else "answer"),
                    original.get((side + "_answer") if side else "original_answer")])
    for aliases in options:
        if aliases is None:
            continue
        values = [str(a) for a in aliases] if isinstance(aliases, (list, tuple)) else [str(aliases)]
        if values:
            return values
    return []


def anls_rollup(cases, lookup):
    scores, pairs, groups = [], set(), set()
    for row, side, prediction in cases:
        if prediction is None:
            continue
        aliases = answer_aliases(row, lookup, side)
        score = official_anls(aliases, prediction)
        if score is None:
            continue
        scores.append(score)
        pairs.add(str(row.get("pair_id")))
        groups.add(group_id(row, lookup))
    return dict(n_individual_answers=len(scores), n_questions_or_pairs=len(pairs), n_documents=len(groups),
                mean_anls=sum(scores) / len(scores) if scores else None,
                sum_anls=sum(scores))


def pair_anls(rows, lookup, fields):
    return {arm: anls_rollup([(row, side, getter(row, side)) for row in rows for side in ("original", "edited")], lookup)
            for arm, getter in fields.items()}


def source_anls(rows, lookup, fields):
    return {arm: anls_rollup([(row, None, getter(row)) for row in rows], lookup) for arm, getter in fields.items()}


def secondary_anls(mining, comparison, attempts, references_directory, lookup):
    _, _, implementation = official_anls_functions()
    rows = mining.get("records", [])
    miner_fields = {arm: (lambda row, side, arm=arm: row.get(side + "_" + arm + "_prediction")) for arm in ("teacher", "student")}
    evaluator_fields = {"student": lambda row, side: row.get(side + "_prediction")}
    result = dict(implementation=implementation, primary_unchanged="exact numeric pair correctness",
                  interpretation="secondary mean over individual answers, not both-correct pair accuracy or a full benchmark score",
                  mining={}, untouched_sources={}, carrier_fit_attempts=[])
    for split in [*sorted({split_name(row) for row in rows}), "all"]:
        selected = rows if split == "all" else [row for row in rows if split_name(row) == split]
        result["mining"][split] = dict(all_attempted=pair_anls(selected, lookup, miner_fields),
                                       teacher_valid=pair_anls([row for row in selected if row.get("teacher_valid")], lookup, miner_fields))
    sources = mining.get("sources", [])
    for split in [*sorted({split_name(row) for row in sources}), "all"]:
        selected = sources if split == "all" else [row for row in sources if split_name(row) == split]
        result["untouched_sources"][split] = source_anls(selected, lookup, {"teacher": lambda row: row.get("teacher"), "b2": lambda row: row.get("student")})
    for attempt in attempts:
        directory = Path(attempt["path"]).parent
        before, after = read_json(directory / "gate_fit_before.json") or {}, read_json(directory / "gate_fit_after.json") or {}
        scores = dict(carrier=attempt["carrier"], active=attempt["active"],
                      before=pair_anls(before.get("records", []), lookup, evaluator_fields),
                      after=pair_anls(after.get("records", []), lookup, evaluator_fields))
        for phase in ("before", "after"):
            heldout = read_json(directory / f"gate_heldout_{phase}.json") or {}
            source = read_json(directory / f"gate_sources_{phase}.json") or {}
            scores["heldout_" + phase] = pair_anls(heldout.get("records", []), lookup, evaluator_fields)
            scores["sources_" + phase] = source_anls(source.get("records", []), lookup, {"student": lambda row: row.get("prediction")})
        result["carrier_fit_attempts"].append(scores)
    if comparison:
        result["comparison"] = dict(
            baseline_pairs=pair_anls(comparison.get("baseline", {}).get("records", []), lookup, evaluator_fields),
            baseline_sources=source_anls(comparison.get("source_baseline", {}).get("records", []), lookup, {"student": lambda row: row.get("prediction")}),
            arms={name: dict(heldout_members=pair_anls(arm.get("heldout", {}).get("records", []), lookup, evaluator_fields),
                             untouched_sources=source_anls(arm.get("sources", {}).get("records", []), lookup, {"student": lambda row: row.get("prediction")}))
                  for name, arm in comparison.get("arms", {}).items()})
    result["references"] = {}
    directory = Path(references_directory)
    for filename, kind in (("rres512_sources.json", "sources"), ("rres512_pairs.json", "pairs")):
        payload = read_json(directory / filename)
        if not payload:
            result["references"][kind] = dict(status="not_run")
            continue
        records = payload.get("records", [])
        if kind == "sources":
            scores = source_anls(records, lookup, {arm: (lambda row, arm=arm: row.get(arm + "_prediction")) for arm in ("teacher", "b2", "rres512")})
        else:
            scores = pair_anls(records, lookup, {arm: (lambda row, side, arm=arm: row.get("predictions", {}).get(side, {}).get(arm)) for arm in ("teacher", "b2", "rres512")})
        result["references"][kind] = dict(status="complete" if payload.get("complete") else "running", scores=scores)
    invariant = read_json(directory / "invariant_questions.json")
    if invariant:
        # Sibling labels are independent of the edited target question's labels.
        records = [dict(row, answers={"original": row.get("answers", []), "edited": row.get("answers", [])})
                   for row in invariant.get("records", [])]
        fields = {arm: (lambda row, side, arm=arm: row.get("predictions", {}).get(side, {}).get(arm)) for arm in ("teacher", "b2", "online")}
        result["references"]["invariants"] = dict(status="skipped" if invariant.get("skipped_reason") else "complete" if invariant.get("complete") else "running",
                                                  scores=pair_anls([row for row in records if row.get("teacher_both_correct")], lookup, fields),
                                                  all_attempted=pair_anls(records, lookup, fields))
    return result


def split_name(row):
    value = row.get("split", "unknown")
    return {"train": "fit", "val": "validation"}.get(value, value)


def group_id(row, lookup=None):
    for name in ("group_id", "doc_id", "image_id", "source_image"):
        if row.get(name) is not None:
            return str(row[name])
    key = str(row.get("pair_id", "unknown"))
    if lookup and key in lookup:
        return group_id(lookup[key])
    return key


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def mining_counts(rows):
    valid = [r for r in rows if r.get("teacher_valid")]
    broad = sum(not bool(r.get("student_both_correct")) for r in valid)
    strict = sum(bool(r.get("collapse")) and not bool(r.get("student_both_correct")) for r in valid)
    both = sum(bool(r.get("student_both_correct")) for r in valid)
    return dict(attempted=len(rows), n_documents=len({group_id(r) for r in rows}),
                teacher_valid=len(valid), teacher_invalid=len(rows) - len(valid),
                teacher_valid_fraction_of_attempted=rate(len(valid), len(rows)),
                broad_pair_failures=broad, strict_answer_collapses=strict,
                broad_only_failures=broad - strict, student_both_correct=both,
                broad_pair_failure_rate_of_teacher_valid=rate(broad, len(valid)),
                strict_answer_collapse_rate_of_teacher_valid=rate(strict, len(valid)),
                student_both_correct_rate_of_teacher_valid=rate(both, len(valid)))


def source_counts(rows, lookup):
    teacher = sum(bool(r.get("teacher_correct")) for r in rows)
    student = sum(bool(r.get("student_correct", r.get("correct"))) for r in rows)
    teacher_known = sum(r.get("teacher_correct") is not None for r in rows)
    return dict(n_questions=len(rows), n_documents=len({group_id(r, lookup) for r in rows}),
                teacher_correct=teacher, teacher_labelled_questions=teacher_known,
                teacher_accuracy=rate(teacher, teacher_known), student_correct=student,
                student_accuracy=rate(student, len(rows)))


def paired_interval(before, after, field, lookup, seed, n_boot=2000):
    """Pair identical questions and resample whole source groups."""
    previous = {str(r["pair_id"]): r for r in before}
    groups = defaultdict(list)
    for row in after:
        key = str(row["pair_id"])
        if key in previous and field in row and field in previous[key]:
            groups[group_id(row, lookup)].append(int(bool(row[field])) - int(bool(previous[key][field])))
    values = [v for g in groups.values() for v in g]
    result = dict(n_pairs=len(values), n_documents=len(groups),
                  delta=rate(sum(values), len(values)), ci95=None,
                  estimand="question-weighted paired difference; bootstrap whole source documents")
    if len(groups) < 2:
        result["note"] = "fewer than two source documents; no confidence interval"
        return result
    keys = sorted(groups)
    rng, samples = random.Random(seed), []
    for _ in range(n_boot):
        sample = [v for key in rng.choices(keys, k=len(keys)) for v in groups[key]]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    result["ci95"] = [samples[int(.025 * (n_boot - 1))], samples[int(.975 * (n_boot - 1))]]
    result["bootstrap_draws"] = n_boot
    return result


def pair_metrics(payload):
    if not payload:
        return None
    if "summary" in payload:
        return payload["summary"]
    return payload if "n_pairs" in payload else None


def source_metrics(payload, lookup):
    if not payload:
        return None
    rows = payload.get("records", [])
    if not rows and payload.get("n_sources", 0):
        return dict(n_questions=payload["n_sources"], accuracy=payload.get("accuracy"))
    result = source_counts(rows, lookup)
    result["accuracy"] = payload.get("accuracy", result["student_accuracy"])
    return result


def reference_results(directory, lookup, comparison, seed, bootstrap):
    """Accept absent supplementary checks; preserve their actual denominators."""
    directory = Path(directory)
    reference = read_json(directory / "reference_summary.json")
    rres = read_json(directory / "rres512_sources.json")
    rres_pairs = read_json(directory / "rres512_pairs.json")
    invariant = read_json(directory / "invariant_questions.json")
    all_complete = bool(reference and reference.get("complete") and rres and rres.get("complete")
                        and invariant and invariant.get("complete")
                        and (not reference.get("rres512_pairs") or rres_pairs and rres_pairs.get("complete")))
    if rres_pairs and not rres_pairs.get("complete"):
        all_complete = False
    result = dict(status="complete" if all_complete else
                  "running" if reference or rres or rres_pairs or invariant else "not_run",
                  files={name: str((directory / name).resolve()) for name in
                         ("reference_summary.json", "rres512_sources.json", "rres512_pairs.json", "invariant_questions.json")},
                  wall_seconds=reference.get("wall_seconds") if reference else None,
                  limitation=reference.get("limitation") if reference else None,
                  rres512=dict(status="not_run"), rres512_pairs=dict(status="not_run"), invariants=dict(status="not_run"))
    if rres:
        rows = rres.get("records", [])
        accepted = [r for r in rows if r.get("teacher_correct")]
        rates = lambda rr, field: dict(n=sum(r.get(field) is not None for r in rr),
                                      correct=sum(bool(r.get(field)) for r in rr),
                                      accuracy=rate(sum(bool(r.get(field)) for r in rr), sum(r.get(field) is not None for r in rr)))
        primary = {arm: rates(rows, arm + "_correct") for arm in ("teacher", "b2", "rres512")}
        rrows = [dict(r, correct=r["rres512_correct"]) for r in rows]
        brows = [dict(r, correct=r["b2_correct"]) for r in rows]
        rr = dict(status="complete" if rres.get("complete") else "running",
                  reference=rres.get("reference"), n_sources=len(rows),
                  n_documents=len({group_id(r, lookup) for r in rows}), all_sources=primary,
                  teacher_correct_sources={arm: rates(accepted, arm + "_correct") for arm in ("b2", "rres512")},
                  input_visual_tokens=rres.get("input_visual_tokens"), decoder_visual_tokens=rres.get("decoder_visual_tokens"),
                  observed_input_visual_tokens=sorted({r["actual_input_visual_tokens"] for r in rows if "actual_input_visual_tokens" in r}),
                  observed_decoder_visual_tokens=sorted({r["actual_decoder_visual_tokens"] for r in rows if "actual_decoder_visual_tokens" in r}),
                  rres512_minus_b2_rescued=sum(bool(r["rres512_correct"]) and not bool(r["b2_correct"]) for r in rows),
                  rres512_minus_b2_broken=sum(bool(r["b2_correct"]) and not bool(r["rres512_correct"]) for r in rows),
                  rres512_minus_b2=paired_interval(brows, rrows, "correct", lookup, seed, bootstrap),
                  resolution_restored=rres.get("resolution_restored"), latency_claim=rres.get("latency_claim", False),
                  wall_seconds=rres.get("wall_seconds"))
        if comparison:
            rr["arm_minus_rres512"] = {name: paired_interval(rrows, arm.get("sources", {}).get("records", []), "correct", lookup, seed, bootstrap)
                                      for name, arm in comparison.get("arms", {}).items()}
        result["rres512"] = rr
    if rres_pairs:
        rows = rres_pairs.get("records", [])
        make_rows = lambda arm: [dict(r, both_correct=bool(r[arm + "_both_correct"])) for r in rows]
        b2_rows, resolution_rows = make_rows("b2"), make_rows("rres512")
        pair_reference = dict(
            status="complete" if rres_pairs.get("complete") else "running",
            reference=rres_pairs.get("reference"), selection=rres_pairs.get("selection"),
            n_pairs=len(rows), n_documents=len({group_id(r, lookup) for r in rows}),
            both_correct={arm: dict(n=len(rows), correct=sum(bool(r.get(arm + "_both_correct")) for r in rows),
                                    accuracy=rate(sum(bool(r.get(arm + "_both_correct")) for r in rows), len(rows)))
                          for arm in ("teacher", "b2", "rres512")},
            strict_collapse={arm: dict(n=len(rows), count=sum(bool(r.get(arm + "_strict_collapse")) for r in rows),
                                       rate=rate(sum(bool(r.get(arm + "_strict_collapse")) for r in rows), len(rows)))
                             for arm in ("b2", "rres512")},
            rres512_minus_b2_rescued=sum(bool(r["rres512_both_correct"]) and not bool(r["b2_both_correct"]) for r in rows),
            rres512_minus_b2_broken=sum(bool(r["b2_both_correct"]) and not bool(r["rres512_both_correct"]) for r in rows),
            b2_strict_collapses_repaired_to_both_correct=sum(bool(r["b2_strict_collapse"]) and bool(r["rres512_both_correct"]) for r in rows),
            rres512_minus_b2=paired_interval(b2_rows, resolution_rows, "both_correct", lookup, seed, bootstrap),
            input_visual_tokens=rres_pairs.get("input_visual_tokens"), decoder_visual_tokens=rres_pairs.get("decoder_visual_tokens"),
            observed_input_visual_tokens=sorted({n for r in rows for n in r.get("input_tokens", {}).values()}),
            observed_decoder_visual_tokens=sorted({n for r in rows for n in r.get("decoder_tokens", {}).values()}),
            resolution_restored=rres_pairs.get("resolution_restored"), latency_claim=rres_pairs.get("latency_claim", False))
        if comparison:
            pair_reference["arm_minus_rres512"] = {name: paired_interval(resolution_rows, arm.get("heldout", {}).get("records", []), "both_correct", lookup, seed, bootstrap)
                                                   for name, arm in comparison.get("arms", {}).items()}
        result["rres512_pairs"] = pair_reference
    if invariant:
        rows = invariant.get("records", [])
        accepted = [r for r in rows if r.get("teacher_both_correct")]
        after = [dict(r, both_correct=r["online_both_correct"]) for r in accepted]
        before = [dict(r, both_correct=r["b2_both_correct"]) for r in accepted]
        result["invariants"] = dict(status="skipped" if invariant.get("skipped_reason") else
                                    "complete" if invariant.get("complete") else "running",
                                    skipped_reason=invariant.get("skipped_reason"),
                                    candidate_pairs=invariant.get("candidate_pairs"), attempted_pairs=len(rows),
                                    teacher_valid_pairs=len(accepted), teacher_rejected_pairs=len(rows) - len(accepted),
                                    n_accepted_documents=len({group_id(r, lookup) for r in accepted}),
                                    accepted={arm: dict(n=len(accepted), correct=sum(bool(r.get(arm + "_both_correct")) for r in accepted),
                                                        accuracy=rate(sum(bool(r.get(arm + "_both_correct")) for r in accepted), len(accepted)))
                                              for arm in ("teacher", "b2", "online")},
                                    online_rescued=sum(bool(r["online_both_correct"]) and not bool(r["b2_both_correct"]) for r in accepted),
                                    online_broken=sum(bool(r["b2_both_correct"]) and not bool(r["online_both_correct"]) for r in accepted),
                                    online_minus_b2=paired_interval(before, after, "both_correct", lookup, seed, bootstrap),
                                    acceptance=invariant.get("acceptance"), scope=invariant.get("scope"),
                                    pool_restored=invariant.get("pool_restored"), wall_seconds=invariant.get("wall_seconds"))
    return result


def training_log(directory):
    """Summarize the observed log, without equating requested steps with done."""
    directory = Path(directory)
    history = directory / "history.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()] if history.exists() else []
    mining = directory / "mining.jsonl"
    searches = [json.loads(line) for line in mining.read_text().splitlines() if line.strip()] if mining.exists() else []
    recorded = read_json(directory / "training.json") or {}
    schedule = recorded.get("schedule") or read_json(directory / "schedule.json")
    exposure_counts = Counter((str(x["pair_id"]), x["side"]) for update in schedule for x in update) if isinstance(schedule, list) else Counter()
    return dict(logged_optimizer_updates=len(rows),
                final_history=rows[-1] if rows else None,
                nonzero_gradient_updates=sum(bool(r.get("gradient_norm", 0) > 0) for r in rows),
                mining_candidates_scored=len(searches),
                mining_rounds=len({r.get("step") for r in searches}),
                exposure_updates=len(schedule) if isinstance(schedule, list) else None,
                answer_exposures=sum(exposure_counts.values()) if schedule is not None else None,
                exposure_multiset={f"{pair_id}|{side}": count for (pair_id, side), count in sorted(exposure_counts.items())},
                wall_seconds=recorded.get("wall_seconds"), objectives=recorded.get("objectives"),
                forward_counts=recorded.get("forward_counts"),
                contrast_negative_forwards=(recorded.get("forward_counts") or {}).get("training_negative_answer", 0),
                mean_loss=recorded.get("mean_loss"))


def carrier_attempt(path, active=False):
    path = Path(path)
    gate = read_json(path)
    plan = read_json(path.parent / "gate_plan.json") or {}
    if not gate and not plan:
        return None
    config = plan.get("config", {})
    before = read_json(path.parent / "gate_fit_before.json") or {}
    after = read_json(path.parent / "gate_fit_after.json") or {}
    training = training_log(path.parent / "gate")
    # The archived first protocol predates an explicit carrier config key.
    carrier = (gate or {}).get("carrier") or config.get("carrier")
    if carrier is None and path.parent.name == "training_global_gate":
        carrier = "pool"
    return dict(active=active, path=str(path.resolve()), carrier=carrier or "not_recorded",
                trainable_parameters=(gate or {}).get("trainable_parameters", (gate or {}).get("parameter_count")),
                rank=config.get("rank"), requested_updates=config.get("steps"),
                fit_pool_size=plan.get("counts", {}).get("fit"),
                refresh_interval=config.get("round_steps"), config=config,
                gate_pair_ids=plan.get("pair_ids", {}).get("gate", []),
                verdict=(gate or {}).get("verdict", "running"), complete=bool(gate and gate.get("verdict")),
                before=(gate or {}).get("before", before.get("summary")),
                after=(gate or {}).get("after", after.get("summary")),
                heldout=(gate or {}).get("heldout"), heldout_change=(gate or {}).get("heldout_change"),
                source_before=(gate or {}).get("source_baseline", {}).get("accuracy"),
                source_after=(gate or {}).get("sources", {}).get("accuracy"),
                native_identity=(gate or {}).get("native_identity"),
                frozen_decoder_unchanged=(gate or {}).get("frozen_decoder_unchanged"),
                decoder_sample_checksum_unchanged=(gate or {}).get("decoder_sample_checksum_unchanged"),
                training=training)


def online_search_diagnostic(directory, lookup):
    """Distinguish baseline blind spots from failures introduced during fit."""
    directory = Path(directory)
    files = [("initial_shared_by_fixed_online", directory / "comparison_initial_mining.jsonl"),
             ("online_refresh", directory / "online/mining.jsonl")]
    rounds = []
    for phase, path in files:
        if not path.exists():
            continue
        by_step = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_step[row.get("step")].append(row)
        for step, rows in sorted(by_step.items(), key=lambda item: -1 if item[0] is None else item[0]):
            known, unknown, unlabelled = [], 0, 0
            for row in rows:
                previous = lookup.get(str(row.get("pair_id")))
                if previous is None or not previous.get("teacher_valid"):
                    unknown += 1
                    continue
                if "original_correct" not in row or "edited_correct" not in row:
                    unlabelled += 1
                    continue
                both = bool(row["original_correct"] and row["edited_correct"])
                strict = bool(row.get("strict_collapse", False)) and not both
                known.append((bool(previous.get("student_both_correct")), both, strict))
            baseline_correct = sum(old for old, _, _ in known)
            baseline_fail = len(known) - baseline_correct
            current_both = sum(now for _, now, _ in known)
            broken = sum(old and not now for old, now, _ in known)
            rescued = sum(not old and now for old, now, _ in known)
            failed = len(known) - current_both
            rounds.append(dict(phase=phase, step=step, path=str(path.resolve()),
                               scored_candidates=len(rows), unique_candidates=len({r.get("pair_id") for r in rows}),
                               labelled_teacher_valid_candidates=len(known), unknown_or_teacher_invalid=unknown,
                               proxy_only_unlabelled_candidates=unlabelled,
                               baseline_both_correct=baseline_correct, baseline_failures=baseline_fail,
                               current_both_correct=current_both, current_failures=failed,
                               current_strict_collapses=sum(strict for _, _, strict in known),
                               baseline_correct_to_failure=broken, baseline_failure_to_both_correct=rescued,
                               baseline_correct_to_strict_collapse=sum(old and strict for old, _, strict in known),
                               broken_fraction_of_baseline_correct=rate(broken, baseline_correct),
                               rescued_fraction_of_baseline_failures=rate(rescued, baseline_fail),
                               broken_fraction_of_current_failures=rate(broken, failed)))
    return dict(status="observed" if rounds else "not_run", rounds=rounds,
                initial_shared_rounds=sum(r["phase"] == "initial_shared_by_fixed_online" for r in rounds),
                online_refresh_rounds=sum(r["phase"] == "online_refresh" for r in rounds),
                total_observed_rounds=len(rounds),
                interpretation="Same teacher-valid fit pairs: baseline-correct to current failure measures training breakage; baseline-failure to two correct answers measures rescue. Proxy-only rows remain unlabelled.")


def training_generation_sanity(directory):
    path = Path(directory) / "train_generation_sanity.json"
    payload = read_json(path)
    if payload is None:
        return dict(status="not_run", path=str(path.resolve()))
    result = dict(status="complete" if payload.get("complete") else "running", path=str(path.resolve()),
                  purpose=payload.get("purpose"), pool_restored=payload.get("pool_restored"), arms={})
    for name, arm in payload.get("arms", {}).items():
        records = arm.get("records", [])
        nll = [row["teacher_forced_answer_nll"] for row in records if row.get("teacher_forced_answer_nll") is not None]
        result["arms"][name] = dict(n=arm.get("n", len(records)), correct=arm.get("correct"),
                                   accuracy=rate(arm.get("correct", 0), arm.get("n", len(records))),
                                   teacher_forced_nll_min=min(nll) if nll else None,
                                   teacher_forced_nll_max=max(nll) if nll else None,
                                   teacher_forced_nll_mean=sum(nll) / len(nll) if nll else None,
                                   records=records)
    return result


def summarize(args):
    mining = read_json(args.mining, required=True)
    manifest = read_json(args.manifest)
    gate = read_json(args.gate)
    comparison = read_json(args.comparison)
    rows = mining.get("records", [])
    lookup = {str(r["pair_id"]): r for r in rows}
    splits = sorted({split_name(r) for r in rows}, key=lambda s: ({"fit": 0, "validation": 1, "test": 2}.get(s, 3), s))
    counts = {s: mining_counts([r for r in rows if split_name(r) == s]) for s in splits}
    counts["all"] = mining_counts(rows)
    flag_discrepancies = []
    for r in rows:
        if not all(side + "_teacher_prediction" in r and side + "_student_prediction" in r for side in ("original", "edited")):
            continue
        tv = all(normalize(r[side + "_teacher_prediction"]) == normalize(r[side + "_answer"]) for side in ("original", "edited"))
        both = all(normalize(r[side + "_student_prediction"]) == normalize(r[side + "_answer"]) for side in ("original", "edited"))
        strict = tv and not both and normalize(r["original_student_prediction"]) == normalize(r["edited_student_prediction"])
        for field, value in (("teacher_valid", tv), ("student_both_correct", both), ("blind", tv and not both), ("collapse", strict)):
            if field in r and bool(r[field]) != value:
                flag_discrepancies.append(dict(pair_id=r["pair_id"], field=field, stored=r[field], recomputed=value))
    examples, seen = [], set()
    candidates = [r for r in rows if r.get("teacher_valid") and r.get("collapse") and not r.get("student_both_correct")]
    candidates.sort(key=lambda r: ({"test": 0, "validation": 1, "fit": 2}.get(split_name(r), 3), str(r["pair_id"])))
    for r in candidates:
        if group_id(r) in seen:
            continue
        seen.add(group_id(r))
        examples.append({k: r.get(k) for k in ("pair_id", "split", "doc_id", "image_id", "question",
                         "original_answer", "edited_answer", "original_teacher_prediction", "edited_teacher_prediction",
                         "original_student_prediction", "edited_student_prediction", "original_image", "edited_image")})
        if len(examples) == 3:
            break
    sources = mining.get("sources", [])
    source_summary = {s: source_counts([r for r in sources if split_name(r) == s], lookup)
                      for s in sorted({split_name(r) for r in sources})}
    source_summary["all"] = source_counts(sources, lookup)
    invariants = mining.get("invariants", [])
    invariant_valid = [r for r in invariants if r.get("teacher_both_correct")]
    seed = mining.get("seed", 20261108)
    references_directory = Path(args.references_dir) if args.references_dir else Path(args.comparison).parent
    previous_paths = [Path(p) for p in args.previous_gate] if args.previous_gate else [OUT / name / "gate_result.json" for name in ("training_global_gate", "training_global_readout")]
    previous_paths = [p for p in previous_paths if p.resolve() != Path(args.gate).resolve()]
    attempts = [carrier_attempt(p) for p in previous_paths]
    attempts = [a for a in attempts if a is not None]
    active_attempt = carrier_attempt(args.gate, active=True)
    if active_attempt:
        attempts.append(active_attempt)
    if active_attempt and len(attempts) > 1:
        active_attempt["same_gate_pairs_as_previous"] = bool(active_attempt["gate_pair_ids"]) and active_attempt["gate_pair_ids"] == attempts[-2]["gate_pair_ids"]
    infrastructure_paths = [Path(args.infrastructure)] if args.infrastructure else [Path(args.gate).parent / "infrastructure_events.json", *[p.parent / "infrastructure_events.json" for p in previous_paths]]
    infrastructure_events, infrastructure_seen = [], set()
    for path in infrastructure_paths:
        payload = read_json(path) or {}
        for event in payload.get("events", []):
            signature = json.dumps(event, sort_keys=True)
            if signature not in infrastructure_seen:
                infrastructure_events.append(dict(event, artifact_path=str(path.resolve())))
                infrastructure_seen.add(signature)
    result = dict(schema_version=1, generated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  inputs={k: str(Path(getattr(args, k)).resolve()) for k in ("mining", "manifest", "gate", "comparison")},
                  status=dict(mining_complete=mining.get("stage") == "complete", gate_present=gate is not None,
                              comparison_present=comparison is not None,
                              comparison_complete=bool(comparison and comparison.get("complete"))),
                  setup={k: mining.get(k) for k in ("model", "selector", "token_budget", "resolution", "native_deepstack", "native_mrope", "seed")},
                  generated_candidates=manifest.get("counts") if manifest else None,
                  mining=counts, strict_collapse_examples=examples,
                  prediction_flag_discrepancies=flag_discrepancies,
                  references=reference_results(references_directory, lookup, comparison, seed, args.bootstrap),
                  infrastructure_events=infrastructure_events, carrier_attempts=attempts,
                  active_training_config=active_attempt.get("config") if active_attempt else None,
                  train_generation_sanity=training_generation_sanity(Path(args.comparison).parent),
                  untouched_sources=source_summary,
                  invariants=dict(attempted=len(invariants), teacher_valid=len(invariant_valid),
                                  student_both_correct=sum(bool(r.get("student_both_correct")) for r in invariant_valid),
                                  student_both_correct_rate=rate(sum(bool(r.get("student_both_correct")) for r in invariant_valid), len(invariant_valid))),
                  limitations=["DocVQA VAL custom pilot partitions; not an official benchmark train/test result.",
                               "Image/document-disjoint partitions do not establish unseen-layout transfer.",
                               f"{len(attempts)} carrier attempts were screened; the six-arm comparison uses the final carrier and one seed. This does not cover the full method-design space.",
                               f"Only {active_attempt.get('fit_pool_size') if active_attempt else 'an unrecorded number of'} teacher-valid fit pairs trained the comparison carrier; this tiny adaptation pool is insufficient to establish a strong pretrained compressor.",
                               "Unrelated-question invariance constraints were not trained; sibling-question pairs were evaluated only, with the actual count reported separately.",
                               "No field swaps, arithmetic/sum interventions, or unseen-layout evaluation were implemented. The full proposed method was not tested.",
                               "Data-selection contrasts require the same objective: inspect each arm's positive/negative forwards and loss settings.",
                               "Locality and the answer-relation objective changed together in the final carrier screen; any recovery cannot be attributed to either alone."])
    if gate:
        result["gate"] = {k: gate.get(k) for k in ("verdict", "criterion", "requested_pairs", "actual_pairs", "fit_selection_strict_collapses",
                          "fit_selection_broad_only_failures", "train_both_correct_gain", "before", "after", "heldout", "heldout_change",
                          "fit_collapse_changes", "heldout_collapse_changes", "native_identity", "initial_sha256", "frozen_decoder_unchanged",
                          "decoder_sample_checksum_unchanged", "limitation", "reason", "carrier", "trainable_parameters")}
        result["gate"]["sources"] = source_metrics(gate.get("sources"), lookup)
        result["gate"]["source_baseline"] = source_metrics(gate.get("source_baseline"), lookup)
        result["gate"]["training"] = training_log(Path(args.gate).parent / "gate")
        if gate.get("sources") and gate.get("source_baseline"):
            result["gate"]["source_change"] = paired_interval(gate["source_baseline"].get("records", []), gate["sources"].get("records", []), "correct", lookup, seed, args.bootstrap)
    if comparison:
        baseline = comparison.get("baseline", {})
        source_baseline = comparison.get("source_baseline", {})
        comp = dict(complete=bool(comparison.get("complete")), config=comparison.get("config"),
                    carrier=comparison.get("carrier"), trainable_parameters=comparison.get("trainable_parameters"),
                    objectives=comparison.get("objectives"), baseline=pair_metrics(baseline),
                    source_baseline=source_metrics(source_baseline, lookup), arms={},
                    paired_contrasts=comparison.get("paired_contrasts", {}), source_contrasts={},
                    forward_counts=comparison.get("forward_counts"), native_identity=comparison.get("native_identity"),
                    frozen_decoder_unchanged=comparison.get("frozen_decoder_unchanged"),
                    decoder_sample_checksum_unchanged=comparison.get("decoder_sample_checksum_unchanged"),
                    interpretation=comparison.get("interpretation"))
        comp["initial_search"] = read_json(Path(args.comparison).parent / "comparison_initial_hard.json")
        comp["search_diagnostic"] = online_search_diagnostic(Path(args.comparison).parent, lookup)
        for name, arm in comparison.get("arms", {}).items():
            comp["arms"][name] = dict(heldout=pair_metrics(arm.get("heldout")), sources=source_metrics(arm.get("sources"), lookup),
                                      heldout_change=arm.get("heldout_change"), heldout_collapse_changes=arm.get("heldout_collapse_changes"),
                                      initial_sha256=arm.get("initial_sha256"), training=training_log(Path(args.comparison).parent / name))
            comp["arms"][name]["source_change"] = paired_interval(source_baseline.get("records", []), arm.get("sources", {}).get("records", []), "correct", lookup, seed, args.bootstrap)
        if "online" in comparison.get("arms", {}):
            online = comparison["arms"]["online"]
            for name, arm in comparison["arms"].items():
                if name == "online":
                    continue
                key = "online_minus_" + name
                comp["paired_contrasts"].setdefault(key, paired_interval(arm.get("heldout", {}).get("records", []), online.get("heldout", {}).get("records", []), "both_correct", lookup, seed, args.bootstrap))
                comp["source_contrasts"][key] = paired_interval(arm.get("sources", {}).get("records", []), online.get("sources", {}).get("records", []), "correct", lookup, seed, args.bootstrap)
        hashes = {arm["initial_sha256"] for arm in comp["arms"].values() if arm.get("initial_sha256")}
        comp["same_initialization_hash"] = len(hashes) == 1 and len(comp["arms"]) > 1
        updates = {arm["training"]["logged_optimizer_updates"] for arm in comp["arms"].values()}
        comp["same_observed_update_budget"] = len(updates) == 1 and len(comp["arms"]) > 1
        if "online" in comp["arms"]:
            a = comp["arms"]["online"]["training"]
            comp["same_online_exposure_multiset"] = {}
            for name in ("replay", "replay_ce"):
                if name in comp["arms"]:
                    b = comp["arms"][name]["training"]
                    comp["same_online_exposure_multiset"][name] = bool(a["exposure_multiset"]) and a["exposure_multiset"] == b["exposure_multiset"]
            comp["replay_exposure_multiset_equal"] = comp["same_online_exposure_multiset"].get("replay")
        result["comparison"] = comp
    result["secondary_anls"] = secondary_anls(mining, comparison, attempts, references_directory, lookup)
    if comparison and comparison.get("complete"):
        arms = comparison.get("arms", {})
        floor = bool(arms) and all((arm.get("heldout", {}).get("summary", {}).get("n_pairs", 0) > 0
                                  and arm["heldout"]["summary"].get("both_correct") == 0) for arm in arms.values())
        source_floor = bool(arms) and all(arm.get("sources", {}).get("n_sources", 0) > 0 and arm["sources"].get("accuracy") == 0 for arm in arms.values())
        base_pair = comparison.get("baseline", {}).get("summary", {}).get("both_correct_rate")
        base_source = comparison.get("source_baseline", {}).get("accuracy")
        both_tasks_degraded = (base_pair is not None and base_source is not None and bool(arms)
                               and all(arm.get("heldout", {}).get("summary", {}).get("both_correct_rate", 1) < base_pair
                                       and arm.get("sources", {}).get("accuracy", 1) < base_source for arm in arms.values()))
        result["assessment"] = dict(comparison_complete=True, all_control_arms_at_pair_floor=floor,
                                     all_control_arms_at_untouched_source_floor=source_floor,
                                     all_control_arms_worse_than_baseline_on_both_tasks=both_tasks_degraded,
                                     interpretation="All-zero held-out outcomes cannot rank mining algorithms or establish their equivalence. The tested training carrier fails to generalize; the general mining proposal remains unresolved." if floor else
                                     "Assess matched contrasts and untouched-source outcomes with uncertainty; this pilot does not establish publication value.")
    return result


def percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def change_text(value):
    if not value or value.get("delta") is None:
        return "not available"
    text = f"{100 * value['delta']:+.1f} pp"
    ci = value.get("ci95")
    if ci is not None:
        text += f" [95% CI {100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
    return text + f"; {value.get('n_pairs', 0)} questions / {value.get('n_documents', 0)} documents"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def make_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    split_rows = [[s, c["attempted"], c["teacher_valid"], c["broad_pair_failures"], c["strict_answer_collapses"], c["student_both_correct"]]
                  for s, c in summary["mining"].items()]
    tables = [("Mining counts: failures and both-correct use teacher-valid pairs", split_rows,
               ["Split", "Attempted", "Teacher valid", "Pair fail", "Strict collapse", "Both correct"])]
    if summary.get("carrier_attempts"):
        rows = []
        for attempt in summary["carrier_attempts"]:
            b, a = attempt.get("before") or {}, attempt.get("after") or {}
            verdict = attempt.get("verdict")
            label = "unresolved" if verdict == "UNRESOLVED_CARRIER_OR_OPTIMIZATION" else verdict
            rows.append([attempt["carrier"] + (" (active)" if attempt["active"] else " (previous)"),
                         f"{attempt['training']['logged_optimizer_updates']}/{attempt.get('requested_updates')}",
                         f"{b.get('both_correct', '—')}/{b.get('n_pairs', '—')}",
                         f"{a.get('both_correct', '—')}/{a.get('n_pairs', '—')}",
                         f"{a.get('strict_answer_collapses', '—')}/{a.get('n_pairs', '—')}", label])
        tables.append(("All carrier attempts retained", rows, ["Carrier", "Done/planned", "Fit before", "Fit after", "Collapse after", "Gate verdict"]))
    comparison = summary.get("comparison")
    if comparison:
        rows = []
        for name, values in [("B2 baseline", dict(heldout=comparison.get("baseline"), sources=comparison.get("source_baseline"))), *comparison["arms"].items()]:
            h, s = values.get("heldout") or {}, values.get("sources") or {}
            rows.append([name, f"{h.get('both_correct', 0)}/{h.get('n_pairs', 0)}", percent(h.get("both_correct_rate")),
                         f"{s.get('student_correct', '?')}/{s.get('n_questions', 0)}", percent(s.get("accuracy"))])
        reference = summary.get("references", {}).get("rres512", {})
        if reference.get("all_sources"):
            r = reference["all_sources"]["rres512"]
            pair = summary.get("references", {}).get("rres512_pairs", {}).get("both_correct", {}).get("rres512", {})
            rows.append(["Rres512 reference", f"{pair['correct']}/{pair['n']}" if pair else "—", percent(pair.get("accuracy")),
                         f"{r['correct']}/{r['n']}", percent(r["accuracy"])])
        tables.append(("Fixed evaluation pools: held-out pairs and untouched real pages", rows,
                       ["Arm", "Pair correct", "Pair accuracy", "Real QA correct", "Real QA accuracy"]))
    elif summary.get("gate") and not summary.get("carrier_attempts"):
        g = summary["gate"]
        rows = [[name, (g.get(key) or {}).get("both_correct"), (g.get(key) or {}).get("n_pairs"), percent((g.get(key) or {}).get("both_correct_rate"))]
                for name, key in (("Fit before", "before"), ("Fit after", "after"), ("Held-out after", "heldout"))]
        tables.append(("Carrier gate: " + str(g.get("verdict")), rows, ["Evaluation", "Both correct", "Pairs", "Accuracy"]))
    pair_reference = summary.get("references", {}).get("rres512_pairs", {})
    if pair_reference.get("both_correct") and not comparison:
        rows = []
        for name, r in pair_reference["both_correct"].items():
            collapse = pair_reference["strict_collapse"].get(name)
            rows.append([name, f"{r['correct']}/{r['n']}", percent(r["accuracy"]),
                         f"{collapse['count']}/{collapse['n']}" if collapse else "—"])
        tables.append(("Same held-out teacher-valid pairs: native resolution reference", rows,
                       ["Model", "Both correct", "Pair accuracy", "Strict collapse"]))
    invariants = summary.get("references", {}).get("invariants", {})
    if invariants.get("accepted"):
        rows = [[name, f"{r['correct']}/{r['n']}", percent(r["accuracy"])] for name, r in invariants["accepted"].items()]
        tables.append((f"Unrelated questions: {invariants['teacher_valid_pairs']}/{invariants['attempted_pairs']} teacher-verified pairs",
                       rows, ["Arm", "Both correct", "Accuracy"]))
    heights = [max(1.8, .26 * (len(rows) + 1) + .5) for _, rows, _ in tables]
    figure, axes = plt.subplots(len(tables), 1, figsize=(11.8, sum(heights) + 1), squeeze=False,
                                gridspec_kw={"height_ratios": heights})
    for axis, (title, rows, columns) in zip(axes[:, 0], tables):
        axis.axis("off")
        axis.set_title(title, loc="left", fontsize=11, y=1.02, pad=6)
        table = axis.table(cellText=rows, colLabels=columns, bbox=[0, 0, 1, .86], cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9 if len(rows) > 7 else 10)
        for (row, column), item in table.get_celld().items():
            item.set_edgecolor("#d6dde6")
            if row == 0:
                item.set_facecolor("#243b53")
                item.set_text_props(color="white", weight="bold")
            elif row % 2:
                item.set_facecolor("#edf2f7")
    status = summary["status"]
    complete = (status["mining_complete"] and status["gate_present"]
                and (not status["comparison_present"] or status["comparison_complete"])
                and summary.get("references", {}).get("status") != "running")
    figure.suptitle("Semantic-difference pilot" + ("" if complete else " — incomplete snapshot"), fontsize=15)
    figure.text(.01, .015, "DocVQA VAL custom partitions; teacher-valid synthetic edits and untouched real QA have separate denominators.", fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, .94), h_pad=2.6)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def report_text(s):
    total = s["mining"]["all"]
    g, comparison = s.get("gate"), s.get("comparison")
    assessment = s.get("assessment", {})
    if assessment.get("all_control_arms_at_pair_floor"):
        baseline = comparison.get("baseline") or {}
        leading = (f"**Blind-spot mining produced verified failures, but the tested training pipeline failed to generalize.** "
                   f"Every completed control arm ({len(comparison['arms'])} arms) scored zero both-correct held-out pairs; "
                   f"the B2 baseline scored {baseline.get('both_correct')}/{baseline.get('n_pairs')}. "
                   "This implementation is not a viable method on the tested held-out pool. "
                   "The all-zero primary outcomes cannot rank the mining algorithms or establish equivalence, and do not reject the broader proposal.")
        if assessment.get("all_control_arms_at_untouched_source_floor"):
            leading += " All trained arms also scored zero on the untouched real-question pool."
    elif g and g.get("verdict") != "PASS":
        leading = ("**Blind-spot mining produced verified failures, but the tested carrier did not pass its fit-recovery gate.** "
                   "The training implementation is unresolved, and no mining-control comparison is interpreted as evidence against the general proposal.")
    elif assessment.get("all_control_arms_worse_than_baseline_on_both_tasks"):
        leading = ("**Blind-spot mining produced verified failures, but every trained control degraded both held-out pair accuracy and untouched real-question accuracy "
                   "relative to the B2 baseline.** The principal result is failure of the tested training pipeline to generalize; "
                   "the broader semantic-difference mining proposal remains unresolved.")
    else:
        leading = ("This pilot separates the availability of verified compression failures from carrier trainability, "
                   "held-out generation and the value of refreshed mining.")
    lines = ["# Semantic-difference mining: feasibility pilot results", "", "2026-11-08 (Asia/Shanghai).", "", leading, "",
             f"The full model verified {total['teacher_valid']}/{total['attempted']} attempted pairs. "
             f"Within those teacher-valid pairs, B2 failed on {total['broad_pair_failures']}/{total['teacher_valid']} "
             f"and returned the same normalized answer on {total['strict_answer_collapses']}/{total['teacher_valid']}. "
             "This establishes the presence of candidate compression-specific training signals; training and transfer are assessed separately below.", "",
             "The [preregistered protocol](cf_mining_pilot_prereg.md) defines the gates and controls. "
             "The [primary-source related-work review](cf_mining_prior_art.md) bounds the novelty claim; this pilot does not establish novelty.", "",
             "## Setup and accounting", "",
             f"Model: `{s['setup'].get('model')}`; {s['setup'].get('resolution')} square input, "
             f"1024 → {s['setup'].get('token_budget')} visual tokens, native DeepStack and 3-D mRoPE. "
             "Edits change one numeric digit on a real page; both members have the same local rerendering. "
             "Untouched pages remain a separate real-document evaluation.", ""]
    if s.get("generated_candidates"):
        c = s["generated_candidates"]
        lines += [f"CPU candidate construction scanned {c.get('scanned_questions')} numeric questions on {c.get('scanned_images')} pages "
                  f"and produced {c.get('pairs')} pairs. Model screening attempted the {total['attempted']} pairs shown below. "
                  "Unscreened candidates are not counted as model-verified failures.", ""]
    lines += ["| Split | Attempted | Teacher valid | Teacher invalid | Broad pair failure / valid | Strict collapse / valid | Student both correct / valid |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, c in s["mining"].items():
        n = c["teacher_valid"]
        lines.append(f"| {name} | {c['attempted']} | {n} | {c['teacher_invalid']} | {c['broad_pair_failures']}/{n} | {c['strict_answer_collapses']}/{n} | {c['student_both_correct']}/{n} |")
    lines += ["", "Broad failure means at least one incorrect member. Strict collapse is the subset with the same normalized student output. "
              "Only pairs with correct full-model answers on both members enter these failure denominators. "
              "Correctness uses exact numeric value normalization with percent signs preserved; changing the answer is not sufficient to count as correct.", "",
              "## Three observed strict collapses", "",
              "| Split / pair | Question | Correct original → edit | Full teacher original → edit | Compressed original → edit |",
              "| --- | --- | --- | --- | --- |"]
    for r in s["strict_collapse_examples"]:
        lines.append("| " + " | ".join(cell(v) for v in (f"{r['split']} / {r['pair_id']}", r["question"],
                    f"{r['original_answer']} → {r['edited_answer']}", f"{r['original_teacher_prediction']} → {r['edited_teacher_prediction']}",
                    f"{r['original_student_prediction']} → {r['edited_student_prediction']}")) + " |")
    if len(s["strict_collapse_examples"]) < 3:
        lines += ["", f"Only {len(s['strict_collapse_examples'])} distinct-source strict-collapse examples were available; none were invented."]
    lines += ["", "## Untouched real pages", "",
              "| Split | Questions / source documents | Full teacher accuracy | B2 accuracy |", "| --- | ---: | ---: | ---: |"]
    for name, c in s["untouched_sources"].items():
        lines.append(f"| {name} | {c['n_questions']} / {c['n_documents']} | {c['teacher_correct']}/{c['teacher_labelled_questions']} ({percent(c['teacher_accuracy'])}) | {c['student_correct']}/{c['n_questions']} ({percent(c['student_accuracy'])}) |")
    lines += ["", "These denominators include teacher errors; they are not restricted to teacher-valid edited pairs.", ""]
    if s.get("carrier_attempts"):
        lines += ["## All carrier attempts", "",
                  "Every discovered completed carrier gate and the active attempt are retained below. "
                  "The [preregistered amendments](cf_mining_pilot_prereg.md#trainability-amendment-after-the-first-carrier-failed) "
                  "first add separate zero-initialized low-rank value/output readouts to the four visual streams. "
                  "The final amendment combines a fixed Gaussian local-attention bias (sigma 1.5 merged-token units) "
                  "with a weight-one sequence margin `softplus(1 - log p(correct|image) + log p(opposite answer|image))`. "
                  "These changes follow failed fit gates; held-out performance did not select them. "
                  "Because locality and the margin changed together, this screen cannot attribute any recovery to either alone. "
                  "The carriers are experimental training interfaces, not architectural novelty claims.", "",
                  "| Carrier / attempt | Recorded trainable parameters | Logged / requested updates | Fit both correct before → after | Strict collapse after | Gate verdict | Untouched QA before → after |",
                  "| --- | ---: | ---: | --- | ---: | --- | --- |"]
        for attempt in s["carrier_attempts"]:
            before, after = attempt.get("before") or {}, attempt.get("after") or {}
            lines.append(f"| {attempt['carrier']} / {'active' if attempt['active'] else 'previous'} | "
                         f"{attempt.get('trainable_parameters') if attempt.get('trainable_parameters') is not None else 'not recorded'} | "
                         f"{attempt['training']['logged_optimizer_updates']} / {attempt.get('requested_updates')} | "
                         f"{before.get('both_correct', 'n/a')}/{before.get('n_pairs', 'n/a')} → {after.get('both_correct', 'n/a')}/{after.get('n_pairs', 'n/a')} | "
                         f"{after.get('strict_answer_collapses', 'n/a')}/{after.get('n_pairs', 'n/a')} | "
                         f"{attempt['verdict']} | {percent(attempt.get('source_before'))} → {percent(attempt.get('source_after'))} |")
        active = next((a for a in s["carrier_attempts"] if a["active"]), None)
        if active:
            lines += ["", f"The active carrier records rank {active.get('rank')}, "
                      f"{active.get('requested_updates')} requested gate updates and refresh interval {active.get('refresh_interval')}. "
                      f"The gate pair IDs match the archived attempt: `{active.get('same_gate_pairs_as_previous', 'not applicable')}`. "
                      "Comparison-arm budgets are reported from the actual final logs below.", ""]
        for previous in (a for a in s["carrier_attempts"] if not a["active"] and a["complete"]):
            b, a = previous.get("before") or {}, previous.get("after") or {}
            final_loss = (previous["training"].get("final_history") or {}).get("loss")
            lines += [f"The archived {previous['carrier']} gate made {b.get('both_correct')}/{b.get('n_pairs')} fit pairs both correct "
                      f"before and {a.get('both_correct')}/{a.get('n_pairs')} after {previous['training']['logged_optimizer_updates']} logged updates; "
                      f"{a.get('strict_answer_collapses')}/{a.get('n_pairs')} pairs collapsed afterward (last recorded loss {final_loss}).", ""]
        lines += ["Fewer identical outputs or falling average loss do not count as recovery: both answers must be correct. "
                  "A failed carrier gate does not by itself reject the broader semantic-difference mining idea. "
                  "No training-control comparison is run unless the final carrier passes its fit gate.", ""]
    if g:
        before, after = g.get("before") or {}, g.get("after") or {}
        lines += ["## Carrier training gate", "", f"Observed verdict: **{g.get('verdict')}**. "
                  f"Fit both-correct pairs changed from {before.get('both_correct', 'n/a')}/{before.get('n_pairs', 'n/a')} "
                  f"to {after.get('both_correct', 'n/a')}/{after.get('n_pairs', 'n/a')}; "
                  f"the selected gate contained {g.get('fit_selection_strict_collapses', 'n/a')} strict collapses "
                  f"and {g.get('fit_selection_broad_only_failures', 'n/a')} other pair failures.", "",
                  f"Held-out paired change: {change_text(g.get('heldout_change'))}. "
                  f"Untouched-page paired change: {change_text(g.get('source_change'))}.", "",
                  f"Frozen model version audit: `{g.get('frozen_decoder_unchanged')}`; sampled decoder checksum unchanged: "
                  f"`{g.get('decoder_sample_checksum_unchanged')}`. Observed logged optimizer updates: "
                  f"{g.get('training', {}).get('logged_optimizer_updates', 0)}.", "",
                  "The gate tests whether this carrier and optimizer can repair fit failures. A pass does not establish generalization; "
                  "an unresolved gate does not falsify the general mining proposal.", ""]
    if s.get("infrastructure_events"):
        lines += ["## Implementation audit", "",
                  "Two launches stopped before any optimizer update: a float32 checksum-index interpolation exceeded a large embedding bound, "
                  "and a VLMEvalKit import disabled global autograd. Exact integer index interpolation and an explicit training autograd context fixed these issues. "
                  "These are infrastructure failures before training, not failed method experiments. Mining predictions and the experimental protocol were retained.", ""]
        if g:
            identity = g.get("native_identity") or {}
            lines += [f"The completed active gate records native zero-gate checks `{json.dumps(identity, sort_keys=True)}`; "
                      f"{g.get('training', {}).get('nonzero_gradient_updates', 0)}/{g.get('training', {}).get('logged_optimizer_updates', 0)} logged updates "
                      "had nonzero compressor gradients. Frozen-weight and checksum checks are reported with the gate above.", ""]
    sanity = s.get("train_generation_sanity", {})
    if sanity.get("arms"):
        lines += ["## Post-hoc training-generation diagnostic", "",
                  "After the comparison, a diagnostic checked the first eight fit original images. "
                  "These are individual training answers, not held-out cases or both-correct edit pairs. "
                  "The diagnostic did not choose a model or checkpoint.", "",
                  "| Arm | Correct training originals | Teacher-forced answer NLL range |",
                  "| --- | ---: | ---: |"]
        for name, arm in sanity["arms"].items():
            lo, hi = arm.get("teacher_forced_nll_min"), arm.get("teacher_forced_nll_max")
            span = f"{lo:.6g}–{hi:.6g}" if lo is not None and hi is not None else "not available"
            lines.append(f"| {name} | {arm['correct']}/{arm['n']} | {span} |")
        ordinary = sanity["arms"].get("ordinary")
        if ordinary and ordinary["n"] and ordinary["correct"] == ordinary["n"]:
            lines += ["", "Ordinary training can generate its memorized fit answers with low teacher-forced NLL. "
                      "This rules out an across-the-board inability of the interface to turn learned fit answers into generation, "
                      "while leaving other implementation limits possible. The observed held-out collapse is consistent with severe memorization/generalization failure. "
                      f"Adapter state restored after the diagnostic: `{sanity.get('pool_restored')}`.", ""]
    if comparison:
        relation = bool((comparison.get("config") or {}).get("relation_weight", 0) > 0)
        lines += ["## Matched training controls", "",
                  "The controls use the recorded carrier, optimizer-update budget and positive-answer exposure budget. "
                  "Ordinary uses original values, random uses teacher-valid edited pairs, fixed uses initial failures, "
                  "online refreshes fit examples using the current compressor, and replay shuffles the same online exposure multiset. "
                  "Final checkpoints are evaluated without held-out checkpoint selection.", "",
                  f"Observed audits: shared initialization hash = `{comparison.get('same_initialization_hash')}`; "
                  f"equal logged update counts = `{comparison.get('same_observed_update_budget')}`; "
                  f"exact online-data exposure multisets = `{comparison.get('same_online_exposure_multiset', {})}`.", "",
                  "| Arm | Held-out both correct | Pair accuracy | Change from B2, document-paired 95% CI | Untouched QA accuracy | Untouched change from B2 | Logged updates |",
                  "| --- | ---: | ---: | --- | ---: | --- | ---: |"]
        for name, values in [("B2 baseline", dict(heldout=comparison.get("baseline"), sources=comparison.get("source_baseline"))), *comparison["arms"].items()]:
            h, source = values.get("heldout") or {}, values.get("sources") or {}
            lines.append(f"| {name} | {h.get('both_correct', 0)}/{h.get('n_pairs', 0)} | {percent(h.get('both_correct_rate'))} | "
                         f"{change_text(values.get('heldout_change'))} | {percent(source.get('accuracy'))} | {change_text(values.get('source_change'))} | "
                         f"{values.get('training', {}).get('logged_optimizer_updates', '—')} |")
        lines += ["", "| Online contrast | Held-out both-correct change | Untouched QA change |", "| --- | --- | --- |"]
        for name, value in comparison["paired_contrasts"].items():
            lines.append(f"| {name} | {change_text(value)} | {change_text(comparison['source_contrasts'].get(name))} |")
        lines += ["", "| Arm | Actual updates | Positive-answer forwards | Negative-answer forwards | Mining-answer forwards | Training wall seconds |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, arm in comparison["arms"].items():
            training = arm.get("training") or {}
            forwards = training.get("forward_counts") or {}
            lines.append(f"| {name} | {training.get('logged_optimizer_updates')} | {forwards.get('training_answer', 'n/a')} | "
                         f"{forwards.get('training_negative_answer', 0)} | {forwards.get('mining_answer', 'n/a')} | {training.get('wall_seconds')} |")
        if relation:
            lines += ["", "Ordinary and `replay_ce` use CE plus bucket KL. Random, fixed, online, and replay additionally use the same sequence-margin objective. "
                      "`replay_ce` uses the exact online positive-data multiset with CE plus KL only. "
                      "Contrast arms perform an additional negative-answer forward per positive exposure. "
                      "Equal optimizer steps and positive exposures therefore do **not** imply equal total computation. "
                      "The ordinary contrast confounds edited data, loss, and extra forwards; the replay_ce contrast also includes a loss/compute difference. "
                      "Random/fixed/online/replay share the loss, while online adds refreshed search cost.", ""]
        lines += ["The table separates observed training and mining forwards. One initial current-model hard-set search is shared by fixed and online; "
                  f"its recorded plan is `{json.dumps(comparison.get('initial_search'), sort_keys=True)}`. "
                  "Per-arm objective-name unions may include previously executed arms; negative-forward counts identify actual contrast computation.", ""]
        search = comparison.get("search_diagnostic", {})
        if search.get("rounds"):
            lines += ["### What the current-model search finds", "",
                      f"Observed search rounds: {search.get('initial_shared_rounds')} initial round shared by fixed/online and "
                      f"{search.get('online_refresh_rounds')} online refresh rounds ({search.get('total_observed_rounds')} total). "
                      "The same fit pairs are joined to their B2-initialization outcomes. Baseline-correct → current-failure counts training breakage; "
                      "baseline-failure → current-both-correct counts rescue. Repeated screening rounds are not independent new examples.", "",
                      "| Search phase / step | Scored / labelled candidates | Current both correct | Current failures | Current strict collapse | New failure / baseline correct | Rescue / baseline failures |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for r in search["rounds"]:
                lines.append(f"| {r['phase']} / {r['step']} | {r['scored_candidates']} / {r['labelled_teacher_valid_candidates']} | "
                             f"{r['current_both_correct']} | {r['current_failures']} | {r['current_strict_collapses']} | "
                             f"{r['baseline_correct_to_failure']}/{r['baseline_both_correct']} | {r['baseline_failure_to_both_correct']}/{r['baseline_failures']} |")
            if any(r["proxy_only_unlabelled_candidates"] for r in search["rounds"]):
                lines += ["", "Proxy-only search rows without generated-answer correctness are excluded from these outcome counts."]
            if any(r["phase"] == "online_refresh" and r["baseline_correct_to_failure"] for r in search["rounds"]):
                lines += ["", "The refreshed candidate failures include pairs that the initial compressed model answered correctly. "
                          "Thus the search is partly reacting to training-induced breakage. A plausible interpretation is that the refreshed hard pool "
                          "can track damage caused by the updating carrier; these logs alone do not establish the cause of the generalization failure."]
            lines += [""]
        lines += ["", "Intervals resample whole source documents and preserve paired question outcomes. "
                  "A confidence interval crossing zero does not resolve an advantage. Online-versus-replay differences address ordering/refresh, "
                  "while similar replay performance can attribute benefit to selected data. Any relation-loss comparison is subject to the extra-forward compute difference.", ""]
        if assessment.get("all_control_arms_at_pair_floor"):
            lines += ["Every arm is at the primary accuracy floor. Degenerate paired intervals such as `[0, 0]` describe identical observed zero outcomes; "
                      "they do not demonstrate algorithmic equivalence or a sufficiently sensitive comparison. "
                      "The carrier/generalization failure takes precedence over any ranking among these mining schedules.", ""]
    else:
        lines += ["## Matched training controls", "", "No completed comparison artifact was available. "
                  "The advantage of online mining over random edits, fixed failures, ordinary training, and same-data replay remains untested.", ""]
    references = s.get("references", {})
    rres, inv = references.get("rres512", {}), references.get("invariants", {})
    lines += ["## Same-source quality references", ""]
    pair_ref = references.get("rres512_pairs", {})
    if pair_ref.get("both_correct"):
        lines += [f"Same-pair Rres512 status: **{pair_ref.get('status')}**. "
                  f"The {pair_ref['n_pairs']} held-out pairs ({pair_ref['n_documents']} source documents) "
                  "were selected by teacher validity before the resolution-reference predictions, with no filtering by compressed-model success.", "",
                  "| Same held-out pairs | Both correct / pairs | Pair accuracy | Strict answer collapse / pairs |",
                  "| --- | ---: | ---: | ---: |"]
        for name, r in pair_ref["both_correct"].items():
            collapse = pair_ref["strict_collapse"].get(name)
            lines.append(f"| {name} | {r['correct']}/{r['n']} | {percent(r['accuracy'])} | "
                         f"{str(collapse['count']) + '/' + str(collapse['n']) if collapse else '—'} |")
        lines += ["", f"Rres512 minus B2 in both-correct accuracy: {change_text(pair_ref.get('rres512_minus_b2'))}; "
                  f"rescued {pair_ref.get('rres512_minus_b2_rescued')}, broke {pair_ref.get('rres512_minus_b2_broken')}, "
                  f"and repaired {pair_ref.get('b2_strict_collapses_repaired_to_both_correct')} B2 strict collapses to two correct answers. "
                  f"Observed per-member input/decoder token counts: {pair_ref.get('observed_input_visual_tokens')}/{pair_ref.get('observed_decoder_visual_tokens')}.", "",
                  "This reference tests whether the same answer-critical differences remain difficult for an alternative compression baseline. "
                  "Failures mined from B2 alone do not establish a weakness shared by stronger compressors.", ""]
        if pair_ref.get("arm_minus_rres512"):
            lines += ["| Trained arm minus Rres512 | Paired both-correct change on matching pairs |", "| --- | --- |"]
            for name, contrast in pair_ref["arm_minus_rres512"].items():
                lines.append(f"| {name} | {change_text(contrast)} |")
            lines += [""]
    else:
        lines += [f"Same-pair Rres512 reference: **{pair_ref.get('status', 'not_run')}**. No held-out pair result is inferred from the untouched-page reference.", ""]
    if rres.get("all_sources"):
        lines += [f"Rres512 status: **{rres.get('status')}**. This keeps all 256 visual tokens produced at 512×512, "
                  "whereas B2 starts from 1024 tokens and keeps 256. Both use native DeepStack and 3-D positions. "
                  "These are accuracy references; no end-to-end latency claim follows.", "",
                  "| Same untouched source pool | Correct / questions | Accuracy |", "| --- | ---: | ---: |"]
        for name, r in rres["all_sources"].items():
            lines.append(f"| {name} | {r['correct']}/{r['n']} | {percent(r['accuracy'])} |")
        lines += ["", f"Rres512 minus B2: {change_text(rres.get('rres512_minus_b2'))}; "
                  f"rescued {rres.get('rres512_minus_b2_rescued')} and broke {rres.get('rres512_minus_b2_broken')} questions. "
                  f"Observed input/decoder token counts: {rres.get('observed_input_visual_tokens')}/{rres.get('observed_decoder_visual_tokens')}; "
                  f"resolution restored to {rres.get('resolution_restored')}.", ""]
        if rres.get("arm_minus_rres512"):
            lines += ["| Trained arm minus Rres512 | Paired accuracy change on matching questions |", "| --- | --- |"]
            for name, contrast in rres["arm_minus_rres512"].items():
                lines.append(f"| {name} | {change_text(contrast)} |")
            lines += [""]
    else:
        lines += [f"Same-source Rres512 reference: **{rres.get('status', 'not_run')}**. No result is inferred from historical experiments.", ""]
    if inv.get("accepted"):
        lines += [f"Unrelated sibling-question check: **{inv.get('status')}**; "
                  f"{inv['teacher_valid_pairs']}/{inv['attempted_pairs']} attempted pairs passed full-teacher verification on both images "
                  f"({inv['teacher_rejected_pairs']} rejected). These are invariance checks, not field-swap or arithmetic interventions.", "",
                  "| Teacher-verified invariant pairs | Both correct / accepted | Accuracy |", "| --- | ---: | ---: |"]
        for name, r in inv["accepted"].items():
            lines.append(f"| {name} | {r['correct']}/{r['n']} | {percent(r['accuracy'])} |")
        lines += ["", f"Online minus B2: {change_text(inv.get('online_minus_b2'))}; "
                  f"rescued {inv.get('online_rescued')}, broke {inv.get('online_broken')}; "
                  f"adapter state restored after the check: `{inv.get('pool_restored')}`. "
                  f"These {inv['attempted_pairs']} sibling-question pairs were evaluated only; unrelated-question invariance constraints were not trained.", ""]
        if inv.get("skipped_reason"):
            lines += [f"Skip reason: {inv['skipped_reason']}.", ""]
    else:
        lines += [f"Supplementary online/B2 unrelated-question check: **{inv.get('status', 'not_run')}**. "
                  "No invariant preservation result is available.", ""]
    anls = s.get("secondary_anls")
    if anls:
        lines += ["## Secondary DocVQA ANLS", "",
                  "Exact numeric correctness remains primary: a one-digit error fails even when the text is nearly identical. "
                  "Secondary ANLS uses the repository's official `anls_compute` and `hit_calculate` functions against all available original/edited aliases, "
                  "with their own case/whitespace/string rules and no Decimal normalization. "
                  "`anls_compute` returns normalized edit distance; `hit_calculate` converts the best alias to similarity and applies the official threshold. "
                  "For example, `1997` versus `1996` has distance 0.25 and ANLS 0.75, while primary exact correctness is zero. "
                  "The scores below average individual answers, not both-correct pairs, and are not full-benchmark DocVQA results.", "",
                  "| Evaluation | Model / arm | Scored individual answers | Secondary ANLS (0–100) |",
                  "| --- | --- | ---: | ---: |"]
        def append_anls(scope, arm, values):
            value = values.get("mean_anls")
            score = "n/a" if value is None else f"{100 * value:.2f}"
            lines.append(f"| {scope} | {arm} | {values.get('n_individual_answers', 0)} | {score} |")
        for arm, values in anls["mining"].get("all", {}).get("teacher_valid", {}).items():
            append_anls("Teacher-valid mined pair members", arm, values)
        for arm, values in anls["untouched_sources"].get("all", {}).items():
            append_anls("Untouched real questions", arm, values)
        for attempt in anls.get("carrier_fit_attempts", []):
            for phase in ("before", "after"):
                for arm, values in attempt[phase].items():
                    append_anls(f"{attempt['carrier']} fit members {phase}", arm, values)
        ac = anls.get("comparison", {})
        for arm, values in ac.get("baseline_pairs", {}).items():
            append_anls("Fixed held-out pair members", "B2 baseline", values)
        for name, arm in ac.get("arms", {}).items():
            for values in arm.get("heldout_members", {}).values():
                append_anls("Fixed held-out pair members", name, values)
            for values in arm.get("untouched_sources", {}).values():
                append_anls("Untouched real questions", name, values)
        for kind, reference in anls.get("references", {}).items():
            for arm, values in reference.get("scores", {}).items():
                append_anls("Reference pair members" if kind == "pairs" else "Teacher-verified invariant members" if kind == "invariants" else "Reference untouched questions", arm, values)
        lines += ["", f"Official-function source SHA-256: `{anls['implementation']['source_sha256']}`. "
                  "The analysis loads only those pure function definitions, avoiding model/package imports. "
                  "ANLS partial credit must not be interpreted as preservation of the answer-critical digit distinction.", ""]
    iv = s["invariants"]
    original_invariance = (f"Mining-stage unrelated-field checks: {iv['teacher_valid']}/{iv['attempted']} teacher-verified invariant pairs; "
                           f"compressed model both-correct on {iv['student_both_correct']}/{iv['teacher_valid']}.") if iv["attempted"] else "The original mining-stage invariant check was not run; supplementary checks are reported separately above."
    lines += ["## Limits and next decision", "", original_invariance, ""]
    for limitation in s["limitations"]:
        lines.append("- " + limitation)
    lines += ["", "Continue only on evidence from real untouched QA, matched controls and additional sources/layouts. "
              "Strict-collapse yield alone supports mining availability, not publication value or a final architecture.", "",
              "Machine-readable accounting and all observed denominators are in "
              "`Qwen_vl/outputs/cf_mining/report/summary.json`; the compact table figure is `overview.png` alongside it.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mining", default=str(OUT / "mining.json"))
    ap.add_argument("--manifest", default=str(OUT / "manifest.json"))
    ap.add_argument("--gate", default=str(OUT / "training" / "gate_result.json"))
    ap.add_argument("--comparison", default=str(OUT / "training" / "comparison.json"))
    ap.add_argument("--references-dir", help="defaults to the comparison artifact directory")
    ap.add_argument("--infrastructure", help="defaults to infrastructure_events.json beside the gate")
    ap.add_argument("--previous-gate", action="append", help="repeatable archived gate path; defaults to training_global_gate and training_global_readout")
    ap.add_argument("--out", default=str(OUT / "report"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--write-report", action="store_true")
    ap.add_argument("--report", default=str(ROOT / "docs" / "cf_mining_pilot_results.md"))
    args = ap.parse_args()
    if args.bootstrap < 2:
        ap.error("--bootstrap must be at least 2")
    result = summarize(args)
    out = Path(args.out)
    atomic_json(out / "summary.json", result)
    if not args.no_figure:
        make_figure(result, out / "overview.png")
    if args.write_report:
        state = result["status"]
        if not state["mining_complete"] or not result.get("gate", {}).get("verdict") or (state["comparison_present"] and not state["comparison_complete"]):
            raise RuntimeError("report requires complete mining, a completed gate verdict, and completion of any existing comparison; JSON/figure snapshot was saved")
        if result["references"]["status"] == "running":
            raise RuntimeError("supplementary reference checks are still running; JSON/figure snapshot was saved")
        if result["prediction_flag_discrepancies"]:
            raise RuntimeError("stored prediction flags disagree with exact recomputation; inspect summary before writing a report")
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(report_text(result), encoding="utf-8")
        print(str(report.resolve()))
    print(json.dumps(dict(status=result["status"], mining=result["mining"]["all"], summary=str((out / "summary.json").resolve())), ensure_ascii=False))


if __name__ == "__main__":
    main()
