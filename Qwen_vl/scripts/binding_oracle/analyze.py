"""CPU-only, pending-safe analysis of the preregistered binding oracle pilot.

Recompute correctness from the manifest rather than trusting stored flags.
No model, torch, or VLMEvalKit package is imported. Optional ANLS extracts only
the repository's three pure scoring functions and uses hit_calculate, because
anls_compute itself returns distance, not the final DocVQA score.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import random
import re
from types import SimpleNamespace


QWEN = Path(__file__).resolve().parents[2]
DEFAULT = QWEN / "outputs/binding_oracle"
MEMBERS = ("original_f0", "original_f1", "swapped_f0", "swapped_f1")
CORE_ARMS = ("teacher", "visionzip", "visionzip_ppe", "field_anchor_control_ppe",
             "field_oracle", "field_oracle_ppe", "random_oracle_ppe",
             "spatial_oracle_ppe", "rres512")
AUDIT_ARMS = ("historical", "historical_ds_fixed")
SENSITIVITY_ARMS = ("line_anchor_control_ppe", "line_oracle_ppe")
PRIMARY = "field_oracle_ppe"
REFERENCE = "field_anchor_control_ppe"


def normalize(value):
    value = str(value).strip().lower().strip("'\"`")
    value = re.sub(r"[\s,$£€]", "", value).replace("−", "-")
    suffix = "%" if value.endswith("%") else ""
    number = value[:-1] if suffix else value
    try:
        return str(Decimal(number).normalize()) + suffix
    except InvalidOperation:
        return value


def correct(prediction, answers):
    return any(normalize(prediction) == normalize(answer) for answer in answers)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def anls_scorer():
    source = QWEN / "VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py"
    content = source.read_text()
    names = {"anls_compute", "hit_calculate", "levenshtein_distance"}
    functions = [node for node in ast.parse(content).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise RuntimeError("official DocVQA scoring functions could not be located")
    namespace = {"np": SimpleNamespace(min=min, max=max,
                                      mean=lambda x: sum(x) / len(x)),
                 "listinstr": lambda patterns, value: any(x in value for x in patterns)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)

    def score(prediction, answers):
        distances = [namespace["anls_compute"](answer, prediction) for answer in answers]
        return float(namespace["hit_calculate"]([{"match": distances}], "DocVQA")[0])
    return score, dict(source=str(source), sha256=hashlib.sha256(content.encode()).hexdigest(),
                       interpretation="official distance then hit_calculate similarity/threshold; no Decimal normalization")


def paired_bootstrap(before, after, draws=10000, seed=20261108):
    """Resample complete source documents while keeping page pairs intact."""
    common = sorted(set(before) & set(after))
    groups = defaultdict(list)
    for pair_id in common:
        a, b = before[pair_id], after[pair_id]
        if a["source_document_id"] != b["source_document_id"]:
            raise ValueError("paired page changed source document")
        groups[a["source_document_id"]].append(int(b["all_correct"]) - int(a["all_correct"]))
    differences = [x for values in groups.values() for x in values]
    result = dict(n_pages=len(common), n_documents=len(groups), pair_ids=common,
                  rescued=sum(after[p]["all_correct"] and not before[p]["all_correct"] for p in common),
                  broken=sum(before[p]["all_correct"] and not after[p]["all_correct"] for p in common),
                  delta=sum(differences) / len(differences) if differences else None,
                  ci95=None, bootstrap_draws=draws, bootstrap_seed=seed,
                  complete_pairing=set(before) == set(after),
                  missing_before_pair_ids=sorted(set(after) - set(before)),
                  missing_after_pair_ids=sorted(set(before) - set(after)))
    result["net_repairs"] = result["rescued"] - result["broken"]
    if len(groups) >= 2:
        keys = sorted(groups)
        rng = random.Random(seed)
        samples = []
        for _ in range(draws):
            sample = [x for key in rng.choices(keys, k=len(keys)) for x in groups[key]]
            samples.append(sum(sample) / len(sample))
        samples.sort()
        result["ci95"] = [samples[math.floor(.025 * (draws - 1))],
                          samples[math.floor(.975 * (draws - 1))]]
        if all(x == 0 for x in differences):
            result["interval_note"] = "identical observed outcomes; a degenerate interval does not establish algorithmic equivalence"
    elif groups:
        result["interval_note"] = "fewer than two independent source documents"
    return result


def paired_question_bootstrap(before, after, draws=10000, seed=20261108):
    """Secondary exact-question difference, resampling source documents intact."""
    common = sorted(set(before) & set(after))
    groups = defaultdict(list)
    rescued = broken = 0
    for pair_id in common:
        a, b = before[pair_id], after[pair_id]
        if a["source_document_id"] != b["source_document_id"]:
            raise ValueError("paired question changed source document")
        if set(a["questions"]) != set(b["questions"]):
            raise ValueError("paired complete pages have different question sets")
        for key in sorted(a["questions"]):
            old, new = bool(a["questions"][key]["correct"]), bool(b["questions"][key]["correct"])
            groups[a["source_document_id"]].append(int(new) - int(old))
            rescued += int(new and not old)
            broken += int(old and not new)
    differences = [value for values in groups.values() for value in values]
    output = dict(n_pages=len(common), n_questions=len(differences), n_documents=len(groups),
                  rescued=rescued, broken=broken, net_question_repairs=rescued - broken,
                  delta=sum(differences) / len(differences) if differences else None,
                  ci95=None, bootstrap_draws=draws, bootstrap_seed=seed,
                  unit="exact numeric-value question; source-document cluster resampling",
                  interpretation="secondary endpoint only; does not replace the preregistered all-four primary")
    if len(groups) >= 2:
        rng = random.Random(seed)
        keys = sorted(groups)
        values = []
        for _ in range(draws):
            sample = [value for key in rng.choices(keys, k=len(keys)) for value in groups[key]]
            values.append(sum(sample) / len(sample))
        values.sort()
        output["ci95"] = [values[math.floor(.025 * (draws - 1))], values[math.floor(.975 * (draws - 1))]]
        if all(value == 0 for value in differences):
            output["interval_note"] = "identical observed question outcomes; not equivalence evidence"
    return output


def _field_answers(item, side, field):
    return list(item["fields"][field]["answers_" + side])


def _bag_matches(predictions, answer_lists):
    """Small exact bipartite match; supports multiple legitimate aliases."""
    if len(predictions) != len(answer_lists):
        return False
    def visit(index, used):
        if index == len(predictions):
            return True
        return any(correct(predictions[index], aliases) and visit(index + 1, used | {j})
                   for j, aliases in enumerate(answer_lists) if j not in used)
    return visit(0, set())


def _entry(entry, answers, score_anls=None):
    result = dict(prediction=entry["prediction"], answers=answers,
                  correct=correct(entry["prediction"], answers),
                  truncated=entry.get("truncated"), n_vis=entry.get("n_vis"),
                  ds_lengths=entry.get("ds_lengths"), selector=entry.get("selector"),
                  decode_positions=entry.get("decode_positions"),
                  expected_decode_start=entry.get("expected_decode_start"))
    if score_anls:
        result["anls"] = score_anls(entry["prediction"], answers)
    return result


def dataset_stratum(item):
    """Use dataset/provenance, never model outcomes, to separate populations."""
    values = [item.get("dataset", ""), item.get("provenance", {}).get("qa_source", "")]
    for value in values:
        value = str(value).upper()
        if "TRAIN" in value:
            return "TRAIN"
        if "VAL" in value:
            return "VAL"
    return "UNKNOWN"


def geometry_entry_ok(entry, arm, expected, base_directory, cache, issues, context):
    """Read only a prediction's referenced diagnostic; never search old files."""
    if expected is None or arm in ("teacher", "rres512"):
        return True
    raw = (entry.get("selector") or {}).get("diagnostic_path")
    path = Path(raw) if raw else None
    if path is not None and not path.is_absolute():
        path = Path(base_directory) / path
    key = str(path) if path is not None else None
    if key not in cache:
        try:
            diag = json.loads(path.read_text()) if path is not None else {}
            cache[key] = (diag.get("geometry_version"), None)
        except (OSError, ValueError) as error:
            cache[key] = (None, type(error).__name__)
    actual, error = cache[key]
    if actual == expected:
        return True
    issues.append(dict(**context, arm=arm, diagnostic_path=key,
                       expected_geometry_version=expected, diagnostic_geometry_version=actual,
                       reason="missing/unreadable diagnostic" if error or path is None else "geometry version mismatch",
                       action="excluded from scoring; remains pending"))
    return False


def score_page(item, raw, arm, score_anls=None, source=False, geometry_version=None,
               base_directory=DEFAULT, geometry_cache=None, geometry_issues=None):
    table = raw.get("questions" if source else "members", {})
    keys = ("f0", "f1") if source else MEMBERS
    entries = {}
    for key in keys:
        entry = table.get(key, {}).get(arm)
        if entry is None or "prediction" not in entry:
            continue
        if not geometry_entry_ok(entry, arm, geometry_version, base_directory,
                                 geometry_cache if geometry_cache is not None else {},
                                 geometry_issues if geometry_issues is not None else [],
                                 dict(pair_id=item["pair_id"], question=key, source=source)):
            continue
        field = int(key[-1])
        side = "original" if source else key.split("_", 1)[0]
        entries[key] = _entry(entry, _field_answers(item, side, field), score_anls)
    result = dict(pair_id=item["pair_id"], source_document_id=str(item["source_document_id"]),
                  source_image=item["source_image"], split=item.get("split"),
                  dataset=item.get("dataset"), stratum=dataset_stratum(item),
                  complete=len(entries) == len(keys), expected_questions=len(keys),
                  available_questions=len(entries), questions=entries)
    if not result["complete"]:
        result["missing_questions"] = [key for key in keys if key not in entries]
        return result
    result["all_correct"] = all(row["correct"] for row in entries.values())
    predictions = [entries[key]["prediction"] for key in keys]
    answer_lists = [entries[key]["answers"] for key in keys]
    result["answer_multiset_exact"] = _bag_matches(predictions, answer_lists)
    value_groups = [_field_answers(item, "original", field) for field in (0, 1)]
    result["answer_set_complete"] = all(any(correct(pred, aliases) for pred in predictions)
                                        for aliases in value_groups)
    result["binding_wrong_despite_complete_multiset"] = result["answer_multiset_exact"] and not result["all_correct"]
    side_results = {}
    for side in (("source",) if source else ("original", "swapped")):
        side_keys = keys if source else (side + "_f0", side + "_f1")
        answers = [_field_answers(item, "original" if source else side, field) for field in (0, 1)]
        p0, p1 = [entries[key]["prediction"] for key in side_keys]
        assigned = correct(p0, answers[0]) and correct(p1, answers[1])
        swapped = correct(p0, answers[1]) and correct(p1, answers[0]) and not assigned
        side_results[side] = dict(both_correct=assigned, numeric_values_both_recognized=assigned or swapped,
                                  both_values_recognized_but_field_assignment_swapped=swapped)
    result["sides"] = side_results
    result["any_side_assignment_swap"] = any(x["both_values_recognized_but_field_assignment_swapped"] for x in side_results.values())
    result["all_sides_numeric_sets_recovered"] = all(x["numeric_values_both_recognized"] for x in side_results.values())
    return result


def summary(rows, expected):
    complete = [row for row in rows if row["complete"]]
    questions = [question for row in rows for question in row["questions"].values()]
    n = len(complete)
    return dict(status="complete" if n == expected else "pending", expected_pages=expected,
                complete_pages=n, pending_pages=expected - n,
                n_documents=len({row["source_document_id"] for row in complete}),
                all_correct=sum(row["all_correct"] for row in complete),
                page_accuracy=sum(row["all_correct"] for row in complete) / n if n else None,
                individual_questions=len(questions),
                expected_individual_questions=sum(row["expected_questions"] for row in rows),
                individual_correct=sum(row["correct"] for row in questions),
                question_accuracy=sum(row["correct"] for row in questions) / len(questions) if questions else None,
                any_side_assignment_swap=sum(row["any_side_assignment_swap"] for row in complete),
                all_sides_numeric_sets_recovered=sum(row["all_sides_numeric_sets_recovered"] for row in complete),
                answer_set_complete=sum(row["answer_set_complete"] for row in complete),
                answer_multiset_exact=sum(row["answer_multiset_exact"] for row in complete),
                binding_wrong_despite_complete_multiset=sum(row["binding_wrong_despite_complete_multiset"] for row in complete),
                truncated_questions=sum(bool(row["truncated"]) for row in questions),
                secondary_anls=(sum(row["anls"] for row in questions) / len(questions)
                                if questions and "anls" in questions[0] else None))


def comparisons(arms, pairs, draws, seed):
    out = {}
    for label, before, after, interpretation in pairs:
        b = {row["pair_id"]: row for row in arms.get(before, {}).get("records", []) if row["complete"]}
        a = {row["pair_id"]: row for row in arms.get(after, {}).get("records", []) if row["complete"]}
        result = paired_bootstrap(b, a, draws, seed)
        result["question_endpoint"] = paired_question_bootstrap(b, a, draws, seed)
        # Historical audit arms are scheduled only on the first 12 candidate
        # pages. Match their declared subset, not an invented missing tail.
        expected = len({r["pair_id"] for r in arms.get(before, {}).get("records", [])}
                       & {r["pair_id"] for r in arms.get(after, {}).get("records", [])})
        result.update(before=before, after=after, interpretation=interpretation,
                      expected_pages=expected, status="complete" if expected and result["n_pages"] == expected else "pending")
        out[label] = result
    return out


CONTRASTS = [
    ("restricted_aggregation_same_anchors", REFERENCE, PRIMARY,
     "same region-repaired anchors, different assignments; PPE representatives may also change"),
    ("oracle_minus_visionzip_ppe", "visionzip_ppe", PRIMARY, "joint allocation/assignment/position difference"),
    ("oracle_minus_random", "random_oracle_ppe", PRIMARY, "field labels versus exact-count randomized labels"),
    ("oracle_minus_spatial", "spatial_oracle_ppe", PRIMARY, "field labels versus 4x4 equal-area labels, not size-matched"),
    ("oracle_minus_rres512", "rres512", PRIMARY, "same decoder K; different encoder work, no equal-compute claim"),
    ("anchor_allocation_only", "visionzip_ppe", REFERENCE, "region-informed anchor allocation with unrestricted assignments"),
    ("ordinary_ppe_only", "visionzip", "visionzip_ppe", "same features/anchors, native-position versus PPE operator"),
    ("oracle_ppe_only", "field_oracle", PRIMARY, "same field-constrained features/anchors, position operator changes"),
    ("historical_ds_repair", "historical", "historical_ds_fixed", "D238/C18; correct DS anchor omission only"),
    ("contextual_budget_change", "historical_ds_fixed", "visionzip", "D238/C18 to D205/C51; same corrected stream rule"),
    ("line_restricted_aggregation", "line_anchor_control_ppe", "line_oracle_ppe", "eligible OCR-line subset only; sensitivity analysis"),
]


def diagnostic_analysis(arms, base_directory):
    by_arm = {}
    errors = []
    for arm, payload in arms.items():
        paths = {q.get("selector", {}).get("diagnostic_path")
                 for row in payload["records"] for q in row["questions"].values()
                 if q.get("selector")}
        records = []
        for raw in sorted(path for path in paths if path):
            path = Path(raw)
            if not path.is_absolute():
                path = base_directory / path
            if not path.exists():
                errors.append(dict(arm=arm, path=str(path), reason="pending diagnostic file"))
                continue
            diag = json.loads(path.read_text())
            reps = diag.get("ppe_representatives", [])
            labels = diag.get("ppe_representative_field_labels", [])
            unique = [len({token for tokens, labs in zip(reps, labels)
                           for token, label in zip(tokens, labs) if label == field}) for field in (1, 2)]
            slots = diag.get("ppe_field_position_slots")
            records.append(dict(path=str(path), image=diag.get("image"),
                                geometry_version=diag.get("geometry_version"), fetch_size=diag.get("fetch_size"),
                                original_anchor_idx=diag.get("original_anchor_idx"),
                                anchor_idx=diag.get("anchor_idx"), dominant_idx=diag.get("dominant_idx"),
                                replacement_count=diag.get("replacement_count"),
                                field_tokens_kept=diag.get("field_tokens_kept"),
                                ppe_field_position_slots=slots, unique_field_position_ids=unique,
                                both_fields_have_ppe_position=all(x > 0 for x in slots) if slots is not None else None,
                                all_tokens_accounted_once=diag.get("all_tokens_accounted_once"),
                                unassigned_count=diag.get("unassigned_count"),
                                cross_label_assignees=diag.get("cross_label_assignees")))
        by_arm[arm] = dict(n_images=len(records),
                           both_fields_have_ppe_position=sum(r["both_fields_have_ppe_position"] is True for r in records),
                           missing_field_position_images=sum(r["both_fields_have_ppe_position"] is False for r in records),
                           total_anchor_replacements=sum(r["replacement_count"] or 0 for r in records), records=records)
    control = {r["image"]: r for r in by_arm.get(REFERENCE, {}).get("records", [])}
    same = []
    for oracle in by_arm.get(PRIMARY, {}).get("records", []):
        if oracle["image"] in control:
            ref = control[oracle["image"]]
            complete = all(isinstance(row[key], list) and len(row[key]) > 0
                           for row in (oracle, ref) for key in ("anchor_idx", "dominant_idx"))
            same.append(dict(image=oracle["image"], complete=complete,
                             anchors_equal=complete and oracle["anchor_idx"] == ref["anchor_idx"],
                             dominant_equal=complete and oracle["dominant_idx"] == ref["dominant_idx"],
                             control_position_slots=ref["ppe_field_position_slots"],
                             oracle_position_slots=oracle["ppe_field_position_slots"]))
    return dict(by_arm=by_arm, missing_or_pending_files=errors, same_anchor_checks=same,
                all_matched_anchors_equal=all(r["anchors_equal"] and r["dominant_equal"] for r in same) if same else None,
                interpretation="representative counts are a joint content/position diagnostic; same anchors do not hold PPE representative positions fixed")


def literal_identifier_sensitivity(items, arms, sources):
    """Descriptive code check; keep the registered Decimal scorer unchanged."""
    def literal(value):
        return re.sub(r"[\s,$£€]", "", str(value).strip().strip("'\"`")).replace("−", "-")
    def leading_zero(answers):
        return any(re.match(r"^[+-]?0[0-9]", literal(answer)) is not None for answer in answers)
    fields = [dict(pair_id=item["pair_id"], field_id=field.get("field_id", f"f{index}"),
                   dataset=item.get("dataset"), question=field.get("question"),
                   answers_original=field["answers_original"])
              for item in items for index, field in enumerate(item["fields"])
              if leading_zero(field["answers_original"])]
    item_byid = {item["pair_id"]: item for item in items}
    def score(container, source=False):
        output = {}
        for arm, payload in container.items():
            records = []
            expected = 0
            for row in payload["records"]:
                item = item_byid[row["pair_id"]]
                for key in (("f0", "f1") if source else MEMBERS):
                    side = "original" if source else key.split("_", 1)[0]
                    expected += int(leading_zero(_field_answers(item, side, int(key[-1]))))
                for key, question in row["questions"].items():
                    if not leading_zero(question["answers"]):
                        continue
                    literal_correct = literal(question["prediction"]) in {literal(a) for a in question["answers"]}
                    records.append(dict(pair_id=row["pair_id"], question=key, prediction=question["prediction"],
                                        answers=question["answers"], primary_correct=question["correct"],
                                        literal_correct=literal_correct))
            output[arm] = dict(expected_questions=expected, available_questions=len(records),
                               pending_questions=expected - len(records),
                               primary_value_correct=sum(r["primary_correct"] for r in records),
                               literal_correct=sum(r["literal_correct"] for r in records),
                               primary_credit_without_literal_match=sum(r["primary_correct"] and not r["literal_correct"] for r in records),
                               records=records)
        return output
    collisions = [item["pair_id"] for item in items
                  if ({normalize(a) for a in _field_answers(item, "original", 0)}
                      & {normalize(a) for a in _field_answers(item, "original", 1)})]
    return dict(planned_leading_zero_fields=len(fields), planned_leading_zero_pairs=len({f["pair_id"] for f in fields}),
                all_planned_pair_targets_distinct_under_primary=not collisions,
                numeric_target_collision_pair_ids=collisions,
                fields=fields, arms=score(arms), untouched_unfiltered=score(sources, source=True),
                literal_rule="remove whitespace, comma, currency symbols and framing quotes; preserve leading zeros and decimal places",
                interpretation="secondary sensitivity; Decimal primary evaluates numeric-value binding, not literal identifier transcription")


def decision(analysis):
    screen = analysis["teacher_screen"]
    if screen["pending"]:
        return dict(verdict="PENDING_SCREEN", reason="teacher 4/4 verification is incomplete")
    if screen["independent_accepted_pages"] < 20:
        return dict(verdict="UNDERPOWERED", reason="fewer than 20 teacher-valid independent pages; no ADVANCE allowed")
    if analysis.get("interface_checks", {}).get("violations"):
        return dict(verdict="INADEQUATE_INTERFACE", reason="native budget/DeepStack/assignment or same-anchor checks failed")
    names = ("restricted_aggregation_same_anchors", "oracle_minus_visionzip_ppe",
             "oracle_minus_random", "oracle_minus_spatial")
    contrasts = analysis["contrasts"]
    if any(contrasts[name]["status"] != "complete" for name in names):
        return dict(verdict="PENDING_ARMS", reason="all required arms must finish on the same accepted pages")
    if analysis.get("interface_checks", {}).get("pending"):
        return dict(verdict="HOLD_INTERFACE_PENDING", reason="required native/PPE gates or complete same-anchor diagnostics are missing")
    primary = contrasts[names[0]]
    if primary["ci95"] is None:
        return dict(verdict="HOLD", reason="insufficient independent source documents for the paired interval")
    controls_win = all(contrasts[name]["delta"] > 0 for name in names[1:])
    if primary["net_repairs"] >= 3 and primary["ci95"][0] > 0 and controls_win:
        sources = analysis["source_contrasts"]["restricted_aggregation_same_anchors"]
        if sources["status"] != "complete":
            return dict(verdict="HOLD_SOURCE_QA_PENDING", reason="primary conditions met; untouched QA inspection is incomplete")
        if sources["net_repairs"] < 0:
            return dict(verdict="HOLD_SOURCE_QA_DAMAGE", reason="primary conditions met but untouched QA regressed")
        raw_sources = analysis.get("unfiltered_source_contrasts", {}).get("restricted_aggregation_same_anchors")
        if raw_sources is not None and raw_sources["status"] != "complete":
            return dict(verdict="HOLD_SOURCE_QA_PENDING", reason="primary conditions met; unfiltered untouched QA inspection is incomplete")
        if raw_sources is not None and raw_sources["net_repairs"] < 0:
            return dict(verdict="HOLD_SOURCE_QA_DAMAGE", reason="primary conditions met but unfiltered untouched QA regressed")
        return dict(verdict="ADVANCE", reason="pilot threshold met; requires independent replication and deployable-region validation")
    if primary["net_repairs"] <= 0 or not controls_win:
        return dict(verdict="STOP_THIS_MECHANISM", reason="adequately sampled oracle has no net repair or fails to beat a simple control; does not refute binding preservation generally")
    return dict(verdict="HOLD", reason="positive intermediate evidence does not meet net-repair/interval thresholds")


def _population_view(name, items, combined, draws, seed, base_directory):
    """Subset already planned rows; do not take new per-stratum audit prefixes."""
    ids = {item["pair_id"] for item in items}
    screen = combined["teacher_screen"]
    accepted_ids = ids & set(screen["accepted_pair_ids"])
    accepted = [item for item in items if item["pair_id"] in accepted_ids]
    accepted_list = [item["pair_id"] for item in accepted]
    rejected = [pair_id for pair_id in screen["rejected_pair_ids"] if pair_id in ids]
    pending = [pair_id for pair_id in screen["pending_pair_ids"] if pair_id in ids]
    subset_screen = dict(planned=len(items),
                         attempted=sum(row["available_questions"] > 0 for row in screen["records"] if row["pair_id"] in ids),
                         verified=len(accepted) + len(rejected),
                         accepted=len(accepted), rejected=len(rejected), pending=len(pending),
                         acceptance_rate=len(accepted) / (len(accepted) + len(rejected)) if accepted or rejected else None,
                         independent_accepted_pages=len({item["source_image"] for item in accepted}),
                         accepted_source_documents=len({str(item["source_document_id"]) for item in accepted}),
                         accepted_pair_ids=accepted_list, rejected_pair_ids=rejected, pending_pair_ids=pending)
    def subset_arms(payload):
        out = {}
        for arm, info in payload.items():
            rows = [row for row in info["records"] if row["pair_id"] in ids]
            out[arm] = dict(summary=summary(rows, len(rows)), records=rows)
        return out
    arms = subset_arms(combined["arms"])
    sources = subset_arms(combined["sources_same_teacher_valid_pages"])
    raw_sources = subset_arms(combined["unfiltered_source_arms"])
    images = {str(item.get(key)) for item in items for key in ("source_image", "original_image", "swapped_image")}
    violations = [v for v in combined["interface_checks"]["violations"]
                  if (("pair_id" not in v and "image" not in v)
                      or v.get("pair_id") in ids or v.get("image") in images)]
    output = dict(population=name, teacher_screen=subset_screen,
                  arms=arms, sources_same_teacher_valid_pages=sources,
                  unfiltered_source_arms=raw_sources,
                  contrasts=comparisons(arms, CONTRASTS, draws, seed),
                  source_contrasts=comparisons(sources, CONTRASTS[:6], draws, seed),
                  unfiltered_source_contrasts=comparisons(raw_sources, CONTRASTS[:6], draws, seed),
                  diagnostics=diagnostic_analysis(arms, Path(base_directory)),
                  interface_checks=dict(violations=violations,
                                        pending=[p for p in combined["interface_checks"].get("pending", [])
                                                 if (("pair_id" not in p and "image" not in p and "images" not in p)
                                                     or p.get("pair_id") in ids or p.get("image") in images
                                                     or set(p.get("images", [])) & images)]))
    output["decision"] = decision(output)
    # Full questions remain once in combined arms, with explicit dataset/stratum.
    for container in (output["arms"], output["sources_same_teacher_valid_pages"], output["unfiltered_source_arms"]):
        for info in container.values():
            info["pair_ids"] = [row["pair_id"] for row in info.pop("records")]
    for info in output["diagnostics"]["by_arm"].values():
        info.pop("records")
    return output


def analyze(predictions, manifest, draws=10000, seed=20261108, source_pages=24, score_anls=None, base_directory=DEFAULT):
    manifest_rows = manifest["records"]
    ids = [item["pair_id"] for item in manifest_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate manifest pair IDs")
    active = set(predictions.get("active_pair_ids", ids))
    # The supplied manifest declares the planned population. A queued extension
    # must stay pending before the runner activates it; never score old records
    # outside the currently active prediction IDs.
    items = list(manifest_rows)
    unknown = active - set(ids)
    if unknown:
        raise ValueError(f"active prediction IDs missing from manifest: {sorted(unknown)}")
    byid = {row["pair_id"]: row for row in predictions.get("records", []) if row["pair_id"] in active}
    source_byid = {row["pair_id"]: row for row in predictions.get("sources", []) if row["pair_id"] in active}
    geometry_version = predictions.get("geometry_version")
    geometry_cache, geometry_issues = {}, []
    geometry_args = dict(geometry_version=geometry_version or "MISSING_ACTIVE_GEOMETRY_VERSION", base_directory=base_directory,
                         geometry_cache=geometry_cache, geometry_issues=geometry_issues)
    accepted, rejected, pending = [], [], []
    teacher_rows = []
    inconsistencies = []
    for item in items:
        raw = byid.get(item["pair_id"], {})
        teacher = score_page(item, raw, "teacher", score_anls)
        teacher_rows.append(teacher)
        if not teacher["complete"]:
            pending.append(item["pair_id"])
        elif teacher["all_correct"]:
            accepted.append(item)
        else:
            rejected.append(item["pair_id"])
        if teacher["complete"] and "teacher_valid" in raw and bool(raw["teacher_valid"]) != teacher["all_correct"]:
            inconsistencies.append(dict(pair_id=item["pair_id"], reason="cached teacher_valid differs from recomputed 4/4"))
    observed = {arm for row in byid.values() for entry in row.get("members", {}).values() for arm in entry}
    scheduled = CORE_ARMS + AUDIT_ARMS + SENSITIVITY_ARMS
    arm_names = list(scheduled) + sorted(observed - set(scheduled))
    audit_ids = {item["pair_id"] for item in items[:12]}
    source_ids = {item["pair_id"] for item in items[:source_pages]}
    arms, sources, raw_sources = {}, {}, {}
    for arm in arm_names:
        expected = [item for item in accepted
                    if (arm not in AUDIT_ARMS or item["pair_id"] in audit_ids)
                    and (arm not in SENSITIVITY_ARMS or item.get("eligible_for_line_oracle", False))]
        rows = [score_page(item, byid.get(item["pair_id"], {}), arm, score_anls, **geometry_args) for item in expected]
        arms[arm] = dict(summary=summary(rows, len(expected)), records=rows)
        if arm not in AUDIT_ARMS + SENSITIVITY_ARMS:
            source_expected = [item for item in accepted if item["pair_id"] in source_ids]
            rows = [score_page(item, source_byid.get(item["pair_id"], {}), arm, score_anls, source=True,
                               **geometry_args) for item in source_expected]
            sources[arm] = dict(summary=summary(rows, len(source_expected)), records=rows)
            raw_rows = [score_page(item, source_byid.get(item["pair_id"], {}), arm, score_anls,
                                   source=True, **geometry_args) for item in items[:source_pages]]
            raw_sources[arm] = dict(summary=summary(raw_rows, len(raw_rows)), records=raw_rows)
    all_source_rows = [score_page(item, source_byid.get(item["pair_id"], {}), "teacher", score_anls, source=True)
                       for item in items[:source_pages]]
    output = dict(schema_version=2, population="combined", generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  snapshot_stage=predictions.get("stage"), seed=seed, bootstrap_draws=draws,
                  planned_pages=len(items), active_prediction_pages=len(active),
                  inactive_planned_pair_ids=[item["pair_id"] for item in items if item["pair_id"] not in active],
                  geometry=dict(active_version=geometry_version,
                                historical_invalidated_predictions=predictions.get("invalidated_compressed_predictions", 0),
                                excluded_current_entries=geometry_issues,
                                referenced_diagnostics_checked=len(geometry_cache),
                                policy="only active manifest IDs and referenced diagnostics matching the active geometry version; teacher/rres do not use field geometry"),
                  planned_subsets=dict(audit_first12_pair_ids=[item["pair_id"] for item in items[:12]],
                                       untouched_source_pair_ids=[item["pair_id"] for item in items[:source_pages]]),
                  teacher_screen=dict(planned=len(items), attempted=sum(row["available_questions"] > 0 for row in teacher_rows),
                                      verified=len(accepted) + len(rejected),
                                      accepted=len(accepted), rejected=len(rejected), pending=len(pending),
                                      acceptance_rate=len(accepted) / (len(accepted) + len(rejected)) if accepted or rejected else None,
                                      independent_accepted_pages=len({item["source_image"] for item in accepted}),
                                      accepted_source_documents=len({str(item["source_document_id"]) for item in accepted}),
                                      accepted_pair_ids=[item["pair_id"] for item in accepted],
                                      rejected_pair_ids=rejected, pending_pair_ids=pending,
                                      cached_flag_inconsistencies=inconsistencies, records=teacher_rows),
                  arms=arms, sources_same_teacher_valid_pages=sources,
                  unfiltered_source_arms=raw_sources,
                  unfiltered_source_teacher=dict(summary=summary(all_source_rows, len(items[:source_pages])), records=all_source_rows),
                  contrasts=comparisons(arms, CONTRASTS, draws, seed),
                  source_contrasts=comparisons(sources, CONTRASTS[:6], draws, seed),
                  unfiltered_source_contrasts=comparisons(raw_sources, CONTRASTS[:6], draws, seed),
                  diagnostics=diagnostic_analysis(arms, Path(base_directory)),
                  limitations=["QA-grounded value boxes and OCR lines are not gold semantic fields",
                               "same decoder K is not equal total computation; Rres512 also reduces encoder tokens",
                               "same-anchor restriction can change PPE position representatives, so it is a joint oracle",
                               "VAL and fresh TRAIN are reported separately; their combined endpoint is a mixed-population pilot",
                               "Decimal primary preserves numeric value rather than identifier characters; leading zeros and decimal formatting are checked only in a separate descriptive sensitivity",
                               "teacher rejection/pending records are retained; missing arms are not scored as zero",
                               "a null or tied result only limits this tested mechanism and cannot refute binding preservation generally"])
    violations = []
    interface_pending = []
    if geometry_version is None:
        interface_pending.append(dict(reason="missing active geometry version"))
    inspected = 0
    for endpoint, container in (("edited_pair", arms), ("untouched_source", raw_sources)):
        for arm, payload in container.items():
            expected_tokens = 1024 if arm == "teacher" else 256
            for row in payload["records"]:
                for key, question in row["questions"].items():
                    inspected += 1
                    context = dict(endpoint=endpoint, arm=arm, pair_id=row["pair_id"], question=key)
                    if question["n_vis"] is None or question["ds_lengths"] is None:
                        interface_pending.append(dict(**context, reason="missing token/DeepStack audit metadata"))
                    if question["n_vis"] is not None and question["n_vis"] != expected_tokens:
                        violations.append(dict(**context, reason="wrong visual-token count"))
                    if question["ds_lengths"] is not None and question["ds_lengths"] != [expected_tokens] * 3:
                        violations.append(dict(**context, reason="wrong DeepStack lengths"))
                    positions, start = question["decode_positions"], question["expected_decode_start"]
                    if positions is None or start is None:
                        interface_pending.append(dict(**context, reason="missing native decode-position metadata"))
                    elif positions != list(range(start, start + len(positions))):
                        violations.append(dict(**context, reason="decode positions differ from native continuation"))
    diagnostics = output["diagnostics"]
    for check in diagnostics["same_anchor_checks"]:
        if not check["complete"]:
            interface_pending.append(dict(image=check["image"], reason="missing primary anchor/dominant lists"))
        elif not check["anchors_equal"] or not check["dominant_equal"]:
            violations.append(dict(image=check["image"], reason="primary contrast has different anchors/dominants"))
    expected_anchor_images = {str(item[key]) for item in accepted for key in ("original_image", "swapped_image")}
    matched_anchor_images = {str(row["image"]) for row in diagnostics["same_anchor_checks"] if row["complete"]}
    missing_anchor_images = sorted(expected_anchor_images - matched_anchor_images)
    if missing_anchor_images:
        interface_pending.append(dict(reason="primary same-anchor diagnostics pending", images=missing_anchor_images))
    for arm, payload in diagnostics["by_arm"].items():
        for row in payload["records"]:
            if row["unassigned_count"] not in (None, 0) or row["all_tokens_accounted_once"] is False:
                violations.append(dict(arm=arm, image=row["image"], reason="source tokens were lost"))
            if arm == PRIMARY and row["cross_label_assignees"] not in (None, 0):
                violations.append(dict(arm=arm, image=row["image"], reason="field oracle has cross-label members"))
    gate_path = Path(base_directory) / "gpu_gates.json"
    gates = json.loads(gate_path.read_text()) if gate_path.exists() else None
    integration_path = Path(base_directory) / "integration_gates.json"
    integration = json.loads(integration_path.read_text()) if integration_path.exists() else None
    if gates is None or gates.get("pass_all", gates.get("all_passed")) is not True:
        interface_pending.append(dict(reason="native PPE identity gate missing or not explicitly passed"))
    if gates is not None and gates.get("pass_all", gates.get("all_passed")) is False:
        violations.append(dict(reason="recorded native GPU interface gate failed"))
    if integration is None or integration.get("pass_all") is not True:
        interface_pending.append(dict(reason="actual K256 integration gate missing or not explicitly passed"))
    if integration is not None and integration.get("pass_all") is False:
        violations.append(dict(reason="recorded K256 integration gate failed"))
    if integration is not None and integration.get("pass_all") is True:
        for name in ("visionzip", "field_anchor_control", "field_oracle"):
            check = integration.get("checks", {}).get(name, {})
            if not (check.get("passed") is True and check.get("n_vis") == 256
                    and check.get("ds_lengths") == [256] * 3):
                interface_pending.append(dict(reason="incomplete K256 selector integration metadata", selector=name))
    output["interface_checks"] = dict(inspected_question_records=inspected, violations=violations,
                                       pending=interface_pending,
                                       native_gate_file=str(gate_path) if gates is not None else None,
                                       native_gate_payload=gates,
                                       integration_gate_file=str(integration_path) if integration is not None else None,
                                       integration_gate_payload=integration,
                                       expected_same_anchor_images=len(expected_anchor_images),
                                       same_anchor_images_checked=len(diagnostics["same_anchor_checks"]))
    output["decision"] = decision(output)
    output["literal_identifier_sensitivity"] = literal_identifier_sensitivity(items, arms, raw_sources)
    output["combined"] = dict(teacher_screen={k: v for k, v in output["teacher_screen"].items() if k != "records"},
                              arms={arm: payload["summary"] for arm, payload in arms.items()},
                              contrasts=output["contrasts"], source_contrasts=output["source_contrasts"],
                              unfiltered_source_arms={arm: payload["summary"] for arm, payload in raw_sources.items()},
                              unfiltered_source_contrasts=output["unfiltered_source_contrasts"],
                              decision=output["decision"])
    groups = defaultdict(list)
    for item in items:
        groups[dataset_stratum(item)].append(item)
    output["strata"] = {name: _population_view(name, group, output, draws, seed, base_directory)
                        for name, group in sorted(groups.items())}
    return output


def _ratio(n, d):
    return f"{n}/{d}" if d else "pending"


def markdown_report(data):
    screen = data["teacher_screen"]
    lines = ["# VisionZip / PPE binding oracle analysis", "", f"Decision: **{data['decision']['verdict']}**. {data['decision']['reason']}", "",
             f"Active geometry: `{data['geometry']['active_version']}`. Historical invalidated answers: {data['geometry']['historical_invalidated_predictions']}; currently excluded entries with incompatible/missing referenced diagnostics: {len(data['geometry']['excluded_current_entries'])}.", "",
             f"Teacher 4/4 verification: {screen['accepted']} accepted, {screen['rejected']} rejected, {screen['pending']} pending / {screen['planned']} planned pages; {screen['attempted']} pages attempted. Accepted pages span {screen['accepted_source_documents']} source documents.", "",
             "Missing predictions remain pending. Primary correctness requires all four field questions on both image members.", "",
             "| Arm | Status | 4/4 pages | Individual answers | Any side swaps two recognized values | Four-answer multiset complete | Untouched 2/2 pages |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, payload in data["arms"].items():
        s = payload["summary"]
        source = data["sources_same_teacher_valid_pages"].get(arm, {}).get("summary", {})
        lines.append(f"| {arm} | {s['status']} ({s['complete_pages']}/{s['expected_pages']}) | {_ratio(s['all_correct'],s['complete_pages'])} | {_ratio(s['individual_correct'],s['individual_questions'])} | {s['any_side_assignment_swap'] if s['complete_pages'] else 'pending'} | {s['answer_multiset_exact'] if s['complete_pages'] else 'pending'} | {_ratio(source.get('all_correct',0),source.get('complete_pages',0))} |")
    lines += ["", "| Paired contrast | Complete matched pages | Rescued / broken | Net repairs | Page accuracy difference | Document bootstrap 95% CI |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, c in data["contrasts"].items():
        delta = "pending" if c["delta"] is None else f"{100*c['delta']:+.1f} pp"
        interval = "pending" if c["ci95"] is None else f"[{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}] pp"
        repairs = f"{c['rescued']} / {c['broken']}" if c["n_pages"] else "pending"
        net = c["net_repairs"] if c["n_pages"] else "pending"
        lines.append(f"| {name} ({c['status']}) | {c['n_pages']}/{c['expected_pages']} | {repairs} | {net} | {delta} | {interval} |")
    question = data["contrasts"]["restricted_aggregation_same_anchors"]["question_endpoint"]
    qdelta = "pending" if question["delta"] is None else f"{100*question['delta']:+.1f} pp"
    qinterval = "pending" if question["ci95"] is None else f"[{100*question['ci95'][0]:+.1f}, {100*question['ci95'][1]:+.1f}] pp"
    lines += ["", f"Secondary identical-anchor question endpoint: {question['n_questions']} matched questions on {question['n_pages']} pages, {question['rescued']} repaired / {question['broken']} broken; exact question accuracy difference {qdelta}, document bootstrap 95% CI {qinterval}. This records partial-answer gains and does not replace the registered all-four decision."]
    lines += ["", "The untouched source panel keeps the globally planned first-page prefix. Its unfiltered results include pages rejected by the edited-pair teacher; the accepted-subset endpoint above uses the same teacher-valid pair IDs as the primary experiment.", "",
              "| Untouched source arm, unfiltered planned prefix | Status | 2/2 pages | Exact individual answers |",
              "| --- | --- | ---: | ---: |"]
    for arm, payload in data["unfiltered_source_arms"].items():
        s = payload["summary"]
        lines.append(f"| {arm} | {s['status']} ({s['complete_pages']}/{s['expected_pages']}) | {_ratio(s['all_correct'],s['complete_pages'])} | {_ratio(s['individual_correct'],s['individual_questions'])} |")
    for name, population in data["strata"].items():
        s = population["teacher_screen"]
        c = population["contrasts"]["restricted_aggregation_same_anchors"]
        interval = "pending" if c["ci95"] is None else f"[{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}] pp"
        lines += ["", f"## {name} source stratum", "",
                  f"Teacher: {s['accepted']} accepted, {s['rejected']} rejected, {s['pending']} pending / {s['planned']} planned; {s['attempted']} attempted. Accepted pages span {s['accepted_source_documents']} source documents. Decision: {population['decision']['verdict']}.", "",
                  "| Arm | Complete pages / expected | 4/4 correct | Exact individual answers | Untouched 2/2, unfiltered prefix |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for arm, payload in population["arms"].items():
            a = payload["summary"]
            source = population["unfiltered_source_arms"].get(arm, {}).get("summary", {})
            lines.append(f"| {arm} | {a['complete_pages']}/{a['expected_pages']} | {_ratio(a['all_correct'],a['complete_pages'])} | {_ratio(a['individual_correct'],a['individual_questions'])} | {_ratio(source.get('all_correct',0),source.get('complete_pages',0))} |")
        lines += ["", f"Primary identical-anchor contrast ({c['status']}): {c['n_pages']}/{c['expected_pages']} matched pages; net repairs {c['net_repairs'] if c['n_pages'] else 'pending'}; document bootstrap 95% CI {interval}."]
        q = c["question_endpoint"]
        qdelta = "pending" if q["delta"] is None else f"{100*q['delta']:+.1f} pp"
        qinterval = "pending" if q["ci95"] is None else f"[{100*q['ci95'][0]:+.1f}, {100*q['ci95'][1]:+.1f}] pp"
        lines += ["", f"Secondary identical-anchor exact-question difference: {qdelta}; document bootstrap 95% CI {qinterval}, {q['n_questions']} matched questions."]
    lines += ["", "Anchor allocation is field_anchor_control_ppe minus visionzip_ppe. Restricted aggregation is field_oracle_ppe minus its identical-anchor unrestricted control. These effects must not be combined into a claim about aggregation alone.", "",
              "PPE target-position coverage is reported per image, distinguishing position slots from unique original token coordinates:", "",
              "| Arm | Images with diagnostics | Both value regions represented in PPE | Missing a value region | Anchor replacements |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm, d in data["diagnostics"]["by_arm"].items():
        if d["n_images"]:
            lines.append(f"| {arm} | {d['n_images']} | {d['both_fields_have_ppe_position']} | {d['missing_field_position_images']} | {d['total_anchor_replacements']} |")
    literal = data["literal_identifier_sensitivity"]
    lines += ["", f"The manifest has {literal['planned_leading_zero_fields']} leading-zero fields on {literal['planned_leading_zero_pairs']} pairs. Primary Decimal exactness evaluates numeric value; it does not establish character-exact identifier preservation. A separate literal check removes currency/comma/space but preserves leading zeros and decimal places:", "",
              "| Leading-zero target sensitivity | Available / expected questions | Decimal value correct | Literal correct | Extra primary credits |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for arm, payload in literal["arms"].items():
        if payload["available_questions"]:
            lines.append(f"| {arm} | {payload['available_questions']}/{payload['expected_questions']} | {payload['primary_value_correct']} | {payload['literal_correct']} | {payload['primary_credit_without_literal_match']} |")
    lines += ["", "The numeric-set endpoints check recognition without proving correct field assignment. PPE representatives may change when clusters change; a positive oracle supports joint content/position preservation. A content-only claim requires an additional crossed-position control.", "",
              f"Interface audit: {len(data['interface_checks']['violations'])} violations, {len(data['interface_checks']['pending'])} pending requirements. Missing gates/diagnostics cannot produce ADVANCE.", "",
              f"Intervals use {data['bootstrap_draws']:,} source-document bootstrap draws, seed {data['seed']}. Degenerate zero intervals do not establish algorithmic equivalence. Fewer than 20 accepted independent pages cannot produce ADVANCE.", ""]
    lines += [f"- {value}" for value in data["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT / "predictions.json"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out")
    parser.add_argument("--report")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--source-pages", type=int, default=24)
    parser.add_argument("--include-anls", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 1 or args.source_pages < 0:
        parser.error("bootstrap must be positive and source-pages nonnegative")
    prediction_path = Path(args.predictions)
    scorer, scoring_metadata = anls_scorer() if args.include_anls else (None, None)
    prediction_text, manifest_text = prediction_path.read_text(), Path(args.manifest).read_text()
    result = analyze(json.loads(prediction_text), json.loads(manifest_text),
                     args.bootstrap, args.seed, args.source_pages, scorer, prediction_path.parent)
    result.update(inputs=dict(predictions=str(prediction_path.resolve()), manifest=str(Path(args.manifest).resolve()),
                              predictions_sha256=hashlib.sha256(prediction_text.encode()).hexdigest(),
                              manifest_sha256=hashlib.sha256(manifest_text.encode()).hexdigest()),
                  optional_anls=scoring_metadata)
    out = Path(args.out) if args.out else prediction_path.parent / "analysis.json"
    report = Path(args.report) if args.report else prediction_path.parent / "analysis.md"
    atomic_json(out, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(result))
    screen_keys = ("planned", "attempted", "verified", "accepted", "rejected", "pending", "accepted_source_documents")
    print(json.dumps(dict(decision=result["decision"],
                          teacher_screen={k: result["teacher_screen"][k] for k in screen_keys},
                          strata={name: {k: value["teacher_screen"][k] for k in screen_keys}
                                  for name, value in result["strata"].items()},
                          primary_question=result["contrasts"]["restricted_aggregation_same_anchors"]["question_endpoint"],
                          analysis=str(out), report=str(report)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
