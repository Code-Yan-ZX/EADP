"""Independent CPU-only structural/provenance audit of the complete pilot.

No model is loaded, no output is changed, and no page is excluded. The
coverage summaries compare routing indices, not answer correctness.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from PIL import Image

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
OUT = QWEN / "outputs/catchup_rerank"
ROUTED = ("baseline", "dense_full", "dense_2k", "ltc_2k")
ARMS = ("full", "rres512") + ROUTED


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, check_name, ok, **context):
        self.rows.append(dict(name=check_name, passed=bool(ok), **context))

    def indices(self, values, name, n, size=None, **context):
        valid = (isinstance(values, list)
                 and all(type(x) is int and 0 <= x < n for x in values)
                 and values == sorted(set(values)))
        self.check(name, valid and (size is None or len(values) == size),
                   expected_size=size, observed_size=len(values) if isinstance(values, list) else None,
                   **context)
        return set(values) if valid else set()


def source_audit(manifest, checks):
    provenance = manifest["provenance"]
    source_path = Path(provenance["source_manifest"])
    checks.check("source_manifest_hash", digest(source_path) == provenance["source_manifest_sha256"])
    source = json.loads(source_path.read_text())
    reserves = [x for x in source["items"] if x["role"] == "reserve"]
    items = manifest["items"]
    checks.check("source_reserve_count", len(reserves) == len(items) == 16)
    checks.check("manifest_items_records_equal", manifest.get("records") == items)
    checks.check("manifest_complete", manifest.get("complete") is True)
    parents = [x["parent_id"] for x in items]
    checks.check("independent_16_parents", len(set(parents)) == len(parents) == 16)
    checks.check("unique_16_images", len({x["image_sha256"] for x in items}) == 16)
    excluded = set(source["provenance"]["excluded_parent_documents"])
    checks.check("previous_parent_exclusions", not (set(parents) & excluded))
    old_path = Path(provenance["source_predictions_audited"])
    checks.check("previous_predictions_hash", digest(old_path) == provenance["source_predictions_sha256"])
    old_rows = [json.loads(line) for line in old_path.read_text().splitlines() if line]
    checks.check("no_previous_ocr_panel_parent_overlap",
                 not (set(parents) & {x["parent_id"] for x in old_rows}))
    parquet_hashes = {}
    for rank, (item, reserve) in enumerate(zip(items, reserves)):
        qa = reserve["qa"][0]
        ident = item["item_id"]
        checks.check("source_order_and_first_qa",
                     item["parent_id"] == reserve["parent_id"]
                     and item["doc_id"] == reserve["doc_id"]
                     and item["selection_rank"] == rank
                     and item["source_role"] == "reserve"
                     and item["selected_qa_index"] == 0
                     and item["selected_source_qa"] == qa
                     and item["question"] == qa["question"]
                     and item["question_id"] == qa["question_id"]
                     and item["answers"] == qa["answers"], item_id=ident)
        image = Path(item["image_path"])
        checks.check("source_image_hash", digest(image) == item["image_sha256"]
                     == item["source_sha256"] == reserve["source_sha256"], item_id=ident)
        with Image.open(image) as picture:
            checks.check("source_image_dimensions", list(picture.size) == item["source_image_size"], item_id=ident)
        path = item["source_parquet"]
        if path not in parquet_hashes:
            parquet_hashes[path] = digest(path)
        checks.check("source_parquet_hash", parquet_hashes[path] == item["source_parquet_sha256"], item_id=ident)
    return dict(source_manifest=str(source_path), source_parquets=len(parquet_hashes),
                no_outcome_selection=True, retained_all_16=True)


def leakage_review(checks):
    """Re-execute only the simple prompt function with sentinel annotations.

    This is a source-level data-flow check plus frozen source hashes, not
    proof about hidden external dependencies or pretrained-model exposure.
    """
    tree = ast.parse((HERE / "run.py").read_text())
    message = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "message")
    isolated = ast.fix_missing_locations(ast.Module(body=[message], type_ignores=[]))
    namespace = {}
    exec(compile(isolated, str(HERE / "run.py"), "exec"), namespace)
    sentinel = dict(image_path="IMAGE_SENTINEL", question="QUESTION_SENTINEL",
                    answers=["SECRET_GOLD_SENTINEL"], preferred_answer="SECRET_GOLD_SENTINEL",
                    selected_source_qa={"value_bbox": [11, 22, 33, 44],
                                        "provided_ocr_text": "SECRET_OCR_SENTINEL"})
    actual = namespace["message"](sentinel)
    expected = [dict(type="image", value="IMAGE_SENTINEL"),
                dict(type="text", value="QUESTION_SENTINEL\nAnswer with the value only.")]
    checks.check("prompt_image_and_question_only", actual == expected)
    sensitive = {"answers", "preferred_answer", "selected_source_qa", "value_bbox",
                 "value_polygon", "provided_ocr_text", "context_line_text"}
    route_tree = ast.parse((HERE / "route.py").read_text())
    accesses = sorted({node.value for node in ast.walk(route_tree)
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)
                       and node.value in sensitive})
    checks.check("routing_has_no_qa_answer_or_annotation_field_access", not accesses,
                 sensitive_field_accesses=accesses)
    return dict(scope="Frozen prompt/routing source and complete fixed sample provenance; no labels or silver boxes enter prompt or router.",
                limitation="Does not establish absence of pretrained-model dataset exposure; does not prove arbitrary external dependencies label-free.")


def auxiliary_timing_amendment(frozen, artifact_dir, checks):
    """Keep the QA freeze strict; separately verify the disclosed timing fix."""
    metadata_path = artifact_dir / "latency_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    checks.check("auxiliary_benchmark_original_hash_disclosed",
                 metadata["benchmark_script_sha256_at_freeze"]
                 == frozen["script_sha256"]["benchmark.py"])
    checks.check("auxiliary_benchmark_executed_hash_matches",
                 metadata["benchmark_script_sha256_executed"] == digest(HERE / "benchmark.py"))
    failed = artifact_dir / metadata["failed_attempt_log"]
    completed = artifact_dir / metadata["completed_attempt_log"]
    results = artifact_dir / metadata["results"]
    failure_text = failed.read_text()
    checks.check("auxiliary_import_failure_preserved",
                 "ImportError: cannot import name 'resolution' from 'run'" in failure_text
                 and "from run import resolution, predict_native" in failure_text)
    checks.check("auxiliary_completed_log_and_change_description",
                 completed.is_file() and completed.stat().st_size > 0
                 and bool(metadata.get("change")))
    measurements = json.loads(results.read_text())
    expected = {(r, a) for r in range(3) for a in ("baseline", "ltc_2k", "rres512")}
    checks.check("auxiliary_timing_complete_nine_measurements",
                 len(measurements["rows"]) == 9
                 and {(r["repeat"], r["arm"]) for r in measurements["rows"]} == expected
                 and measurements["warmups_per_arm"] == 1
                 and all(r["max_new_tokens"] == 1 and math.isfinite(r["total_ms"])
                         and r["total_ms"] > 0 for r in measurements["rows"]))
    return dict(status="DISCLOSED_POSTFREEZE_AUXILIARY_TIMING_CHANGE",
                original_freeze_unchanged=True, affects_qa_runner=False,
                explanation=metadata["change"],
                script_sha256_at_freeze=metadata["benchmark_script_sha256_at_freeze"],
                script_sha256_executed=metadata["benchmark_script_sha256_executed"],
                metadata_sha256=digest(metadata_path), failed_log_sha256=digest(failed),
                completed_log_sha256=digest(completed), result_sha256=digest(results),
                limitation="Auxiliary latency run is not an exact execution of the originally frozen benchmark.py. QA runner, routing and data hashes remain strictly checked.")


def audit(predictions, manifest, frozen, gate, checks):
    cfg = predictions["config"]
    rows = predictions["records"]
    items = manifest["items"]
    checks.check("formal_output_complete", predictions.get("complete") is True)
    checks.check("all_16_records", len(rows) == cfg.get("sample_count") == len(items) == 16)
    checks.check("record_order_and_no_duplicates", [r["item_id"] for r in rows] == [x["item_id"] for x in items]
                 and len({r["item_id"] for r in rows}) == len(rows))
    checks.check("record_unique_parents", len({r["parent_id"] for r in rows}) == 16)
    checks.check("configuration", cfg["K"] == 256 and cfg["route_layers"] == [2, 7]
                 and cfg["max_tokens"] == 24 and cfg["seed"] == 20261108
                 and cfg["arms"] == list(ROUTED))
    checks.check("frozen_setting", frozen["input_visual_tokens"] == 1024
                 and frozen["active_tokens"] == 256 and frozen["candidate_count"] == 512
                 and frozen["route_layers_zero_based"] == [2, 7])
    checks.check("native_gates_passed", gate.get("pass_all") is True)
    checks.check("gate_route_hash", gate["route_sha256"] == digest(HERE / "route.py"))
    for name, expected in frozen["script_sha256"].items():
        if name == "benchmark.py":
            # This auxiliary file was repaired after an import-only failed
            # run. Its old/new hashes and preserved logs are checked under
            # an explicit amendment, never substituted into the QA freeze.
            continue
        checks.check("frozen_script_hash", digest(HERE / name) == expected, script=name)
    case_rows = []
    for row, item in zip(rows, items):
        ident = row["item_id"]
        ctx = dict(item_id=ident)
        checks.check("record_matches_source", all(row[k] == item[k] for k in
                     ("item_id", "parent_id", "question", "answers", "image_sha256")), **ctx)
        preds = row["predictions"]
        checks.check("all_six_arms_present", set(preds) == set(ARMS), **ctx)
        for arm in ARMS:
            p = preds[arm]
            checks.check("prediction_complete", isinstance(p.get("text"), str)
                         and type(p.get("correct")) is bool and type(p.get("truncated")) is bool
                         and isinstance(p.get("token_ids"), list)
                         and all(type(t) is int and t >= 0 for t in p["token_ids"])
                         and len(p["token_ids"]) <= cfg["max_tokens"], arm=arm, **ctx)
            checks.check("truncation_cap_consistent", p["truncated"] == (len(p["token_ids"]) >= cfg["max_tokens"]), arm=arm, **ctx)
        for arm, n in (("full", 1024), ("rres512", 256)):
            p = preds[arm]
            checks.check("native_reference_budget_and_ds", p["n_vis"] == n
                         and p["ds_lengths"] == [n, n, n], arm=arm, **ctx)
        base = preds["baseline"]["diag"]
        n, k = 1024, 256
        full_length = base["prefill_cache_lengths"][0]
        expected_cache = [full_length if i < 2 else full_length - n + k for i in range(36)]
        expected_prepared = expected_cache[:7] + [0] * 29
        first, base_keep, shortlist = (set(base[x]) for x in ("first_keep", "baseline_second_keep", "shortlist"))
        anchors = first & base_keep
        deferred = set(range(n)) - first
        per_arm = {}
        for arm in ROUTED:
            p, d = preds[arm], preds[arm]["diag"]
            actx = dict(arm=arm, **ctx)
            sets = {}
            for name, size in (("first_keep", k), ("second_keep", k), ("baseline_second_keep", k),
                               ("shortlist", 2*k), ("common_anchors", None),
                               ("rerank_targets", None), ("reentered", None)):
                sets[name] = checks.indices(d.get(name), "valid_sorted_unique_" + name, n, size, **actx)
            checks.check("common_prefix_anchor_shortlist", all(d[x] == base[x] for x in
                         ("first_keep", "baseline_second_keep", "shortlist", "common_anchors"))
                         and sets["common_anchors"] == anchors, **actx)
            checks.check("fixed_native_configuration", d["variant"] == arm and d["n_vis"] == n
                         and d["K"] == k and d["route_layers"] == [2, 7]
                         and d["route_layers_zero_based"] is True
                         and d["official_four_stage_reproduction"] is False, **actx)
            selected = sets["second_keep"]
            reentered = selected - first
            checks.check("readmission_count_and_membership", sets["reentered"] == reentered
                         and d["corrected_count"] == (len(reentered) if anchors else 0)
                         and d["execution_ltc"] is True
                         and d["empty_anchor_correction_skipped"] == bool(reentered and not anchors), **actx)
            changes = k - len(selected & base_keep)
            checks.check("selection_change_count", d["selection_changed"] == changes, **actx)
            expected_targets = (set() if arm == "baseline" else
                                deferred if arm.startswith("dense") else shortlist & deferred)
            checks.check("key_replacement_target_membership", sets["rerank_targets"] == expected_targets, **actx)
            if arm in ("dense_2k", "ltc_2k"):
                checks.check("selection_within_stale_shortlist", selected <= shortlist, **actx)
            if arm == "baseline":
                checks.check("baseline_uses_stale_selection", selected == base_keep, **actx)
            checks.check("declared_key_only_intervention", all(d[x] is True for x in
                         ("deferred_key_replacements_only", "nonvisual_keys_unchanged", "query_unchanged",
                          "softmax_over_all_prompt_keys", "all_original_text_positions_preserved"))
                         and d["dense_reference_is_execution_state"] is False, **actx)
            checks.check("dense_denominators_matched", d["dense_2k_uses_identical_dense_full_scores"] == arm.startswith("dense"), **actx)
            checks.check("same_ltc_operator", d["ltc"] == dict(topm=8, spatial_lambda=.15,
                         tau=.07, gamma=.8, norm_preserve=False), **actx)
            checks.check("all_prefill_cache_lengths", d["prefill_cache_lengths"] == expected_cache, **actx)
            feed_count = len(p["token_ids"]) - int(d["stop_reason"] == "max_new_tokens")
            checks.check("eos_and_decode_counts", d["stop_reason"] in ("eos", "max_new_tokens")
                         and feed_count >= 0 and len(d["decode_positions"]) == feed_count
                         and d["generated_tokens"] == len(p["token_ids"])
                         and d["truncated"] == p["truncated"]
                         and (d["stop_reason"] == "max_new_tokens") == p["truncated"], **actx)
            # For one 32x32 visual grid, visual RoPE consumes 32 positions,
            # while all 1024 image IDs consume cache slots. Text is unchanged.
            decode_start = full_length - n + 32
            checks.check("native_decode_positions", d["decode_start"] == decode_start
                         and d["decode_positions"] == list(range(decode_start, decode_start + feed_count)), **actx)
            checks.check("all_final_cache_lengths", d["final_cache_lengths"] == [x + feed_count for x in expected_cache], **actx)
            checks.check("preparation_cache_immutable", d["prepared_cache_lengths_before"]
                         == d["prepared_cache_lengths_after"] == expected_prepared, **actx)
            checks.check("actual_three_deepstack_lengths", d["native_ds_original_lengths"] == [n]*3
                         and d["ds_injection_lengths"] == [n, n, k], **actx)
            checks.check("finite_nonnegative_timing", all(isinstance(v, (int, float))
                         and math.isfinite(v) and v >= 0 for v in p["timings"].values()), **actx)
            per_arm[arm] = dict(reentered=len(reentered), selection_changed=changes,
                               shortlist_overlap=len(selected & shortlist), anchors=len(anchors))
        dense = set(preds["dense_full"]["diag"]["second_keep"])
        dense_deferred = dense & deferred
        case_rows.append(dict(item_id=ident, parent_id=row["parent_id"],
                              dense_topk_in_shortlist=len(dense & shortlist), dense_topk_count=k,
                              dense_topk_shortlist_coverage=len(dense & shortlist)/k,
                              dense_deferred_in_shortlist=len(dense_deferred & shortlist),
                              dense_deferred_count=len(dense_deferred),
                              dense_deferred_shortlist_recall=(len(dense_deferred & shortlist)/len(dense_deferred)
                                                              if dense_deferred else None),
                              arms=per_arm))
    return case_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=OUT / "predictions.json")
    p.add_argument("--manifest", type=Path, default=OUT / "data/manifest.json")
    p.add_argument("--frozen", type=Path, default=OUT / "run_manifest.json")
    p.add_argument("--gates", type=Path, default=OUT / "gates.json")
    p.add_argument("--output", type=Path, default=OUT / "audit_results.json")
    args = p.parse_args()
    checks = Checks()
    predictions = json.loads(args.predictions.read_text())
    if predictions.get("complete") is not True:
        print(json.dumps(dict(status="PENDING", reason="Formal predictions are incomplete; no audit output written.")))
        return 2
    try:
        manifest = json.loads(args.manifest.read_text())
        frozen = json.loads(args.frozen.read_text())
        gate = json.loads(args.gates.read_text())
        checks.check("frozen_manifest_hash", digest(args.manifest) == frozen["manifest_sha256"]
                     == predictions["config"]["manifest_sha256"])
        checks.check("frozen_gate_hash", digest(args.gates) == frozen["gate_sha256"])
        source = source_audit(manifest, checks)
        leakage = leakage_review(checks)
        cases = audit(predictions, manifest, frozen, gate, checks)
        amendment = auxiliary_timing_amendment(frozen, args.predictions.parent, checks)
        denom = sum(r["dense_deferred_count"] for r in cases)
        summary = dict(
            n_documents=len(cases),
            dense_topk_shortlist_coverage_macro=mean(r["dense_topk_shortlist_coverage"] for r in cases),
            dense_topk_shortlist_coverage_micro=sum(r["dense_topk_in_shortlist"] for r in cases)/(256*len(cases)),
            dense_deferred_shortlist_recall_macro=mean(r["dense_deferred_shortlist_recall"] for r in cases
                                                     if r["dense_deferred_shortlist_recall"] is not None),
            dense_deferred_shortlist_recall_micro=sum(r["dense_deferred_in_shortlist"] for r in cases)/denom if denom else None,
            dense_deferred_denominator=denom,
            arms={arm: {key: dict(mean=mean(r["arms"][arm][key] for r in cases),
                                  minimum=min(r["arms"][arm][key] for r in cases),
                                  maximum=max(r["arms"][arm][key] for r in cases),
                                  total=sum(r["arms"][arm][key] for r in cases))
                        for key in ("reentered", "selection_changed", "shortlist_overlap", "anchors")}
                  for arm in ROUTED})
        result = dict(status="PASS" if all(c["passed"] for c in checks.rows) else "FAIL",
                      checks=checks.rows, n_checks=len(checks.rows), source_audit=source,
                      qa_and_routing_frozen_hashes="STRICTLY_CHECKED",
                      auxiliary_timing_amendment=amendment,
                      leakage_review=leakage, routing_summary=summary, routing_per_document=cases,
                      limitation="Structural/hash/source audit and observed routing coverage; does not establish causal optimality, QA gains or benchmark generalization.")
    except Exception as exc:
        result = dict(status="FAIL_INPUT", error_type=type(exc).__name__, error=str(exc),
                      n_checks=len(checks.rows), checks=checks.rows)
    result["artifact_sha256"] = dict(predictions=digest(args.predictions), manifest=digest(args.manifest),
                                      frozen=digest(args.frozen), gates=digest(args.gates), audit=digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(status=result["status"], n_checks=result["n_checks"],
                          failures=[r for r in checks.rows if not r["passed"]],
                          error=result.get("error"), output=str(args.output)), indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
