#!/usr/bin/env python3
"""Audit frozen cross-dataset outputs and retain the full question denominator."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.student_broad"

from .metrics import aggregate_metrics, metric_name, parse_prediction, score_item, source_metadata


DEFAULT_ARMS = ["full", "b1", "b2", "rres512"] + [f"bag_rank__seed_{seed}" for seed in (11, 23, 37)]


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _parent(item):
    for field in ("parent_id", "page_id", "image_sha256", "pixel_sha256", "image_id", "item_id"):
        if item.get(field) is not None:
            return str(item[field])
    raise ValueError("item lacks an identity")


def _cluster(item):
    """Group repeated image pixels even when source parent/category differs.

    Manifest parent identity and native MME question-pair groups are unchanged.
    Image-free controls have no pixel identity and fall back to source parent.
    """
    if item.get("has_image", True):
        for field in ("source_pixel_sha256", "pixel_sha256"):
            if item.get(field):
                return "pixels:" + str(item[field])
    return "parent:" + _parent(item)


def paired_bootstrap(reference, candidate, parents, draws=10000, seed=20261112):
    """Question-weighted difference, resampling supplied independent clusters."""
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or len(parents) != len(a):
        raise ValueError("bootstrap requires equal nonempty paired arrays")
    if draws < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid bootstrap data")
    names, inverse = np.unique(np.asarray(parents, dtype=str), return_inverse=True)
    counts, sums = np.bincount(inverse), np.bincount(inverse, weights=b - a)
    rng, samples = np.random.default_rng(seed), np.empty(draws)
    for start in range(0, draws, 128):
        end = min(start + 128, draws)
        indices = rng.integers(0, len(names), size=(end - start, len(names)))
        samples[start:end] = sums[indices].sum(1) / counts[indices].sum(1)
    return {"delta": float(np.mean(b - a)),
            "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
            "n_questions": len(a), "n_independent_clusters": len(names), "bootstrap_draws": draws,
            "improved": int(np.sum(b > a)), "worsened": int(np.sum(b < a)),
            "tied": int(np.sum(b == a))}


def _load_database(path):
    if not path.is_file():
        return [], [], []
    connection = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("prediction database failed SQLite quick_check")
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def records(table):
            result = []
            if table not in tables:
                return result
            for row in connection.execute(f'SELECT * FROM "{table}"'):
                outer = dict(row)
                payload = json.loads(outer.pop("payload"))
                if not isinstance(payload, dict):
                    raise ValueError("output payload must be a JSON object")
                for field, value in outer.items():
                    if field in payload and field != "id" and payload[field] != value:
                        raise ValueError(f"SQLite key disagrees with {field} in payload")
                    payload.setdefault(field, value)
                result.append(payload)
            return result

        timing_table = "timings" if "timings" in tables else "timing"
        return records("predictions"), records(timing_table), records("events")
    finally:
        connection.close()


def _verify_freeze(run_dir):
    path = run_dir / "freeze.json"
    if not path.is_file():
        return None, {"status": "missing"}
    frozen = json.loads(path.read_text())
    identity = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if frozen.get("freeze_sha256") != digest:
        raise ValueError("freeze manifest identity changed")
    for filename, expected in frozen.get("files", {}).items():
        source = Path(filename)
        if not source.is_absolute():
            source = run_dir / source
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen source/artifact changed: {source}")
    return frozen.get("freeze_sha256"), {"status": "verified", "n_files": len(frozen.get("files", {}))}


def _audit_native(record, arm, protocol):
    n, k = protocol.get("n", 1024), protocol.get("k", 256)
    has_image = record.get("has_image", True)
    if type(has_image) is not bool:
        raise ValueError("has_image must be a boolean")
    expected = (n if arm == "full" else k) if has_image else 0
    source = (k if arm == "rres512" else n) if has_image else 0
    keep = record.get("keep_idx")
    if (not isinstance(keep, list) or len(keep) != expected or any(type(x) is not int for x in keep)
            or keep != sorted(set(keep)) or min(keep, default=0) < 0 or max(keep, default=-1) >= source):
        raise ValueError("invalid native token selection/budget")
    if record.get("actual_visual_count") != expected or record.get("source_visual_count") != source:
        raise ValueError("visual count differs from the frozen arm budget")
    if record.get("ds_lengths") != ([expected] * 3 if has_image else []):
        raise ValueError("DeepStack tokens do not match the native visual selection")
    audit = record.get("audit", {})
    if audit.get("native") is not True or audit.get("ds_streams") != (3 if has_image else 0) or audit.get("positions_preserved") is not True:
        raise ValueError("missing native position/DeepStack audit")
    if not has_image and audit.get("text_only_control") is not True:
        raise ValueError("no-image question requires the native text-only control")
    if audit.get("physical_visual_count", expected) != expected:
        raise ValueError("native physical token count mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("positions_sha256", "")):
        raise ValueError("missing original-position trace hash")


def _failed(record):
    return bool(record.get("error") or record.get("failed") or record.get("status") in ("error", "failed"))


def _audit_predictions(records, manifest, arms, protocol, freeze_hash):
    items = {(name, str(item["item_id"])): item for name, data in manifest["datasets"].items()
             for item in data["items"]}
    expected_items = sum(len(data["items"]) for data in manifest["datasets"].values())
    if len(items) != expected_items:
        raise ValueError("duplicate manifest question identity")
    indexed = {}
    for record in records:
        key = (str(record["dataset"]), str(record["item_id"]), str(record["arm"]))
        if key in indexed or key[:2] not in items or key[2] not in arms:
            raise ValueError(f"unexpected/duplicate prediction: {key}")
        item = items[key[:2]]
        if record.get("has_image", True) != item.get("has_image", True):
            raise ValueError("prediction image-presence flag differs from manifest")
        if str(record.get("parent_id")) != _parent(item):
            raise ValueError("prediction parent differs from original manifest")
        if freeze_hash is not None and record.get("freeze_sha256") != freeze_hash:
            raise ValueError("prediction freeze identity mismatch")
        failed = _failed(record)
        if not failed:
            _audit_native(record, key[2], protocol)
            if type(record.get("truncated")) is not bool or not isinstance(record.get("token_ids"), list):
                raise ValueError("missing generation termination/token trace")
            cap = protocol.get("max_new_tokens", 128)
            if record.get("max_new_tokens") != cap or len(record["token_ids"]) > cap:
                raise ValueError("generation cap differs from frozen protocol")
        recomputed = 0.0 if failed else score_item(key[0], record["text"], item["answers"], item.get("category"), metadata=item)
        parsed = None if failed else parse_prediction(key[0], record["text"], metadata=item)
        if "parsed_prediction" in record and record["parsed_prediction"] != parsed:
            raise ValueError(f"saved extraction differs from the original extractor: {key}")
        if "score" in record and not np.isclose(record["score"], recomputed, rtol=0, atol=1e-12):
            raise ValueError(f"stored score disagrees with original-answer scoring: {key}")
        indexed[key] = {**record, "score": recomputed, "failed": failed, "parsed_prediction": parsed}
    # Image-free controls must be identical across arms: there is nothing to
    # prune. This also guards accidental arm-specific text-only prompts.
    for (name, item_id), item in items.items():
        if not item.get("has_image", True):
            successful = [indexed[(name, item_id, arm)] for arm in arms
                          if (name, item_id, arm) in indexed and not indexed[(name, item_id, arm)]["failed"]]
            if successful and any((row["text"], row["token_ids"]) != (successful[0]["text"], successful[0]["token_ids"])
                                  for row in successful[1:]):
                raise ValueError("text-only controls differ across pruning arms")
    expected = {(name, item_id, arm) for name, item_id in items for arm in arms}
    return items, indexed, sorted(expected - set(indexed))


def _metric_summary(rows, n_expected):
    observed = len(rows)
    score_sum = sum(row["score"] for row in rows)
    complete = observed == n_expected and n_expected > 0
    return {"status": "complete" if complete else ("empty" if not n_expected else "incomplete"),
            "n_expected": n_expected, "n_observed": observed,
            "score": score_sum / n_expected if complete else None,
            "observed_score": score_sum / observed if observed else None,
            "full_denominator_score_bounds": ([score_sum / n_expected,
                                                (score_sum + n_expected - observed) / n_expected] if n_expected else None),
            "truncated": sum(bool(row.get("truncated")) for row in rows),
            "failed": sum(row["failed"] for row in rows)}


def _groups(data):
    ids = [str(item["item_id"]) for item in data["items"]]
    groups = {"all": ids}
    groups["image_only"] = [str(item["item_id"]) for item in data["items"] if item.get("has_image", True)]
    if any(not item.get("has_image", True) for item in data["items"]):
        groups["text_only_control"] = [str(item["item_id"]) for item in data["items"] if not item.get("has_image", True)]
    if any(item.get("annotation_warnings") for item in data["items"]):
        groups["annotation_clean"] = [str(item["item_id"]) for item in data["items"] if not item.get("annotation_warnings")]
    for field, label in (("decontaminated_item_ids", "decontaminated"),
                         ("conservative_item_ids", "conservative"),
                         ("novel_item_ids", "no_documented_history")):
        if field in data:
            subset = [str(i) for i in data[field]]
            if len(subset) != len(set(subset)) or not set(subset).issubset(ids):
                raise ValueError(f"invalid frozen {field}")
            groups[label] = subset
    categories = sorted({item["category"] for item in data["items"] if item.get("category")})
    for category in categories:
        groups[f"category: {category}"] = [str(item["item_id"]) for item in data["items"] if item.get("category") == category]
    return groups


def _seed_summary(arms, summaries, groups, seeds):
    learned = [arm for arm in arms if re.search(r"__seed_\d+$", arm)]
    families = sorted({arm.rsplit("__seed_", 1)[0] for arm in learned})
    result = {}
    for family in families:
        names = [f"{family}__seed_{seed}" for seed in seeds]
        result[family] = {}
        for group in groups:
            scores = {name: summaries[name][group]["score"] for name in names if name in summaries}
            complete = len(scores) == len(names) and all(value is not None for value in scores.values())
            values = list(scores.values()) if complete else []
            result[family][group] = {"status": "complete" if complete else "incomplete", "seeds": scores,
                                     "mean": float(np.mean(values)) if values else None,
                                     "range": [min(values), max(values)] if values else None,
                                     "note": "All predeclared seeds; no test-based seed selection. This range is not a confidence interval."}
    return result


def _timing_summary(records, manifest, arms, protocol, freeze_hash, draws, seed):
    config = protocol.get("timing", {})
    caps, blocks = config.get("caps", [1, 16]), config.get("blocks", 3)
    rosters = config.get("item_ids_by_dataset", {})
    items = {(name, str(item["item_id"])): item for name, data in manifest["datasets"].items() for item in data["items"]}
    for name, data in manifest["datasets"].items():
        if name not in rosters and "timing_item_ids" in data:
            rosters[name] = data["timing_item_ids"]
    expected = {(name, str(i), arm, int(cap), block) for name, ids in rosters.items()
                for i in ids for arm in arms for cap in caps for block in range(blocks)}
    indexed = {}
    for row in records:
        key = (str(row["dataset"]), str(row["item_id"]), str(row["arm"]), int(row["cap"]), int(row["block"]))
        if key in indexed or key[:2] not in items or key[2] not in arms or (expected and key not in expected):
            raise ValueError(f"unexpected/duplicate timing: {key}")
        if freeze_hash is not None and row.get("freeze_sha256") != freeze_hash:
            raise ValueError("timing freeze identity mismatch")
        if str(row.get("parent_id")) != _parent(items[key[:2]]):
            raise ValueError("timing parent differs from manifest")
        _audit_native(row, key[2], protocol)
        if not items[key[:2]].get("has_image", True) or row.get("has_image", True) is not True:
            raise ValueError("text-only controls cannot enter visual pruning timing")
        if not np.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] <= 0:
            raise ValueError("invalid timing duration")
        if len(row.get("token_ids", [])) != key[3] or row.get("generated_tokens", key[3]) != key[3]:
            raise ValueError("timing must generate the fixed cap, ignoring EOS")
        indexed[key] = row
    datasets = {}
    for name in manifest["datasets"]:
        ids = [str(i) for i in rosters.get(name, [])]
        if len(ids) != len(set(ids)) or any((name, i) not in items for i in ids):
            raise ValueError("timing roster is not a unique manifest subset")
        parents = [_parent(items[(name, i)]) for i in ids]
        clusters = [_cluster(items[(name, i)]) for i in ids]
        summaries, paired = {}, {}
        page_values = {}
        for arm in arms:
            summaries[arm] = {}
            for cap in caps:
                keys = [(name, i, arm, int(cap), block) for i in ids for block in range(blocks)]
                complete = bool(keys) and all(key in indexed for key in keys)
                if complete:
                    # One image median first; repeated blocks and duplicate
                    # pixels under different source paths stay in one cluster.
                    by_parent = defaultdict(list)
                    for item_id, cluster in zip(ids, clusters):
                        by_parent[cluster].append(float(np.median([indexed[(name, item_id, arm, int(cap), block)]["elapsed_ms"]
                                                                  for block in range(blocks)])))
                    page_values[(arm, int(cap))] = {p: float(np.median(v)) for p, v in by_parent.items()}
                values = list(page_values.get((arm, int(cap)), {}).values())
                summaries[arm][str(cap)] = {"status": "complete" if complete else "incomplete",
                                           "n_expected": len(keys), "n_observed": sum(key in indexed for key in keys),
                                           "n_source_parents": len(set(parents)) if complete else 0,
                                           "n_independent_clusters": len(values),
                                           "median_ms": float(np.median(values)) if values else None,
                                           "p25_p75_ms": np.percentile(values, [25, 75]).tolist() if values else None}
        for arm in arms:
            if "__seed_" not in arm:
                continue
            for reference in ("b1", "b2", "rres512"):
                for cap in caps:
                    a, b = page_values.get((reference, int(cap))), page_values.get((arm, int(cap)))
                    key = f"{arm}_vs_{reference}__cap_{cap}"
                    if not a or not b or set(a) != set(b):
                        paired[key] = {"status": "incomplete"}
                        continue
                    names = sorted(a)
                    av, bv = np.array([a[p] for p in names]), np.array([b[p] for p in names])
                    deltas, speedups = bv - av, av / bv
                    rng = np.random.default_rng(seed)
                    sample_index = rng.integers(0, len(names), size=(draws, len(names)))
                    paired[key] = {"status": "complete", "n_independent_clusters": len(names), "bootstrap_draws": draws,
                                   "median_delta_ms": float(np.median(deltas)),
                                   "delta_ci95_ms": np.percentile(np.median(deltas[sample_index], axis=1), [2.5, 97.5]).tolist(),
                                   "median_speedup": float(np.median(speedups)),
                                   "speedup_ci95": np.percentile(np.median(speedups[sample_index], axis=1), [2.5, 97.5]).tolist()}
        datasets[name] = {"n_roster_items": len(ids), "n_roster_parents": len(set(parents)),
                          "n_independent_clusters": len(set(clusters)),
                          "arms": summaries, "paired": paired}
    complete = (set(rosters) == set(manifest["datasets"]) and bool(expected) and len(indexed) == len(expected))
    return {"status": "complete" if complete else "incomplete", "n_expected": len(expected), "n_observed": len(indexed),
            "missing_keys": sorted(expected - set(indexed)), "datasets": datasets,
            "scope": config.get("scope", "Fresh image preparation, vision encoder, selection and fixed-length generation; excludes offline training/model load."),
            "aggregation": "Median over image-cluster medians; each item median includes all repeated paired blocks. Clusters prefer source_pixel_sha256, then pixel_sha256; image-free fallback is source parent. Duplicate pixels across different source categories/paths remain one cluster. Confidence intervals resample clusters, not individual repeats. Source parent identity and official MME question-pair groups are unchanged.",
            "limitations": "Fixed benchmark image panel and single-device latency; not serving throughput. Cap 1 is TTFT, cap 16 is fixed-length total generation time."}


def build_report(run_dir, draws=10000, seed=20261112, write=True):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    protocol = json.loads((run_dir / "protocol.json").read_text())
    for filename, expected in protocol.get("model_weight_stat", {}).items():
        stat = Path(filename).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError(f"frozen base model weights changed: {filename}")
    arms = protocol.get("arms", DEFAULT_ARMS)
    arms = [entry["name"] if isinstance(entry, dict) else str(entry) for entry in arms]
    if len(arms) != len(set(arms)):
        raise ValueError("duplicate arm name")
    freeze_hash, freeze_status = _verify_freeze(run_dir)
    records, timing_records, events = _load_database(run_dir / "evaluation/predictions.sqlite3")
    items, indexed, missing = _audit_predictions(records, manifest, arms, protocol, freeze_hash)
    datasets = {}
    for name, data in manifest["datasets"].items():
        groups = _groups(data)
        summaries = {arm: {group: _metric_summary([indexed[(name, i, arm)] for i in ids if (name, i, arm) in indexed], len(ids))
                           for group, ids in groups.items()} for arm in arms}
        comparisons = {}
        for arm in arms:
            if "__seed_" not in arm:
                continue
            for reference in ("b1", "b2", "rres512"):
                if reference not in arms:
                    continue
                comparisons[f"{arm}_vs_{reference}"] = {}
                for group, ids in groups.items():
                    valid = bool(ids) and all((name, i, a) in indexed for i in ids for a in (arm, reference))
                    if valid:
                        value = paired_bootstrap([indexed[(name, i, reference)]["score"] for i in ids],
                                                 [indexed[(name, i, arm)]["score"] for i in ids],
                                                 [_cluster(items[(name, i)]) for i in ids], draws, seed)
                        value["n_source_parents"] = len({_parent(items[(name, i)]) for i in ids})
                        value["status"] = "complete"
                    else:
                        value = {"status": "incomplete" if ids else "empty", "n_expected": len(ids)}
                    comparisons[f"{arm}_vs_{reference}"][group] = value
        history = data.get("historical_exposure", {})
        native = {arm: aggregate_metrics(name, data["items"],
                                         {str(item["item_id"]): indexed[(name, str(item["item_id"]), arm)]
                                          for item in data["items"] if (name, str(item["item_id"]), arm) in indexed})
                  for arm in arms}
        native_image_only = None
        if any(not item.get("has_image", True) for item in data["items"]):
            visible_items = [item for item in data["items"] if item.get("has_image", True)]
            native_image_only = {arm: aggregate_metrics(name, visible_items,
                                 {str(item["item_id"]): indexed[(name, str(item["item_id"]), arm)]
                                  for item in visible_items if (name, str(item["item_id"]), arm) in indexed}) for arm in arms}
        native_seed_summary = {}
        for family in sorted({arm.rsplit("__seed_", 1)[0] for arm in arms if "__seed_" in arm}):
            names = [f"{family}__seed_{seed_value}" for seed_value in protocol.get("seeds", [11, 23, 37])]
            scores = {arm: native[arm]["primary_score"] for arm in names if arm in native}
            complete = len(scores) == len(names) and all(value is not None for value in scores.values())
            values = list(scores.values()) if complete else []
            native_seed_summary[family] = {"seeds": scores, "mean": float(np.mean(values)) if values else None,
                                           "range": [min(values), max(values)] if values else None,
                                           "primary_unit": native[names[0]]["primary_unit"],
                                           "note": "Mean of native aggregate scores for every declared seed; range is not a CI."}
        exposure = Counter("exposed" if item.get("history_exposed") else "not_documented"
                           if item.get("history_exposed") is False and history.get("coverage") != "unknown"
                           else "unknown" for item in data["items"])
        datasets[name] = {"metric": metric_name(name), "n_questions": len(data["items"]),
                          "n_parents": len({_parent(item) for item in data["items"]}),
                          "n_independent_clusters": len({_cluster(item) for item in data["items"]}),
                          "independent_cluster_policy": "source_pixel_sha256 first, then pixel_sha256 for images; parent_id fallback for no-image controls. Source parent identity and native benchmark group identities remain unchanged.",
                          "subgroup_sizes": {group: len(ids) for group, ids in groups.items()},
                          "history_exposure_counts": dict(exposure),
                          "history_exposure_note": data.get("history_exposure_note", data.get("history_note", history.get("note", "History labels concern this project, not backbone pretraining exposure."))),
                          "historical_exposure": history,
                          "overlap_audit": {field: data.get(field) for field in
                                            ("training_overlap_count", "near_duplicate_suspected_count")},
                          "arms": summaries, "seed_summary": _seed_summary(arms, summaries, groups, protocol.get("seeds", [11, 23, 37])),
                          "native_metrics": native, "native_image_only": native_image_only,
                          "native_seed_summary": native_seed_summary,
                          "paired_comparisons": comparisons,
                          "paired_comparison_metric": "Auxiliary per-row correctness/ANLS differences only. These intervals are NOT confidence intervals for circular accuracy, MME points, POPE F1 or Hallusion grouped accuracy."}
        if name.lower() == "ocrbench":
            datasets[name]["official_aggregation_note"] = "Original OCRBench reports the sum of correct items (out of 1000 for the full benchmark) and sum/10 as percentage. Here score is mean correctness over the explicit frozen roster, with per-category denominators."
            for arm in arms:
                summaries[arm]["all"]["correct_sum"] = (summaries[arm]["all"]["score"] * len(data["items"])) if summaries[arm]["all"]["score"] is not None else None
    timing = _timing_summary(timing_records, manifest, arms, protocol, freeze_hash, draws, seed)
    qa_complete = not missing
    result = {"status": "complete" if qa_complete and timing["status"] == "complete" and freeze_status["status"] == "verified" else "incomplete",
              "qa_status": "complete" if qa_complete else "incomplete", "run_dir": str(run_dir),
              "n_expected": len(items) * len(arms), "n_observed": len(indexed), "missing_keys": missing,
              "freeze_sha256": freeze_hash, "freeze": freeze_status, "arms": arms,
              "datasets": datasets, "timing": timing, "metric_sources": source_metadata(),
              "n_error_events": sum(bool(event.get("error") or event.get("type") == "error" or event.get("status") == "error") for event in events),
              "policy": "Fixed checkpoints trained only on DocVQA TRAIN; no target-data training, tuning or test-seed selection. Missing predictions remain missing; failed attempts and unparsed answers score zero, truncations stay in the primary denominator. Native grouped benchmark scores are primary. Paired bootstrap intervals describe auxiliary per-row scores only, not native grouped metrics; no multiplicity correction. Resampling and timing use source-pixel clusters, so duplicate pixels across source paths/categories remain dependent. No cross-benchmark scalar average is reported.",
              "limitations": "Public benchmarks may have prior project exposure. Decontamination means no exact training image/parent overlap under the recorded checks, not absence from backbone pretraining or proof against near-duplicate contamination. MCQ and yes/no use deterministic local exact matching without an external judge. This must be distinguished from published scores obtained with LLM answer matching. HallusionBench all 1129 is an extended complete-original-table evaluation, including 178 image-free controls, NOT the conventional image-mode denominator. Its image-only 951 subset matches the pinned VLMEvalKit default skip_noimg=True image-mode roster; both are reported separately."}
    if write:
        _atomic_text(run_dir / "evaluation/report.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        _atomic_text(run_dir / "evaluation/report.md", markdown_report(result))
    return result


def _fmt(value, scale=100):
    return f"{value * scale:.3f}" if value is not None else "未完成"


def markdown_report(report):
    lines = ["# Frozen student and resolution reference: broad benchmark evaluation", "",
             f"Overall: **{report['status']}**. QA {report['n_observed']}/{report['n_expected']}; timing {report['timing']['n_observed']}/{report['timing']['n_expected']}.", "",
             report["policy"], "", report["limitations"], "",
             "Native benchmark aggregates are primary. Fraction metrics display as percentages; MME displays native points. No across-benchmark scalar average or cherry-picked seed is reported. Native aggregate confidence intervals are not estimated by the auxiliary row bootstrap.", "",
             "Subsets: decontaminated excludes recorded exact fit/validation image or source-parent overlap; conservative also excludes dHash near-duplicate suspects. no_documented_history additionally excludes indices in configured prior prediction tables. Missing history evidence is not proof that a benchmark is untouched.", ""]
    for name, data in report["datasets"].items():
        lines += [f"## {name}: {data['metric']}", "", f"{data['n_questions']} questions / {data['n_parents']} source parents / {data['n_independent_clusters']} independent pixel/parent clusters. History exposure: {json.dumps(data['history_exposure_counts'], ensure_ascii=False)}.", "", data["independent_cluster_policy"], "", str(data["history_exposure_note"]), "", f"Overlap audit: {json.dumps(data['overlap_audit'], ensure_ascii=False)}.", ""]
        if name == "HallusionBench":
            lines += ["HallusionBench all=1129 is the complete-original-table extended evaluation (951 image questions + 178 text-only controls). It is not the conventional image-mode denominator. The separate image_only=951 panel matches the pinned VLMEvalKit default skip_noimg=True image-mode roster.", ""]
        lines += ["### Primary native benchmark metric", "",
                  "| Arm | Native score | Auxiliary row score (%) | Observed/expected | Unparsed |",
                  "|---|---:|---:|---:|---:|"]
        for arm in report["arms"]:
            row = data["native_metrics"][arm]
            scale = 1 if row["primary_unit"] == "points" else 100
            lines.append(f"| {arm} | {_fmt(row['primary_score'], scale)} | {_fmt(row['row_score'])} | {row['n_observed']}/{row['n_expected']} | {row['unparsed']} |")
        lines.append("")
        for family, summary in data["native_seed_summary"].items():
            scale = 1 if summary["primary_unit"] == "points" else 100
            values = summary["range"]
            lines.append(f"{family}: native three-seed mean {_fmt(summary['mean'], scale)}; range {'–'.join(_fmt(value, scale) for value in values) if values else '未完成'}. The range is not a confidence interval.")
        lines.append("")
        if name in ("MME", "HallusionBench", "POPE"):
            columns = {"MME": ("perception", "reasoning"), "HallusionBench": ("aAcc", "fAcc", "qAcc"),
                       "POPE": ("Overall", "acc", "precision", "recall")}[name]
            lines += ["| Arm | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
            for arm in report["arms"]:
                native = data["native_metrics"][arm]
                overall = next((row for row in native["official"] if row.get("split", "Overall") == "Overall"), {})
                lines.append(f"| {arm} | " + " | ".join(_fmt(overall.get(column), 1) for column in columns) + " |")
            lines += ["", "Detailed native category/group metrics are preserved in report.json.", ""]
        if data.get("native_image_only") is not None:
            lines += ["### Native image-only sensitivity (image-free controls excluded)", "",
                      "This is the conventional HallusionBench image-mode roster (951 questions with the pinned VLMEvalKit default skip_noimg=True). The all=1129 extended panel above retains all original questions, including 178 text-only controls.", "",
                      "| Arm | Native score (%) | Questions |", "|---|---:|---:|"]
            for arm in report["arms"]:
                row = data["native_image_only"][arm]
                lines.append(f"| {arm} | {_fmt(row['primary_score'])} | {row['n_observed']}/{row['n_expected']} |")
            lines.append("")
        lines += ["### Auxiliary per-row diagnostics", "", data["paired_comparison_metric"], ""]
        for group, size in data["subgroup_sizes"].items():
            if group.startswith("category:"):
                continue
            lines += [f"#### {group} (n={size})", "", "| Arm | Auxiliary row score (%) | Observed/expected | Truncated | Failed |", "|---|---:|---:|---:|---:|"]
            for arm in report["arms"]:
                row = data["arms"][arm][group]
                lines.append(f"| {arm} | {_fmt(row['score'])} | {row['n_observed']}/{row['n_expected']} | {row['truncated']} | {row['failed']} |")
            lines.append("")
            for family, summaries in data["seed_summary"].items():
                summary = summaries[group]
                interval = summary["range"]
                lines.append(f"{family}: three-seed mean {_fmt(summary['mean'])}; range {'–'.join(_fmt(v) for v in interval) if interval else '未完成'}. Every seed is reported; this range is not a CI.")
            lines += ["", "| Paired candidate − reference | Difference (pp) | Pixel/parent-cluster bootstrap 95% CI (pp) | Improved / worsened / tied |", "|---|---:|---:|---:|"]
            for key, groups in data["paired_comparisons"].items():
                row = groups[group]
                if row["status"] == "complete":
                    ci = ", ".join(_fmt(v) for v in row["ci95"])
                    lines.append(f"| {key} | {_fmt(row['delta'])} | [{ci}] | {row['improved']} / {row['worsened']} / {row['tied']} |")
                else:
                    lines.append(f"| {key} | 未完成 | — | — |")
            lines.append("")
        categories = [group for group in data["subgroup_sizes"] if group.startswith("category:")]
        if categories:
            lines += ["### Task categories", "", "| Category (n) | " + " | ".join(report["arms"]) + " |",
                      "|---|" + "---:|" * len(report["arms"])]
            for group in categories:
                lines.append(f"| {group.removeprefix('category: ')} ({data['subgroup_sizes'][group]}) | " + " | ".join(_fmt(data["arms"][arm][group]["score"]) for arm in report["arms"]) + " |")
            lines.append("")
        if "official_aggregation_note" in data:
            lines += [data["official_aggregation_note"], ""]
    lines += ["## Latency", "", report["timing"]["scope"], "", report["timing"]["aggregation"], "", report["timing"]["limitations"], ""]
    for name, data in report["timing"]["datasets"].items():
        lines += [f"### {name}: {data['n_independent_clusters']} independent pixel/parent clusters ({data['n_roster_parents']} source parents)", "", "| Arm | Cap | Median ms | Observed/expected |", "|---|---:|---:|---:|"]
        for arm, caps in data["arms"].items():
            for cap, row in caps.items():
                lines.append(f"| {arm} | {cap} | {_fmt(row['median_ms'], 1)} | {row['n_observed']}/{row['n_expected']} |")
        lines += ["", "| Paired candidate − reference | Median Δ ms [95% CI] | Median speedup [95% CI] |", "|---|---:|---:|"]
        for key, row in data["paired"].items():
            if row["status"] == "complete":
                dc = ", ".join(_fmt(v, 1) for v in row["delta_ci95_ms"])
                sc = ", ".join(_fmt(v, 1) for v in row["speedup_ci95"])
                lines.append(f"| {key} | {_fmt(row['median_delta_ms'], 1)} [{dc}] | {_fmt(row['median_speedup'], 1)} [{sc}] |")
            else:
                lines.append(f"| {key} | 未完成 | 未完成 |")
        lines.append("")
    lines += ["## Audit", "", f"Freeze: {report['freeze']['status']}. Stored scores independently recomputed against original answers. Native token budgets, DeepStack alignment, original positions and token caps checked for all successful outputs.", "",
              "Missing outputs are never silently removed from denominators. See report.json for full-denominator bounds, category comparisons, source hashes and every missing key.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    report = build_report(args.run_dir, draws=args.draws)
    print(json.dumps({"status": report["status"], "n_observed": report["n_observed"], "n_expected": report["n_expected"],
                      "report": str(args.run_dir / "evaluation/report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
