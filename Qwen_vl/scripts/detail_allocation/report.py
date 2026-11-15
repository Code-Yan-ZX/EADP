"""Read-only audit and finite-bank regional evidence diagnostic.

Every geometry is fixed before predictions. Max-over-bank scores and the
answer-dependent factorial controls are explicitly offline diagnostics. They
are never presented as an online selector or a deployment speed result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

import numpy as np

from scripts.student_transfer.metrics import score_item as transfer_score


BASELINES = ("full", "b1", "rres512", "uniform256") + tuple(
    f"bag_rank__seed_{seed}" for seed in (11, 23, 37))
CONTROLS = tuple(f"factorial__rep_{rep}__{pattern}"
                 for rep in (0, 1) for pattern in ("00", "10", "01"))


def score_item(dataset, prediction, answers):
    aliases = {"InfoVQA_TRAIN": "DocVQA_VAL", "TextVQA_TRAIN": "TextVQA_VAL"}
    if dataset not in aliases:
        raise ValueError(f"Unsupported diagnostic dataset: {dataset}")
    if dataset == "TextVQA_TRAIN" and len(answers) != 10:
        raise ValueError("TextVQA requires all ten original annotator answers")
    return transfer_score(aliases[dataset], prediction, answers)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def _freeze(run_dir, protocol):
    path = run_dir / "freeze.json"
    if not path.is_file():
        return None, {"status": "missing", "n_files": 0}
    frozen = _read(path)
    if _json_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"}) != frozen.get("freeze_sha256"):
        raise ValueError("Frozen manifest identity changed")
    files = frozen.get("files", {})
    if not files:
        raise ValueError("Freeze must cover source and input files")
    for filename, digest in files.items():
        source = Path(filename)
        if not source.is_absolute():
            source = run_dir / source
        if _sha(source) != digest:
            raise ValueError(f"Frozen source/input/checkpoint changed: {source}")
    for filename, expected in protocol.get("model_weight_stat", {}).items():
        stat = Path(filename).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError(f"Frozen base model weight changed: {filename}")
    for filename, digest in protocol.get("model_metadata_sha256", {}).items():
        if _sha(filename) != digest:
            raise ValueError(f"Frozen base model metadata changed: {filename}")
    return frozen["freeze_sha256"], {"status": "verified", "n_files": len(files)}


def _database(path):
    if not path.is_file():
        return [], [], [], None
    connection = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite quick_check failed")
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def rows(table):
            values = []
            if table not in tables:
                return values
            for raw in connection.execute(f'SELECT * FROM "{table}"'):
                outer = dict(raw)
                payload = json.loads(outer.pop("payload"))
                for key, value in outer.items():
                    if key in payload and key != "id" and payload[key] != value:
                        raise ValueError(f"SQLite key disagrees with payload: {key}")
                    payload.setdefault(key, value)
                values.append(payload)
            return values

        identity = connection.execute("SELECT value FROM metadata WHERE key='freeze_sha256'").fetchone() if "metadata" in tables else None
        return rows("predictions"), rows("timings"), rows("events"), identity[0] if identity else None
    finally:
        connection.close()


def _bank(protocol):
    actions = protocol.get("action_bank")
    if actions is None:
        from .geometry import build_bank
        actions = build_bank()
    if isinstance(actions, dict):
        actions = list(actions.values())
    actions = [dict(action) for action in actions]
    ids = [action["action_id"] for action in actions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate geometry action ID")
    families = Counter(action["family"] for action in actions)
    if families != {"single": 12, "dual": 12}:
        raise ValueError("Diagnostic requires equal banks of 12 single and 12 dual actions")
    for action in actions:
        keep = action["keep_idx"]
        if (len(keep) != 256 or any(type(i) is not int for i in keep)
                or keep != sorted(set(keep)) or min(keep) < 0 or max(keep) >= 1024):
            raise ValueError("Geometry must retain exactly 256 unique native positions")
    from .geometry import build_bank
    canonical = {action["action_id"]: action for action in build_bank()}
    if ({action["action_id"] for action in actions} != set(canonical)
            or any(action != canonical[action["action_id"]] for action in actions)):
        raise ValueError("Action bank differs from the frozen, marginal-matched geometry")
    return {action["action_id"]: action for action in actions}


@lru_cache(maxsize=24)
def _controls(action_json):
    from .geometry import factorial_controls
    return {f"factorial__rep_{action['replicate']}__{action['condition']}": action
            for action in factorial_controls(json.loads(action_json))}


def _failed(record):
    return bool(record.get("error") or record.get("failed")
                or record.get("status") in ("error", "failed"))


def _native(record, arm, protocol, bank):
    expected = 1024 if arm == "full" else 256
    source = 256 if arm == "rres512" else 1024
    keep = record.get("keep_idx")
    if (not isinstance(keep, list) or len(keep) != expected
            or any(type(i) is not int for i in keep) or keep != sorted(set(keep))
            or min(keep, default=0) < 0 or max(keep, default=-1) >= source):
        raise ValueError("Native selection indices/budget invalid")
    if record.get("actual_visual_count") != expected or record.get("source_visual_count") != source:
        raise ValueError("Native source or retained token count changed")
    if record.get("ds_lengths") != [expected] * 3:
        raise ValueError("Native three DeepStack streams not aligned")
    audit = record.get("audit", {})
    if not (audit.get("native") is True and audit.get("positions_preserved") is True
            and audit.get("ds_streams") == 3 and audit.get("fresh_kv") is True
            and audit.get("physical_visual_count", expected) == expected):
        raise ValueError("Native position/DeepStack/fresh-KV audit failed")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("positions_sha256", "")):
        raise ValueError("Missing native position trace")
    if record.get("gradients_enabled") is not False:
        raise ValueError("Generation must be gradient-free")
    dependent = arm in CONTROLS
    if record.get("selector_answer_dependent") is not dependent or audit.get("answer_in_selection") is not dependent:
        raise ValueError("Incorrect answer-dependence audit")
    if dependent:
        if record.get("selection_scope") != "offline_oracle_control":
            raise ValueError("Factorial controls must declare their offline oracle scope")
        action = record.get("selected_dual_action")
        if action not in bank or bank[action]["family"] != "dual":
            raise ValueError("Factorial geometry differs from selected dual action")
        control = _controls(json.dumps(bank[action], sort_keys=True))[arm]
        if keep != control["keep_idx"] or record.get("geometry") != control:
            raise ValueError("Factorial mask/donor geometry differs from its frozen control")
    elif arm in bank and keep != bank[arm]["keep_idx"]:
        raise ValueError("Fixed bank geometry changed")
    elif arm == "uniform256":
        from .geometry import uniform256_indices
        if keep != uniform256_indices():
            raise ValueError("Uniform native lattice changed")
    if "cache_length" in record and "prefill_length" in record:
        positions = record.get("decode_positions", [])
        if record["cache_length"] != record["prefill_length"] + len(positions):
            raise ValueError("Native cache length does not match decode trace")
        if positions and positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError("Native decode positions are not consecutive")


def _audit_predictions(records, manifest, protocol, bank, freeze_hash):
    arms = list(protocol.get("arms", [*BASELINES, *bank, *CONTROLS]))
    expected_arms = set(BASELINES) | set(bank) | set(CONTROLS)
    if len(arms) != len(expected_arms) or set(arms) != expected_arms:
        raise ValueError("Protocol must contain seven baselines, 24 geometries and six controls")
    items = {(name, str(item["item_id"])): item for name, data in manifest["datasets"].items()
             for item in data["items"]}
    if len(items) != sum(len(data["items"]) for data in manifest["datasets"].values()):
        raise ValueError("Duplicate main question ID")
    indexed = {}
    cap = protocol.get("max_new_tokens", 128)
    for record in records:
        key = (str(record["dataset"]), str(record["item_id"]), str(record["arm"]))
        if key in indexed or key[:2] not in items or key[2] not in arms:
            raise ValueError(f"Unexpected/duplicate prediction: {key}")
        item = items[key[:2]]
        if str(record.get("parent_id")) != str(item["parent_id"]):
            raise ValueError("Prediction source parent changed")
        if freeze_hash is not None and record.get("freeze_sha256") != freeze_hash:
            raise ValueError("Prediction freeze differs")
        failed = _failed(record)
        if not failed:
            _native(record, key[2], protocol, bank)
            if (not isinstance(record.get("text"), str) or not isinstance(record.get("token_ids"), list)
                    or type(record.get("truncated")) is not bool or len(record["token_ids"]) > cap):
                raise ValueError("Invalid raw answer/token/truncation trace")
        score = 0. if failed else score_item(key[0], record["text"], item["answers"])
        if "score" in record and not np.isclose(record["score"], score, rtol=0, atol=1e-12):
            raise ValueError(f"Saved score disagrees with original-answer rescoring: {key}")
        indexed[key] = {**record, "score": score, "failed": failed}
    missing = sorted((name, identity, arm) for name, identity in items for arm in arms
                     if (name, identity, arm) not in indexed)
    return arms, items, indexed, missing


def _bootstrap(values, parents, draws, seed):
    values = np.asarray(values, dtype=float)
    names, inverse = np.unique(np.asarray(parents, dtype=str), return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for start in range(0, draws, 128):
        end = min(start + 128, draws)
        ids = rng.integers(0, len(names), size=(end-start, len(names)))
        samples[start:end] = sums[ids].sum(1) / counts[ids].sum(1)
    return samples


def _timing(records, manifest, protocol, bank, freeze_hash, draws):
    config = protocol.get("timing", {})
    arms = config.get("arms", [])
    rosters = config.get("item_ids_by_dataset", {})
    caps, blocks = config.get("caps", [1, 16]), config.get("blocks", 3)
    lookup = {(name, str(item["item_id"])): item for name, data in manifest["datasets"].items()
              for item in [*data["items"], *data.get("dev", [])]}
    expected = {(name, str(identity), arm, int(cap), block)
                for name, ids in rosters.items() for identity in ids for arm in arms
                for cap in caps for block in range(blocks)}
    indexed = {}
    for record in records:
        key = (str(record["dataset"]), str(record["item_id"]), str(record["arm"]),
               int(record["cap"]), int(record["block"]))
        if key not in expected or key in indexed or key[:2] not in lookup:
            raise ValueError("Unexpected/duplicate timing record")
        if freeze_hash is not None and record.get("freeze_sha256") != freeze_hash:
            raise ValueError("Timing freeze changed")
        if str(record.get("parent_id")) != str(lookup[key[:2]]["parent_id"]):
            raise ValueError("Timing parent differs")
        _native(record, key[2], protocol, bank)
        if not np.isfinite(record["elapsed_ms"]) or record["elapsed_ms"] <= 0:
            raise ValueError("Invalid elapsed time")
        if len(record.get("token_ids", [])) != key[3] or record.get("generated_tokens", key[3]) != key[3]:
            raise ValueError("Timing must generate the exact fixed cap")
        indexed[key] = record
    summaries = {}
    for arm in arms:
        summaries[arm] = {}
        for cap in caps:
            grouped = defaultdict(list)
            complete = True
            for name, ids in rosters.items():
                for identity in ids:
                    keys = [(name, str(identity), arm, int(cap), block) for block in range(blocks)]
                    if not all(key in indexed for key in keys):
                        complete = False
                        continue
                    item = lookup[(name, str(identity))]
                    cluster = name + ":" + str(item.get("source_pixel_sha256", item["parent_id"]))
                    grouped[cluster].append(float(np.median([indexed[key]["elapsed_ms"] for key in keys])))
            values = [float(np.median(value)) for value in grouped.values()]
            summaries[arm][str(cap)] = {"status": "complete" if complete and values else "incomplete",
                "n_independent_images": len(values), "median_ms": float(np.median(values)) if complete and values else None}
    return {"status": "complete" if expected and set(indexed) == expected else "incomplete",
            "n_expected": len(expected), "n_observed": len(indexed),
            "missing_keys": sorted(expected-set(indexed)), "arms": summaries,
            "scope": "Fresh image processing, complete vision encoding, fixed geometry selection and native generation. Model loading and offline oracle search excluded.",
            "limitations": "Four development images only; fixed-mask diagnostic costs, not a deployable oracle or selector latency. Every high-resolution geometry still runs the full 1024-token visual encoder. No R-res speed claim transfers to these geometries."}


def _decision(datasets, pool):
    if any(not data["complete"] for data in datasets.values()):
        return "INCOMPLETE"
    if any(data["n_full_good_rres_bad"] < 12 for data in datasets.values()):
        return "HOLD_INSUFFICIENT_GAP"
    if (all(data["dual_minus_single"]["delta"] > 0 for data in datasets.values())
            and pool["delta"] >= .03 and pool["ci95"][0] > 0
            and sum(data["n_strict"] for data in datasets.values()) >= 8
            and all(data["n_strict"] >= 2 for data in datasets.values())):
        return "GO_DIAGNOSIS_ONLY"
    return "STOP_THIS_GEOMETRY"


def build_report(run_dir, draws=10000, write=True):
    run_dir = Path(run_dir).resolve()
    if draws < 1:
        raise ValueError("Positive bootstrap draws required")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = run_dir / "data/manifest.json"
    manifest, protocol = _read(manifest_path), _read(run_dir/"protocol.json")
    if set(manifest["datasets"]) != {"InfoVQA_TRAIN", "TextVQA_TRAIN"}:
        raise ValueError("Both preregistered source training datasets are required")
    freeze_hash, freeze = _freeze(run_dir, protocol)
    bank = _bank(protocol)
    records, timings, events, database_hash = _database(run_dir/"evaluation/predictions.sqlite3")
    if database_hash is not None and freeze_hash is not None and database_hash != freeze_hash:
        raise ValueError("Database and source freeze identities differ")
    arms, items, indexed, missing = _audit_predictions(records, manifest, protocol, bank, freeze_hash)
    datasets, distributions = {}, []
    for offset, (name, data) in enumerate(manifest["datasets"].items()):
        ids = [str(item["item_id"]) for item in data["items"]]
        complete = bool(ids) and all((name, identity, arm) in indexed for identity in ids for arm in arms)
        summaries = {}
        for arm in arms:
            rows = [indexed[(name, identity, arm)] for identity in ids if (name, identity, arm) in indexed]
            summaries[arm] = {"n_observed": len(rows), "n_expected": len(ids),
                "score": float(np.mean([row["score"] for row in rows])) if len(rows) == len(ids) else None,
                "observed_score": float(np.mean([row["score"] for row in rows])) if rows else None,
                "empty": sum(not row.get("text", "").strip() for row in rows),
                "failed": sum(row["failed"] for row in rows),
                "truncated": sum(bool(row.get("truncated")) for row in rows)}
        evidence, differences, parents = [], [], []
        strict_ids, gap_ids = [], []
        if complete:
            for identity in ids:
                item = items[(name, identity)]
                row = {arm: indexed[(name, identity, arm)] for arm in arms}
                best = {family: sorted((action for action in bank if bank[action]["family"] == family),
                                       key=lambda action: (-row[action]["score"], action))[0]
                        for family in ("single", "dual")}
                single, dual = row[best["single"]], row[best["dual"]]
                for arm in CONTROLS:
                    if row[arm].get("selected_dual_action") != best["dual"]:
                        raise ValueError("Factorial control did not use QA-max dual with deterministic tie breaking")
                gap = row["full"]["score"] >= .8 and row["rres512"]["score"] < .5
                strict = (gap and single["score"] < .5 and dual["score"] >= .8
                          and all(row[arm]["score"] < .5 for arm in CONTROLS)
                          and not any(row[arm]["failed"] for arm in arms))
                if gap:
                    gap_ids.append(identity)
                if strict:
                    strict_ids.append(identity)
                differences.append(dual["score"]-single["score"])
                parents.append(str(item["parent_id"]))
                shown = ("full", "rres512", best["single"], best["dual"], *CONTROLS)
                evidence.append({"item_id": identity, "parent_id": str(item["parent_id"]),
                    "image_path": item.get("image_path"), "question": item["question"], "answers": item["answers"],
                    "full_good_rres_bad": gap, "strict": strict,
                    "best_single_action": best["single"], "chosen_dual_action": best["dual"],
                    "single_score": single["score"], "dual_score": dual["score"],
                    "dual_minus_single": differences[-1],
                    "bank_scores": {arm: row[arm]["score"] for arm in bank},
                    "raw_evidence": {arm: {key: row[arm].get(key) for key in
                        ("text", "score", "token_ids", "truncated", "keep_idx", "positions_sha256",
                         "selected_dual_action", "selector_answer_dependent", "selection_scope",
                         "geometry", "donor_item_id", "donor_id", "donor_source", "replacement_diagnostics")} for arm in shown}})
            samples = _bootstrap(differences, parents, draws, 2026111401 + offset)
            distributions.append(samples)
            comparison = {"delta": float(np.mean(differences)), "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
                          "n_questions": len(ids), "n_parents": len(set(parents)), "bootstrap_draws": draws}
            full_gaps = [indexed[(name, identity, "full")]["score"]
                         - indexed[(name, identity, "rres512")]["score"] for identity in ids]
            full_samples = _bootstrap(full_gaps, parents, draws, 2026111451 + offset)
            reference_gap = {"delta": float(np.mean(full_gaps)),
                             "ci95": np.percentile(full_samples, [2.5, 97.5]).tolist(),
                             "n_questions": len(ids), "n_parents": len(set(parents))}
        else:
            comparison = {"delta": None, "ci95": None, "n_questions": len(ids), "bootstrap_draws": draws}
            reference_gap = {"delta": None, "ci95": None, "n_questions": len(ids)}
        datasets[name] = {"complete": complete, "n_questions": len(ids),
            "metric": "ANLS" if name.startswith("Info") else "VQA consensus (ten annotators)",
            "arms": summaries, "dual_minus_single": comparison,
            "n_full_good_rres_bad": len(gap_ids), "full_good_rres_bad_item_ids": gap_ids,
            "n_strict": len(strict_ids), "strict_item_ids": strict_ids, "evidence": evidence,
            "full_minus_rres": reference_gap,
            "finite_bank_oracle_means": {
                "single": float(np.mean([row["single_score"] for row in evidence])) if complete else None,
                "dual": float(np.mean([row["dual_score"] for row in evidence])) if complete else None,
                "selection": "Answer-dependent per-item QA maximum over 12 candidates in each family; never deployable accuracy."},
            "candidate_counts": {family: sum(action["family"] == family for action in bank.values())
                                 for family in ("single", "dual")}}
    all_complete = all(data["complete"] for data in datasets.values())
    if all_complete:
        pooled_samples = np.mean(distributions, axis=0)
        pooled = {"delta": float(np.mean([data["dual_minus_single"]["delta"] for data in datasets.values()])),
                  "ci95": np.percentile(pooled_samples, [2.5, 97.5]).tolist(),
                  "weighting": "Equal dataset weights; independent parent bootstrap within each dataset.", "bootstrap_draws": draws}
    else:
        pooled = {"delta": None, "ci95": None, "weighting": "Equal dataset weights", "bootstrap_draws": draws}
    timing = _timing(timings, manifest, protocol, bank, freeze_hash, draws)
    failures = sum(row["failed"] for row in indexed.values())
    decision = _decision(datasets, pooled)
    errors = sum(bool(event.get("error") or event.get("type") == "error" or event.get("status") == "error") for event in events)
    if failures:
        decision = "INCOMPLETE_ERRORS"
    result = {"status": "complete" if not missing and timing["status"] == "complete"
                and freeze["status"] == "verified" and not failures else "incomplete",
        "qa_status": "complete" if not missing and not failures else "incomplete",
        "n_expected": len(items)*len(arms), "n_observed": len(indexed), "missing_keys": missing,
        "freeze_sha256": freeze_hash, "freeze": freeze, "n_error_events": errors,
        "n_empty": sum(not row.get("text", "").strip() for row in indexed.values()),
        "n_failed": failures, "n_truncated": sum(bool(row.get("truncated")) for row in indexed.values()),
        "datasets": datasets, "equal_dataset_pool": pooled, "timing": timing, "decision": decision,
        "decision_rule": "HOLD if either dataset has fewer than 12 full>=0.8 and R-res<0.5 cases. GO only if both dual-minus-single point estimates>0, equal-dataset delta>=0.03 with CI lower>0, strict total>=8 and each dataset>=2; otherwise STOP_THIS_GEOMETRY.",
        "strict_rule": "full>=0.8; R-res<0.5; best of 12 single geometries<0.5; QA-max of 12 dual geometries>=0.8; all 00/10/01 controls in both donor replicates<0.5. Every record must be present and successful.",
        "interpretation": "Finite, balanced candidate-bank oracle diagnostic. Geometry generation is answer-free; best-over-bank reporting and selected-dual factorial controls use ground-truth scores. Not a deployable selector, training gain, global single-region upper bound, proof of causal locality, or novel method. All QA including truncations stay in denominators. STOP applies only to these geometries.",
        "limitations": "New source-training panels, not untouched benchmarks or proof of absence from base pretraining. Full encoder context can carry information across selected regions. Donors are alternative spatial units from the same encoded image, not edited pixels or an independent image. Two fixed donor pairs reduce one-filler accidents without proving causal necessity or excluding broader geometric effects. No deployment claim follows from an oracle win."}
    if write:
        _write(run_dir/"evaluation/report.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        _write(run_dir/"evaluation/report.md", markdown_report(result))
    return result


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _fmt(value, scale=100):
    return f"{value*scale:.3f}" if value is not None else "incomplete"


def markdown_report(report):
    lines = ["# Fixed-budget regional allocation diagnostic", "",
        f"Status: **{report['status']}**; decision: **{report['decision']}**. QA {report['n_observed']}/{report['n_expected']}; timing {report['timing']['n_observed']}/{report['timing']['n_expected']}.", "",
        report["interpretation"], "", report["limitations"], "", report["decision_rule"], "", report["strict_rule"], ""]
    for name, data in report["datasets"].items():
        lines += [f"## {name} ({data['metric']})", "",
            "| Arm | Score (%) | Observed/expected | Empty | Truncated | Failed |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, row in data["arms"].items():
            lines.append(f"| {arm} | {_fmt(row['score'])} | {row['n_observed']}/{row['n_expected']} | {row['empty']} | {row['truncated']} | {row['failed']} |")
        comparison = data["dual_minus_single"]
        interval = comparison["ci95"]
        lines += ["", f"Balanced 12-vs-12 finite-bank oracle dual − single: {_fmt(comparison['delta'])} pp; 95% parent-bootstrap CI {[_fmt(v) for v in interval] if interval is not None else 'incomplete'}.", "",
            f"Answer-dependent finite-bank best-single mean {_fmt(data['finite_bank_oracle_means']['single'])}%; best-dual mean {_fmt(data['finite_bank_oracle_means']['dual'])}%. Full − R-res reference gap {_fmt(data['full_minus_rres']['delta'])} pp.", "",
            f"Full-good/R-res-bad: {data['n_full_good_rres_bad']}; strict cases: {data['n_strict']}.", "",
            "Strict item IDs: "+", ".join(data["strict_item_ids"]), "",
            "Per-question scores for all 24 candidates, the selected single/dual, all six controls, original answers, and raw generated evidence are saved in report.json.", ""]
    pool = report["equal_dataset_pool"]
    lines += ["## Equal-dataset oracle comparison", "",
        f"Dual − single: {_fmt(pool['delta'])} pp; 95% CI {[_fmt(v) for v in pool['ci95']] if pool['ci95'] is not None else 'incomplete'}.", "", pool["weighting"], "",
        "## Development latency", "", report["timing"]["scope"], "", report["timing"]["limitations"], "",
        "| Arm | Cap | Median across independent image medians (ms) | Images |", "|---|---:|---:|---:|"]
    for arm, caps in report["timing"]["arms"].items():
        for cap, row in caps.items():
            lines.append(f"| {arm} | {cap} | {_fmt(row['median_ms'],1)} | {row['n_independent_images']} |")
    lines += ["", "## Audit", "",
        f"Freeze: {report['freeze']['status']} ({report['freeze']['n_files']} files). Missing QA: {len(report['missing_keys'])}; failed: {report['n_failed']}; empty: {report['n_empty']}; truncated: {report['n_truncated']}; error events: {report['n_error_events']}.", "",
        "Original-answer scores independently recomputed. Native token/DeepStack/position/cache traces and answer-dependence declarations audited. Missing records remain missing; no incomplete-panel success declaration.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    result = build_report(args.run_dir, draws=args.draws)
    print(json.dumps({key: result[key] for key in ("status", "decision", "n_expected", "n_observed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
