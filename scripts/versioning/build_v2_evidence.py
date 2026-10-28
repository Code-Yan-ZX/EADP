#!/usr/bin/env python3
"""Export existing EADP-paper/local evidence without importing model code.

This is a provenance gate, not a benchmark runner or a new score selector.
Unresolved records remain visible in quarantine.json/csv.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from collections import Counter


AUDIT = Path("Qwen_vl/outputs/audit_followup_20261008")
PAPER_TASKS = {
    1: ["VQAv2", "GQA", "VizWiz", "SQA_IMG", "TextVQA", "POPE", "MME", "MMB_EN", "MMB_CN", "MMVet", "Avg9"],
    2: ["VQAv2", "GQA", "VizWiz", "SQA_IMG", "TextVQA", "POPE", "MME", "MMB_EN", "MMB_CN", "MMVet", "Avg9"],
    4: ["TextVQA", "ChartQA", "AI2D", "OCRBench", "HallusionBench", "MME", "MMB_EN", "MMB_CN", "DocVQA", "InfoVQA", "Avg10"],
}
MODELS = {1: "v15", 2: "next", 4: "qwen"}
EXPECTED = {"GQA": 12578, "VizWiz": 4319, "SQA_IMG": 2017, "TextVQA": 5000,
            "POPE": 8910, "MME": 2374, "MMB_EN": 4876, "MMB_CN": 4876,
            "ChartQA": 2500, "AI2D": 3088, "OCRBench": 1000, "DocVQA": 5349, "InfoVQA": 2801, "HallusionBench": 951}


def close(a, b):
    return isinstance(a, (float, int)) and math.isfinite(a) and math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-12)


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def score_number(doc, task):
    """Read exact stored metrics; never treat a source hash as a score."""
    if isinstance(doc.get("value"), (int, float)):
        return doc["value"]
    if isinstance(doc.get("accuracy_percent"), (int, float)):
        return doc["accuracy_percent"]
    if task in ("MMB_EN", "MMB_CN") and "circular_correct" in doc:
        return 100 * doc["circular_correct"] / doc["n_groups"]
    if task == "MME" and "summary" in doc:
        return doc["summary"].get("perception")
    official = doc.get("official", {})
    def scalar(value):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return value.get("0", value.get(0)) if isinstance(value, dict) else value
    if official:
        if task == "MME":
            return scalar(official["perception"]) + scalar(official["reasoning"])
        if task == "HallusionBench":
            return scalar(official["fAcc"])
        if task == "OCRBench":
            return official["Final Score"]
        value = scalar(official["Overall"])
        return value * 100 if task in ("AI2D", "MMB_EN", "MMB_CN") else value
    if task == "GQA" and isinstance(doc.get("per_question"), list):
        rows = doc["per_question"]
        return 100 * sum(r["correct"] for r in rows) / len(rows)
    categories = doc.get("categories", doc)
    if task == "POPE" and all(k in categories for k in ("random", "popular", "adversarial")):
        return sum(200 * categories[k]["TP"] / (2 * categories[k]["TP"] + categories[k]["FP"] + categories[k]["FN"]) for k in ("random", "popular", "adversarial")) / 3
    return None


class PaperTables(HTMLParser):
    """Mirror the archived extraction's stripped-text table cells."""
    def __init__(self):
        super().__init__()
        self.tables, self.table, self.row, self.cell = {}, None, None, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table" and re.match(r"S5\.T[124]\.", attrs.get("id", "")):
            self.table = int(attrs["id"].split(".")[1][1:])
            self.tables[self.table] = []
        if self.table and tag == "tr":
            self.row = []
        if self.row is not None and tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None and data.strip():
            self.cell.append(data.strip())

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(normalized(" ".join(self.cell)))
            self.cell = None
        if tag == "tr" and self.table and self.row is not None:
            self.tables[self.table].append(self.row)
            self.row = None
        if tag == "table":
            self.table = None


class Audit:
    def __init__(self, root):
        self.root = root.resolve()
        self.cache, self.hashes, self.raw_cache = {}, {}, {}
        self.accepted, self.quarantine = [], []
        self.registry = {}

    def path(self, value):
        p = Path(value)
        if not p.is_absolute():
            return self.root / p
        # Recorded amp locations can be moved to another source-root.
        marker = "/EADP_amp/"
        return self.root / str(p).split(marker, 1)[1] if marker in str(p) else p

    def digest(self, value, expected=None):
        p = self.path(value)
        if p not in self.hashes:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            self.hashes[p] = h.hexdigest()
        actual = self.hashes[p]
        if expected and actual != expected:
            raise ValueError(f"SHA256 mismatch: {p}")
        return {"path": str(p), "sha256": actual, "recorded_sha_checked": bool(expected)}

    def read(self, value, expected=None):
        p = self.path(value)
        self.digest(p, expected)
        if p not in self.cache:
            self.cache[p] = json.loads(p.read_text())
        return self.cache[p]

    def raw(self, value, expected_hash=None, expected_n=None):
        p = self.path(value)
        proof = self.digest(p, expected_hash)
        if p not in self.raw_cache:
            if p.suffix == ".jsonl":
                with p.open() as f:
                    rows = [json.loads(line) for line in f if line.strip()]
            else:
                d = self.read(p)
                rows = list(d["records"].values()) if isinstance(d, dict) and "records" in d else d
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"Unsupported prediction format: {p}")
            if any(not isinstance(r, dict) or not any(k in r for k in ("text", "prediction", "answer")) for r in rows):
                raise ValueError(f"No raw answer fields: {p}")
            self.raw_cache[p] = len(rows)
        n = self.raw_cache[p]
        if expected_n is None or n != expected_n:
            raise ValueError(f"Prediction count {n} != recorded/benchmark count {expected_n}: {p}")
        return dict(proof, count=n)

    def fail(self, record, reason):
        self.quarantine.append(dict(record, reason=str(reason)))

    def papers(self):
        base = self.root / AUDIT
        ref = self.read(base / "paper_reference.json")
        html = base / "submission_tables_20261009/paper_tables_source.html"
        html_proof = self.digest(html, ref["sources"]["paper_html_sha256"])
        parser = PaperTables()
        parser.feed(html.read_text())
        for number in (1, 2, 4):
            source = base / f"submission_tables_20261009/paper_table_{number}.json"
            matrix = self.read(source)
            if parser.tables.get(number) != matrix:
                raise ValueError(f"Archived Table {number} extraction differs from primary HTML")
            budget, full_budget = 0, None
            for row in matrix[1:]:
                if len(row) == 1:
                    if row[0].startswith("Upper Bound, All "):
                        full_budget = int(re.search(r"All ([\d,]+) Tokens", row[0])[1].replace(",", ""))
                    if row[0].startswith("Retain "):
                        budget = int(re.search(r"Retain ([\d,]+)", row[0])[1].replace(",", ""))
                    continue
                if len(row) != 12:
                    raise ValueError(f"Unexpected Table {number} row width")
                method = "FULL" if budget == 0 else row[0].split(" (")[0]
                for task, literal in zip(PAPER_TASKS[number], row[1:]):
                    numbers = re.findall(r"\d+(?:\.\d+)?", literal)
                    if not numbers:
                        continue
                    for index, text_value in enumerate(numbers):
                        split = ("historical_test" if index == 0 else "validation") if len(numbers) == 2 else ("validation" if task == "VizWiz" and ("*" in literal or "∗" in literal) else "as_reported")
                        self.accepted.append({"id": f"paper:T{number}:{budget}:{method}:{task}:{split}", "origin": "eadp_paper_reported", "model": MODELS[number], "budget_nominal": full_budget if budget == 0 else budget, "method": method, "task": task, "split": split, "value": float(text_value), "printed_literal": literal, "status": "reported_in_EADP_paper", "source": str(source), "source_sha256": self.digest(source)["sha256"], "primary_html": html_proof, "paper_url": ref["sources"]["paper_url"] + f"#S5.T{number}", "recipe": "Paper-reported; local reproduction identity is not implied."})
            # Independently cross-check FULL/EADP reference rows against matrices.
            for table in ref["tables"]:
                if table["table"] != number:
                    continue
                for row in table["rows"]:
                    for task, value in row["values"].items():
                        if task == "Avg":
                            task = "Avg10" if number == 4 else "Avg9"
                        target_budget = row["budget_total_nominal"]
                        matches = [x for x in self.accepted if x["origin"] == "eadp_paper_reported" and x["model"] == MODELS[number] and x["budget_nominal"] == target_budget and x["method"] == row["method"] and x["task"] == task]
                        if not any(close(x["value"], value) for x in matches):
                            raise ValueError(f"Paper reference cross-check failed: T{number}/{row['method']}/{task}")

    def prediction_links(self, source, doc, cell):
        links = []
        if cell.get("prediction"):
            links.append((cell["prediction"], cell.get("prediction_sha256")))
        for item in cell.get("sources", []):
            if item.get("role") == "complete_prediction_shard":
                links.append((item["path"], item.get("sha256")))
        pred = doc.get("inputs", {}).get("predictions") if isinstance(doc, dict) else None
        if isinstance(pred, dict):
            links.append((pred["path"], pred.get("sha256")))
        if isinstance(doc, dict) and doc.get("result_file"):
            links.append((doc["result_file"], doc.get("prediction_sha256")))
        if str(source).endswith(".jsonl"):
            links.append((source, cell.get("source_sha256")))
        if isinstance(doc, dict) and doc.get("prediction_sha256") and not links:
            links.append((str(source).replace(".official.score.json", ".jsonl"), doc["prediction_sha256"]))
        return links

    def resolve(self, row, task, cell):
        source = self.path(cell["source"])
        proof = self.digest(source, cell.get("source_sha256") or self.registry.get(str(source)))
        doc = {} if source.suffix == ".jsonl" else self.read(source)
        value, n, recipe = cell["value"], cell.get("prediction_count", cell.get("n", EXPECTED.get(task))), cell.get("recipe", "")
        metric = cell.get("evaluator", doc.get("metric", doc.get("summary", {}).get("metric", "Recorded benchmark metric")))
        generation_recipe = doc.get("generation_recipe", doc.get("selected_recipe", doc.get("recipe")))
        if isinstance(doc, dict) and "rows" in doc:
            candidates = [r for r in doc["rows"] if r.get("model") == row["model"] and r.get("budget_parameter", r.get("budget")) == row["budget"] and r.get("method", "").split(" (")[0] == row["method"].split(" (")[0] and (r.get("task", task).lower() in (task.lower(), {"SQA_IMG": "sqa"}.get(task, task.lower()))) and close(r.get("value", r.get("accuracy")), value)]
            if len(candidates) != 1:
                raise ValueError("Summary does not identify one matching scored run")
            matched = candidates[0]
            metric = matched.get("metric", metric)
            generation_recipe = matched.get("generation_recipe", matched.get("recipe", matched.get("protocol", generation_recipe)))
            n = matched.get("generated_n", matched.get("n", n))
            if task in EXPECTED and n != EXPECTED[task]:
                raise ValueError("Summary prediction count differs from full benchmark scope")
            recipe = recipe or matched.get("protocol", "")
            raw_path = matched.get("prediction", matched.get("output"))
            raw = self.raw(raw_path, matched.get("prediction_sha256"), n)
        else:
            links = self.prediction_links(source, doc, cell)
            if not links:
                raise ValueError("No explicit raw-prediction link; summary/file existence alone is insufficient")
            if doc.get("n") is not None:
                n = doc["n"]
            if doc.get("n_rows") is not None:
                n = doc["n_rows"]
            if task in EXPECTED and n != EXPECTED[task]:
                raise ValueError("Score prediction count differs from full benchmark scope")
            raw = self.raw(*links[0], expected_n=n)
            recorded = score_number(doc, task)
            if recorded is not None and not close(recorded, value):
                raise ValueError("Cell differs from exact score.value")
            if doc.get("success") is False or doc.get("summary", {}).get("complete") is False:
                raise ValueError("Score marked incomplete/unsuccessful")
            if recorded is None:
                audit_path = cell.get("candidate_audit")
                if not audit_path:
                    raise ValueError("No exact score value or linked existing candidate audit")
                audit = self.read(audit_path, self.registry.get(str(self.path(audit_path))))
                matches = [candidate for c in audit["cells"] if (c["model"], c["budget"], c["task"]) == (row["model"], row["budget"], task) for candidate in c["candidates"] if candidate.get("source") == cell["source"] and close(candidate.get("value"), value) and candidate.get("source_sha256") == proof["sha256"] and candidate.get("complete")]
                if len(matches) != 1:
                    raise ValueError("No exact value in the linked existing candidate audit")
        caveat = cell.get("caveat", "")
        if task == "VizWiz":
            metric = "EADP released VizWiz accuracy (%): normalized min(matches/3,1)"
            caveat = str(caveat) + " Released EADP evaluator; official leave-one-out scores are a different metric and are not substituted here."
        return {"source": str(source), "source_sha256": proof["sha256"], "score_source_recorded_sha_checked": proof["recorded_sha_checked"], "raw_prediction": raw, "recipe": recipe or doc.get("protocol", "Recorded historical run; see source metadata"), "generation_recipe": generation_recipe, "metric": metric, "caveat": caveat, "verification": "Existing score/candidate value traced to raw answers with benchmark/recorded count; recorded SHA checked when available. Question-ID coverage is not independently re-audited here. No model generation or fresh benchmark scoring."}

    def locals(self):
        base = self.root / AUDIT
        table_path = base / "submission_tables_20261009/latest_tables.json"
        table = self.read(table_path)
        self.registry.update(table.get("sources_sha256", {}))
        self.registry.update(table.get("historical_best_source_sha256", {}))
        self.registry.update(table.get("historical_best_metadata", {}).get("candidate_files_sha256", {}))
        for row in table["rows"]:
            if row.get("kind") != "local":
                continue
            good = {}
            averages = []
            for task, cell in row["cells"].items():
                record = {"id": f"local:table:{row['model']}:{row['budget']}:{row['method']}:{task}", "origin": "local_experiment", "model": row["model"], "budget_cli": row["budget"], "method": row["method"], "task": task, "value": cell.get("value"), "legacy_status": cell.get("status"), "source": cell.get("source"), "selection": "Retained existing historical/table selection; no new maximum selected."}
                if task.startswith("Avg"):
                    averages.append((record, cell))
                    continue
                if cell.get("value") is None:
                    self.fail(record, "No recorded scored value; missing cells are not filled")
                    continue
                try:
                    details = self.resolve(row, task, cell)
                    self.accepted.append(dict(record, status="traceable_local_score", **details))
                    good[task] = self.accepted[-1]
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.fail(record, error)
            for record, cell in averages:
                tasks = PAPER_TASKS[4][:-1] if row["model"] == "qwen" else ["GQA", "POPE", "TextVQA", "SQA_IMG", "MME", "MMB_EN", "MMB_CN"]
                if not all(t in good for t in tasks):
                    self.fail(record, "Derived average withheld: missing accepted components " + ",".join(t for t in tasks if t not in good))
                    continue
                value = sum(good[t]["value"] / (20 if t == "MME" else 10 if t == "OCRBench" else 1) for t in tasks) / len(tasks)
                if not close(cell.get("value"), value):
                    self.fail(record, "Derived average differs from recomputed accepted components")
                    continue
                self.accepted.append(dict(record, value=value, status="derived_from_accepted_local_components", component_ids=[good[t]["id"] for t in tasks], caveat="Historical per-cell selections may mix recipes; this is not one unified or paired run."))

    def later_runs(self):
        base = self.root / AUDIT
        mmb = base / "mmb_eadp_immediate_20261010"
        summary = self.read(mmb / "result_summary.json")
        cell = {"value": summary["rows"][1]["circular_accuracy_percent"], "source": str(mmb / summary["sources"]["score"]), "recipe": "V11 / K128 / alpha.5 beta2 / lang=en / stream-wait / greedy max1024"}
        record = {"id": "local:later:v15:128:EADP:MMB_EN", "origin": "local_experiment", "model": "v15", "budget_cli": 128, "method": "EADP", "task": "MMB_EN", "value": cell["value"], "registration": "Later Oct10 run; separate from inherited table"}
        try:
            self.accepted.append(dict(record, status="traceable_local_score", **self.resolve({"model": "v15", "budget": 128, "method": "EADP"}, "MMB_EN", cell)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.fail(record, error)
        directory = base / "qwen128_mme_rerun_20261010"
        result = self.read(directory / "result.json")
        record = {"id": "local:later:qwen:128:AnchorZip:MME", "origin": "local_experiment", "model": "qwen", "budget_cli": 128, "method": "AnchorZip", "task": "MME", "value": result["summary"]["new"]["total"], "source": str(directory / "result.json"), "recipe": "legacy noDeepStack / 1D / bf16 SDPA / greedy2048 / K128 / completion lambda.25", "caveat": result["comparison_boundary"]}
        try:
            state = self.read(directory / "state.json")
            if not result["complete"] or state["status"] != "complete" or state["exit_code"] != 0:
                raise ValueError("Later Qwen run not complete")
            raw_path = self.path(state["output_root"]) / "legacy_full/acc/L_R_MAIN0125/MME.json"
            raw = self.raw(raw_path, result["fresh_shard_sha256"], result["n"])
            score_path = raw_path.with_name("MME_score.json")
            if not close(score_number(self.read(score_path), "MME"), record["value"]):
                raise ValueError("Later Qwen result differs from fresh score")
            self.accepted.append(dict(record, status="traceable_local_score", source_sha256=self.digest(record["source"])["sha256"], score_source=self.digest(score_path), raw_prediction=raw, verification="Complete new generation shard SHA/count and result equality to fresh stored score checked; no new scoring."))
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.fail(record, error)

    def write(self, out):
        out.mkdir(parents=True, exist_ok=True)
        counts = {"accepted": len(self.accepted), "accepted_by_origin": dict(Counter(r["origin"] for r in self.accepted)), "accepted_local_scores": sum(r.get("status") == "traceable_local_score" for r in self.accepted), "accepted_local_derived_averages": sum(r.get("status") == "derived_from_accepted_local_components" for r in self.accepted), "quarantined": len(self.quarantine), "quarantine_by_category": dict(Counter("dependent_average" if r["task"].startswith("Avg") else "missing_value" if r["value"] is None else "unverified_raw_link" for r in self.quarantine)), "quarantine_reasons": dict(Counter(r["reason"] for r in self.quarantine)), "source_files_hashed": len(self.hashes)}
        for name, records in (("evidence", self.accepted), ("quarantine", self.quarantine)):
            (out / f"{name}.json").write_text(json.dumps({"schema_version": 1, "records": records}, ensure_ascii=False, indent=2) + "\n")
            fields = sorted(set().union(*(r.keys() for r in records))) if records else ["id", "reason"]
            with (out / f"{name}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in records)
        manifest = {"schema_version": 1, "source_root": str(self.root), "counts": counts, "files": [{"path": str(p), "sha256": h} for p, h in sorted(self.hashes.items())], "limits": ["Only inherited main-table cells plus the two registered Oct10 later runs are scanned; this is not an exhaustive index of every local experiment.", "Paper values are copied from archived primary EADP Tables1/2/4, with HTML SHA and matrix equality checked. Printed paper averages remain reported values.", "Local checks establish an existing score-to-raw-answer provenance chain and count/SHA checks, not a fresh benchmark re-score or proof of protocol correctness. Exact question-ID coverage and GT identity are not independently re-audited here.", "Unsupported summary-only, rounded-only or directly supplied numbers are quarantined. Quarantine is not evidence that the underlying experiment never happened.", "Historical per-cell best values retain their old selection and recipe caveats; no new maxima or fair paired method claims are produced.", "Missing local benchmark scores are not replaced by paper values. Raw predictions remain in the original read-only source workspace."]}
        (out / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (out / "README.md").write_text("# V2 evidence snapshot\n\nOnly numbers reported in the EADP paper and traceable existing local experiment scores are accepted. Directly supplied numbers without experiments are excluded.\n\nRegenerate with Python's standard library (CPU only):\n\n```sh\npython scripts/versioning/build_v2_evidence.py --source-root /media/disk2/YZX/research/EADP_amp --output docs/versions/v2\n```\n\n" + f"Accepted: {counts['accepted']} records ({counts['accepted_by_origin']}); quarantined: {counts['quarantined']}.\n\n" + "`evidence.json/csv` contains accepted values; `quarantine.json/csv` explains withheld cells. `provenance.json` records checked source hashes and the exact scope and limitations. Paper budgets are nominal totals; local budgets retain their recorded CLI units (NeXT CLI128 is nominal640). Paper VizWiz test/validation values are separate. Local recipe, English-instruction CN and native/legacy caveats remain explicit. No new generation or score maximization is performed. Derived local averages are emitted only when every required component is accepted.\n")
        print(json.dumps(counts, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/versions/v2"))
    args = parser.parse_args()
    audit = Audit(args.source_root)
    audit.papers()
    audit.locals()
    audit.later_runs()
    audit.write(args.output)


if __name__ == "__main__":
    main()
