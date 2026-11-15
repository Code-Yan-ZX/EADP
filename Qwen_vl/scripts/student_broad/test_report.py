"""End-to-end CPU report fixtures, including native grouping and no-image rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from .metrics import parse_prediction, score_item
from .report import DEFAULT_ARMS, build_report


def fixture(directory):
    options = {"A": "red", "B": "blue"}
    datasets = {
        "MMBench_DEV_EN_V11": [dict(item_id=str(index), parent_id=f"p{index % 1000000}", source_index=index,
                                    options=options, answers=["A"]) for index in (7, 1000007, 9, 1000009)],
        "MMStar": [dict(item_id="s1", parent_id="s1", options=options, answers=["A"]),
                   dict(item_id="s2", parent_id="s2", options={"B": "blue", "C": "green"}, answers=["A"],
                        annotation_warnings=["empty_correct_option:A"])],
        "InfoVQA_VAL": [dict(item_id="i", parent_id="i", answers=["cat"])],
        "ChartQA_TEST": [dict(item_id="c", parent_id="c", answers=["100"])],
        "POPE": [dict(item_id="p1", parent_id="p1", answers=["Yes"], category="random"),
                 dict(item_id="p2", parent_id="p2", answers=["No"], category="random")],
        "HallusionBench": [dict(item_id=f"h{figure}_{question}", parent_id=f"h{figure}",
                                 source_index=f"a_b_c_0_{figure}_{question}", answers=["Yes"],
                                 has_image=bool(figure), category="VS", **{"l2-category": "illusion"})
                           for figure in range(2) for question in range(2)]}
    for items in datasets.values():
        for item in items:
            item.setdefault("has_image", True)
    manifest = {"datasets": {name: {"items": items} for name, items in datasets.items()}}
    protocol = {"n": 4, "k": 2, "max_new_tokens": 128, "arms": DEFAULT_ARMS, "seeds": [11, 23, 37],
                "model_weight_stat": {}, "timing": {"caps": [1, 16], "blocks": 1,
                "item_ids_by_dataset": {name: [next(item["item_id"] for item in items if item["has_image"])]
                                        for name, items in datasets.items()}}}
    source = directory / "frozen_source.py"
    source.write_text("before")
    frozen = {"files": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
    frozen["freeze_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    for filename, content in (("manifest.json", manifest), ("protocol.json", protocol), ("freeze.json", frozen)):
        (directory / filename).write_text(json.dumps(content))
    (directory / "evaluation").mkdir()
    with sqlite3.connect(directory / "evaluation/predictions.sqlite3") as connection:
        connection.execute("CREATE TABLE predictions(dataset TEXT,item_id TEXT,arm TEXT,payload TEXT,PRIMARY KEY(dataset,item_id,arm))")
        connection.execute("CREATE TABLE timings(dataset TEXT,item_id TEXT,arm TEXT,cap INTEGER,block INTEGER,payload TEXT,PRIMARY KEY(dataset,item_id,arm,cap,block))")
        for name, items in datasets.items():
            for item in items:
                for arm in DEFAULT_ARMS:
                    has_image = item["has_image"]
                    count = (4 if arm == "full" else 2) if has_image else 0
                    source_count = (2 if arm == "rres512" else 4) if has_image else 0
                    text = item["answers"][0]
                    if arm.endswith("seed_23") and item["item_id"] == "1000007":
                        text = "B"
                    common = {"dataset": name, "item_id": item["item_id"], "parent_id": item["parent_id"], "arm": arm,
                              "has_image": has_image, "actual_visual_count": count, "source_visual_count": source_count,
                              "keep_idx": list(range(count)), "ds_lengths": [count] * 3 if has_image else [],
                              "positions_sha256": "a" * 64, "freeze_sha256": frozen["freeze_sha256"],
                              "audit": {"native": True, "ds_streams": 3 if has_image else 0,
                                        "positions_preserved": True, "text_only_control": not has_image}}
                    record = {**common, "text": text, "token_ids": [1], "truncated": arm.endswith("seed_37"),
                              "max_new_tokens": 128, "score": score_item(name, text, item["answers"], metadata=item),
                              "parsed_prediction": parse_prediction(name, text, item)}
                    connection.execute("INSERT INTO predictions VALUES(?,?,?,?)", (name, item["item_id"], arm, json.dumps(record)))
                    if item["item_id"] in protocol["timing"]["item_ids_by_dataset"][name]:
                        for cap in (1, 16):
                            timing = {**common, "cap": cap, "block": 0, "token_ids": [1] * cap,
                                      "generated_tokens": cap, "elapsed_ms": 100. + cap - (10 if "seed_" in arm else 0)}
                            connection.execute("INSERT INTO timings VALUES(?,?,?,?,?,?)", (name, item["item_id"], arm, cap, 0, json.dumps(timing)))
    return manifest


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.manifest = fixture(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, transform, dataset="MMBench_DEV_EN_V11", item_id="7", arm="full"):
        with sqlite3.connect(self.directory / "evaluation/predictions.sqlite3") as connection:
            payload, = connection.execute("SELECT payload FROM predictions WHERE dataset=? AND item_id=? AND arm=?", (dataset, item_id, arm)).fetchone()
            row = json.loads(payload)
            transform(row)
            connection.execute("UPDATE predictions SET payload=? WHERE dataset=? AND item_id=? AND arm=?", (json.dumps(row), dataset, item_id, arm))

    def test_native_main_scores_and_auxiliary_differences_are_separate(self):
        report = build_report(self.directory, draws=10)
        self.assertEqual(report["status"], "complete")
        data = report["datasets"]["MMBench_DEV_EN_V11"]
        self.assertEqual(data["native_metrics"]["bag_rank__seed_23"]["primary_score"], .5)
        self.assertEqual(data["arms"]["bag_rank__seed_23"]["all"]["score"], .75)
        self.assertAlmostEqual(data["native_seed_summary"]["bag_rank"]["mean"], 5 / 6)
        self.assertIn("NOT confidence intervals", data["paired_comparison_metric"])
        markdown = (self.directory / "evaluation/report.md").read_text()
        self.assertIn("Primary native benchmark metric", markdown)
        self.assertIn("Auxiliary per-row diagnostics", markdown)
        self.assertIn("Native image-only sensitivity", markdown)
        self.assertIn("complete-original-table extended evaluation", markdown)
        self.assertIn("skip_noimg=True", markdown)

    def test_hallusion_no_image_controls_retained_and_image_subset_separate(self):
        data = build_report(self.directory, draws=10, write=False)["datasets"]["HallusionBench"]
        self.assertEqual(data["native_metrics"]["full"]["n_expected"], 4)
        self.assertEqual(data["native_image_only"]["full"]["n_expected"], 2)
        self.assertEqual(data["subgroup_sizes"]["text_only_control"], 2)
        self.assertEqual(data["subgroup_sizes"]["image_only"], 2)

    def test_missing_official_annotation_kept_and_clean_subset_reported(self):
        data = build_report(self.directory, draws=10, write=False)["datasets"]["MMStar"]
        self.assertEqual(data["native_metrics"]["full"]["primary_score"], .5)
        self.assertEqual(data["arms"]["full"]["annotation_clean"]["score"], 1.)

    def test_missing_prediction_preserves_denominator_and_prevents_native_result(self):
        with sqlite3.connect(self.directory / "evaluation/predictions.sqlite3") as connection:
            connection.execute("DELETE FROM predictions WHERE dataset='MMBench_DEV_EN_V11' AND item_id='1000007' AND arm='full'")
        report = build_report(self.directory, draws=10, write=False)
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["datasets"]["MMBench_DEV_EN_V11"]["native_metrics"]["full"]["primary_score"])
        row = report["datasets"]["MMBench_DEV_EN_V11"]["arms"]["full"]["all"]
        self.assertEqual(row["n_expected"], 4)
        self.assertEqual(row["full_denominator_score_bounds"], [.75, 1.])

    def test_failures_and_truncations_remain_in_main_denominator(self):
        self.mutate(lambda row: row.update(failed=True, score=0, parsed_prediction=None))
        data = build_report(self.directory, draws=10, write=False)["datasets"]["MMBench_DEV_EN_V11"]
        self.assertEqual(data["native_metrics"]["full"]["primary_score"], .5)
        self.assertEqual(data["arms"]["bag_rank__seed_37"]["all"]["truncated"], 4)

    def test_score_parse_budget_and_image_presence_corruption_are_rejected(self):
        database = self.directory / "evaluation/predictions.sqlite3"
        original = database.read_bytes()
        for transform in (lambda row: row.update(score=.1), lambda row: row.update(parsed_prediction="B"),
                          lambda row: row.update(keep_idx=[1] * 4), lambda row: row.update(has_image=False),
                          lambda row: row.update(ds_lengths=[4, 4]), lambda row: row.update(max_new_tokens=129)):
            with self.subTest(transform=transform):
                self.mutate(transform)
                with self.assertRaises(ValueError):
                    build_report(self.directory, draws=10, write=False)
                database.write_bytes(original)

    def test_text_only_arm_disagreement_is_rejected(self):
        self.mutate(lambda row: row.update(text="No", score=0, parsed_prediction="No"), "HallusionBench", "h0_0", "b1")
        with self.assertRaises(ValueError):
            build_report(self.directory, draws=10, write=False)

    def test_native_text_only_requires_explicit_zero_visual_trace(self):
        self.mutate(lambda row: row.update(ds_lengths=[0, 0, 0]), "HallusionBench", "h0_0", "full")
        with self.assertRaises(ValueError):
            build_report(self.directory, draws=10, write=False)

    def test_missing_timing_or_freeze_cannot_claim_completion(self):
        with sqlite3.connect(self.directory / "evaluation/predictions.sqlite3") as connection:
            connection.execute("DELETE FROM timings WHERE rowid=(SELECT min(rowid) FROM timings)")
        report = build_report(self.directory, draws=10, write=False)
        self.assertEqual(report["qa_status"], "complete")
        self.assertEqual(report["status"], "incomplete")
        (self.directory / "freeze.json").unlink()
        self.assertEqual(build_report(self.directory, draws=10, write=False)["status"], "incomplete")

    def test_frozen_dependency_or_freeze_manifest_mutation_is_rejected(self):
        source = self.directory / "frozen_source.py"
        source.write_text("after")
        with self.assertRaises(ValueError):
            build_report(self.directory, draws=10, write=False)
        source.write_text("before")
        frozen = json.loads((self.directory / "freeze.json").read_text())
        frozen["extra"] = "changed"
        (self.directory / "freeze.json").write_text(json.dumps(frozen))
        with self.assertRaises(ValueError):
            build_report(self.directory, draws=10, write=False)

    def test_duplicate_pixels_share_bootstrap_cluster_without_changing_parents(self):
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        items = manifest["datasets"]["MMBench_DEV_EN_V11"]["items"]
        for item in items:
            item["source_pixel_sha256"] = "shared_source_pixels"
            item["pixel_sha256"] = item["item_id"]
        manifest_path.write_text(json.dumps(manifest))
        data = build_report(self.directory, draws=10, write=False)["datasets"]["MMBench_DEV_EN_V11"]
        self.assertEqual(data["n_parents"], 2)
        self.assertEqual(data["n_independent_clusters"], 1)
        pair = data["paired_comparisons"]["bag_rank__seed_23_vs_b1"]["all"]
        self.assertEqual(pair["n_source_parents"], 2)
        self.assertEqual(pair["n_independent_clusters"], 1)
        self.assertEqual(pair["ci95"], [-.25, -.25])
        self.assertEqual(data["native_metrics"]["bag_rank__seed_23"]["n_circular_groups"], 2)

    def test_timing_duplicate_pixels_cluster_across_source_parent_ids(self):
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        items = manifest["datasets"]["MMBench_DEV_EN_V11"]["items"]
        for item in items:
            item["source_pixel_sha256"] = "shared_source_pixels"
        manifest_path.write_text(json.dumps(manifest))
        protocol_path = self.directory / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["timing"]["item_ids_by_dataset"]["MMBench_DEV_EN_V11"] = ["7", "9"]
        protocol_path.write_text(json.dumps(protocol))
        with sqlite3.connect(self.directory / "evaluation/predictions.sqlite3") as connection:
            for arm, cap, payload in connection.execute("SELECT arm,cap,payload FROM timings WHERE dataset='MMBench_DEV_EN_V11' AND item_id='7'").fetchall():
                row = json.loads(payload)
                row.update(item_id="9", parent_id="p9", elapsed_ms=row["elapsed_ms"] + 20)
                connection.execute("INSERT INTO timings VALUES(?,?,?,?,?,?)", ("MMBench_DEV_EN_V11", "9", arm, cap, 0, json.dumps(row)))
        data = build_report(self.directory, draws=10, write=False)["timing"]["datasets"]["MMBench_DEV_EN_V11"]
        self.assertEqual(data["n_roster_parents"], 2)
        self.assertEqual(data["n_independent_clusters"], 1)
        self.assertEqual(data["arms"]["full"]["1"]["n_independent_clusters"], 1)
        self.assertEqual(data["paired"]["bag_rank__seed_11_vs_b1__cap_1"]["n_independent_clusters"], 1)


if __name__ == "__main__":
    unittest.main()
