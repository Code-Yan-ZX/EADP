"""Exercise the actual full scorer main using only fabricated predictions/truth."""
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_random_full_synthetic_driver", HERE / "score.py")
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestFullSynthetic(unittest.TestCase):
    def test_real_main_admission_failure_never_imports_truth_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); d = base / "analysis"; d.mkdir(); marker = base / "truth-read"
            e = base / "e1_analysis"; e.mkdir(); (e / "score.py").write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n")
            class Denied:
                @staticmethod
                def require(): raise ValueError("synthetic missing acceptance")
            with patch.object(driver, "D", d), patch.object(driver, "B", base), patch.object(driver, "load_gate", return_value=Denied()), patch.object(sys, "argv", ["score.py", "--out", str(d / "score001")]):
                with self.assertRaisesRegex(ValueError, "synthetic missing acceptance"): driver.main()
            self.assertFalse(marker.exists()); self.assertFalse((d / "score001").exists())

    def test_real_main_rejects_preloaded_scoring_module(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            class Accepted:
                @staticmethod
                def require(): return {}
            with patch.object(driver, "D", d), patch.object(driver, "load_gate", return_value=Accepted()), patch.object(sys, "argv", ["score.py", "--out", str(d / "score001")]), patch.dict(sys.modules, {"metrics": object()}):
                with self.assertRaisesRegex(ValueError, "preloaded scoring module"): driver.main()
            self.assertFalse((d / "score001").exists())

    def test_real_main_scores_complete_synthetic_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); d = base / "e6/random_full_analysis"; d.mkdir(parents=True)
            shutil.copy(HERE / "paired.py", d / "paired.py")
            e = base / "e1_analysis"; e.mkdir()
            (e / "completion.py").write_text("")
            (e / "metrics.py").write_text("def author_functions(): return None\ndef _author_constants(): return None\ndef score(*a, **k): return {'official': .75, 'independent': .75}\n")
            (e / "score.py").write_text("def read_truths(sources, needed): return {k: ['fake'] for k in needed}\n")
            anchors, records, e1, scored = [], [], [], []
            for task in driver.TASKS:
                for i in range(128):
                    a = {"task": task, "qid": str(i), "group_id": f"{task}-{i}", "message_sha256": "m", "image_sha256": "i", "processor": {"hash": "p"}}
                    anchors.append(a)
                    e1.append({**a, "arm": "P3-D1", "forced": False, "answer": "old", "eos": True, "generation_scalars": {"top_p": .001}, "eos_ids": [42]})
                    scored.append({"task": task, "qid": str(i), "group_id": a["group_id"], "arms": {"P3-D1": {"raw_answer": "old", "eos": True, "score": .5, "independent_score": .5}}})
                    for seed in driver.SEEDS:
                        records.append({**a, "seed": seed, "answer": "fabricated", "eos": True, "truncated": False, "generation_scalars": {"top_p": .001}, "eos_ids": [42], "selection_mode": "e6_random_fixed_seed", "budget": 256, "ds_on": True, "forced": False, "protocol": "natural", "extra_author_greedy": True})
            put(base / "e6/pixelprune_t045_dev384/plan001.json", {"anchors": anchors})
            jsonl(base / "e6/random_full_exec_v2/run_001/records.jsonl", records)
            jsonl(base / "e1/e1full001/raw_predictions.jsonl", e1); jsonl(base / "e1/e1full001_score/scores.jsonl", scored)
            sources = {}
            for task in driver.TASKS:
                p = base / "truth" / task; p.parent.mkdir(exist_ok=True); p.write_text("synthetic")
                sources[task] = {"path": str(p), "sha256": digest(p)}
            meta = base / "meta.json"; put(meta, {"sources": sources})
            put(base / "e1/e1full001_completion.json", {"files": {"dataset_summary": {"path": str(meta), "sha256": digest(meta)}}})
            class FakeAdmission:
                read = staticmethod(lambda p: json.loads(Path(p).read_text()))
                sha = staticmethod(digest)
                require = staticmethod(lambda: {"source_sha256": {}, "plan_sha256": "synthetic"})
            original_path = list(sys.path)
            with patch.object(driver, "D", d), patch.object(driver, "B", base), patch.object(driver, "load_gate", return_value=FakeAdmission()), patch.object(sys, "argv", ["score.py", "--out", str(d / "score001")]):
                try:
                    driver.main()
                finally:
                    sys.path[:] = original_path
                    for name in ("paired", "completion", "metrics", "score"): sys.modules.pop(name, None)
            result = json.loads((d / "score001/analysis.json").read_text())
            self.assertEqual((result["questions"], result["random_generations"]), (384, 1152))
            for task in driver.TASKS:
                self.assertEqual(result["tasks"][task]["questions"], 128)
                self.assertAlmostEqual(result["tasks"][task]["arm_mean_pct"]["eadp"], 50)
                self.assertAlmostEqual(result["tasks"][task]["arm_mean_pct"]["random_three_seed_mean"], 75)
                self.assertAlmostEqual(result["tasks"][task]["contrasts_vs_eadp"]["random_three_seed_mean"]["difference_pp"], 25)


if __name__ == "__main__": unittest.main()
