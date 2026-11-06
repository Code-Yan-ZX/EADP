"""CPU-only checks for strict plan shape and bounded child cleanup."""
import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('author_graph_worker_test',HERE/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

class TestWorker(unittest.TestCase):
    def test_invalid_plan_rejected_before_evidence(self):
        for plan in ({},{'stage':'author CUDA graph feasibility','budgets':[512],'child_timeout_seconds':300,'sha256':{}}):
            with self.assertRaises(ValueError):w.validate_plan(plan)
    def test_missing_sources_rejected(self):
        p={'stage':'author CUDA graph feasibility','budgets':[1,16,128,256,512],
           'child_timeout_seconds':300,'sha256':{}}
        with self.assertRaises(ValueError):w.validate_plan(p)
    def test_child_cleanup_and_durable_journal(self):
        with tempfile.TemporaryDirectory() as d:
            journal=Path(d)/'progress.jsonl'
            w.event(journal,{'event':'started'});w.event(journal,{'event':'failed','k':512})
            self.assertEqual([json.loads(x)['event'] for x in journal.read_text().splitlines()],['started','failed'])
            p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
            try:w.stop_child(p);self.assertIsNotNone(p.poll())
            finally:
                if p.poll() is None:p.kill();p.wait()
    def test_wrong_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'independent_prelaunch_review.json').write_text(json.dumps({
                'review_terminal_path':'/tmp/outside/terminal.json','reviewer_path':'/tmp/outside/review.py'}))
            with patch.object(w,'HERE',root):
                with self.assertRaises(ValueError):w.require_review(root/'plan.json')

if __name__=='__main__':unittest.main(verbosity=2)
