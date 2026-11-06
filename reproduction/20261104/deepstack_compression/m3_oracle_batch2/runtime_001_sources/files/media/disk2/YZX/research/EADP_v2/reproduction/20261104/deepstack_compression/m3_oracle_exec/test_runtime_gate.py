import importlib.util,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime_gate,runtime_worker
class RuntimeGateTests(unittest.TestCase):
    def test_absent_approval_fails_before_model_prediction_or_truth(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runtime_gate,'APPROVAL',Path(d)/'missing'), \
             patch.object(sys,'argv',['runtime_worker.py','--execute','--out',d]), \
             patch.object(runtime_worker.e1_admission,'validate',side_effect=AssertionError('prediction accessed')), \
             patch.object(runtime_worker.lazy_truth,'LazyTruth',side_effect=AssertionError('GT accessed')):
            with self.assertRaisesRegex(ValueError,'locked'):runtime_worker.main()
    def test_budget_rejects_expired_or_wrong_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.json';p.write_text('{"budget_start_unix":100,"budget_end_unix":200}')
            plan={'budget_source':str(p),'budget_source_sha256':runtime_gate.sha(p),'budget_start_unix':100,'budget_end_unix':200,
                  'outer_timeout_seconds':50,'inner_timeout_seconds':40,'question_checkpoint_seconds':10,'snapshot_seconds':3600,
                  'progress_stale_seconds':20,'questions':128,'new_generations_per_question':6,'maximum_new_generations':768}
            runtime_gate.check_budget(plan,140)
            with self.assertRaises(ValueError):runtime_gate.check_budget(plan,151)
            p.write_text('{}')
            with self.assertRaises(Exception):runtime_gate.check_budget(plan,140)
if __name__=='__main__':unittest.main()
