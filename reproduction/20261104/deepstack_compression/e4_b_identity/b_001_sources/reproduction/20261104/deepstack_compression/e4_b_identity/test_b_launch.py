import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import b_launch

class BudgetTests(unittest.TestCase):
    def test_local_loader_ignores_preloaded_same_named_module(self):
        import sys,types
        fake=types.ModuleType('b_gate');fake.__file__='/tmp/b_gate.py'
        with patch.dict(sys.modules,{'b_gate':fake}):
            self.assertEqual(Path(b_launch.local('b_gate').__file__).resolve(),b_launch.HERE/'b_gate.py')
            self.assertIsNot(b_launch.local('b_gate'),fake)
    def test_requires_bounded_supervisor_and_remaining_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);here=base/'b';here.mkdir();(base/'initial_plan.json').write_text(json.dumps({'budget_end_unix':int(time.time())+10000}))
            budget={'inner_timeout_seconds':3600,'outer_timeout_seconds':3720,'budget_source_sha256':'h',
                'budget_end_unix':int(time.time())+10000,'journal_stale_seconds':900,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90}
            (here/'budget_plan.json').write_text(json.dumps(budget));mon=here/'monitor_b001';mon.mkdir()
            started={'worker_pid':os.getpid(),'timeout_seconds':3720,'stale_seconds':900,
                'progress_paths':[str((here/'b_001/generated.jsonl').resolve())],
                'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90}
            (mon/'started.json').write_text(json.dumps(started))
            with patch.object(b_launch,'HERE',here),patch.object(b_launch,'sha',lambda _: 'h'):
                b_launch.check_budget('001',3600)
                started['stale_seconds']=0;(mon/'started.json').write_text(json.dumps(started))
                with self.assertRaises(ValueError):b_launch.check_budget('001',3600)
                with self.assertRaises(ValueError):b_launch.check_budget('001',4000)

if __name__=='__main__':unittest.main()
