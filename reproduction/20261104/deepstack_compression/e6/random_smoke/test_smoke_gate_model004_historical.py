import unittest
import re
from pathlib import Path
from unittest.mock import patch
import smoke_gate
class Tests(unittest.TestCase):
 def test_missing_plan_refuses_both_cpu_and_gpu(self):
  with patch.object(smoke_gate,'D',smoke_gate.D/'absent_dir'):
   for preflight in (True,False):
    with self.subTest(preflight=preflight),self.assertRaisesRegex(ValueError,'plan missing'):smoke_gate.admit(preflight)
 def test_worker_and_gate_bind_same_plan(self):
  d=Path(__file__).resolve().parent
  gate=set(re.findall(r'execution_plan[0-9]+\.json',(d/'smoke_gate.py').read_text()))
  worker=set(re.findall(r'execution_plan[0-9]+\.json',(d/'model_worker.py').read_text()))
  self.assertEqual(len(gate),1);self.assertEqual(gate,worker)
if __name__=='__main__':unittest.main()
