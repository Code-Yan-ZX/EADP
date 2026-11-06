import unittest
import re
import hashlib
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
 def test_worker_and_launcher_pin_actual_gate(self):
  d=Path(__file__).resolve().parent;expected=hashlib.sha256((d/'smoke_gate.py').read_bytes()).hexdigest()
  for name in ('model_worker.py','model_launch.py'):
   found=re.findall(r"hexdigest\(\)!=\'([0-9a-f]{64})\'",(d/name).read_text())
   self.assertEqual(found,[expected])
if __name__=='__main__':unittest.main()
