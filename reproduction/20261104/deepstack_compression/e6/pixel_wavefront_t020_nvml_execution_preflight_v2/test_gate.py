import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gate,worker,launch
class TestLocks(unittest.TestCase):
 def test_gpu_is_unconditionally_locked(self):
  with self.assertRaises(RuntimeError):gate.gpu()
 def test_missing_plan_or_review_never_admits(self):
  with tempfile.TemporaryDirectory() as t,patch.object(gate,'D',Path(t)):
   with self.assertRaises(FileNotFoundError):gate.admit()
 def test_source_loader_rejects_tamper_before_execution(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'bad.py';marker=Path(t)/'marker';p.write_text('open('+repr(str(marker))+',"w").write("bad")')
   for mod in (worker,launch):
    with self.assertRaises(ValueError):mod.load('_nvml_bad',p,'0'*64)
    self.assertFalse(marker.exists())
