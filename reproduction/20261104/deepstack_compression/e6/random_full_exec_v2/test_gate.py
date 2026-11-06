import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gate
import launch,worker,sys,types

class TestFailClosed(unittest.TestCase):
 def test_missing_preliminary_review_rejected(self):
  with patch.object(gate,'plan',return_value={}),patch.object(gate,'prior',return_value={}),patch.object(gate,'review',side_effect=FileNotFoundError):
   with self.assertRaises(FileNotFoundError):gate.admit(True)
 def test_gpu_rejected_without_completed_preflight(self):
  with tempfile.TemporaryDirectory() as d, patch.object(gate,'D',Path(d)),patch.object(gate,'plan',return_value={}),patch.object(gate,'prior',return_value={}),patch.object(gate,'review',return_value='approved'),patch.object(gate,'sha',return_value='sha'):
   with self.assertRaises(FileNotFoundError):gate.admit(False)
 def test_tampered_gate_rejected_before_import(self):
  with tempfile.TemporaryDirectory() as d:
   bad=Path(d)/'gate.py';marker=Path(d)/'executed';bad.write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
   for module in (launch,worker):
    with self.assertRaises(ValueError):module.load_gate(bad)
    self.assertFalse(marker.exists())
 def test_preloaded_foreign_gate_rejected(self):
  fake=types.ModuleType('_random_full_host_gate');fake.__file__='/tmp/untrusted.py'
  with patch.dict(sys.modules,{'_random_full_host_gate':fake}):
   with self.assertRaises(ValueError):launch.load_gate()

if __name__=='__main__':unittest.main()
