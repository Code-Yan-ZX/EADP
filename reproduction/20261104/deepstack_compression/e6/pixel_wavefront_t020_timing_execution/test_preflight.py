import os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import gate,preflight_entry
class Tests(unittest.TestCase):
 def test_gpu_and_missing_review_fail_closed(self):
  with self.assertRaises(RuntimeError):gate.gpu()
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'D',Path(d)):
   with self.assertRaises(ValueError):gate.preflight()
 def test_real_launcher_does_not_create_output_before_approval(self):
  output=Path(preflight_entry.D)/'run_preflight001';self.assertFalse(output.exists())
  r=subprocess.run([sys.executable,str(preflight_entry.D/'preflight_launch.py'),'--out',str(output)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=20)
  self.assertNotEqual(r.returncode,0);self.assertFalse(output.exists())
 def test_reject_foreign_cached_model(self):
  x=types.ModuleType('_foreign_timing_test');x.__file__='/tmp/foreign.py'
  with patch.dict(sys.modules,{'_foreign_timing_test':x}):
   with self.assertRaises(ValueError):preflight_entry.load('_foreign_timing_test',Path(__file__))
if __name__=='__main__':unittest.main()
