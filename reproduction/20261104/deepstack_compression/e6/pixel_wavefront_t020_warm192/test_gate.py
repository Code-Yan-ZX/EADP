import subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gate
D=Path(__file__).resolve().parent
class GateTest(unittest.TestCase):
 def test_gpu_gate_cannot_pass_without_all_independent_proofs(self):
  with tempfile.TemporaryDirectory() as td,patch.object(gate,'D',Path(td)):
   with self.assertRaisesRegex(RuntimeError,'GPU locked'):gate.admit_gpu()
 def test_real_entrypoints_deny_before_creating_run(self):
  for file,args in [('worker.py',['--job-index','0']),('launch.py',['--run-id','001'])]:
   r=subprocess.run([sys.executable,str(D/file),*args],capture_output=True,text=True,timeout=30)
   self.assertNotEqual(r.returncode,0);self.assertIn('GPU locked',r.stderr)
  self.assertFalse((D/'run_001').exists())
if __name__=='__main__':unittest.main()
