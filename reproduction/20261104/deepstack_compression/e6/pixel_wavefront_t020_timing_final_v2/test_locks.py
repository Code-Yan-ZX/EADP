import subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_gate_has_no_plan_and_is_closed(self):
  import gpu_gate
  with self.assertRaises(ValueError):gpu_gate.admit()
 def test_entries_reject_unfrozen_gate_before_any_gpu(self):
  for script,args in [('worker.py',['--job-index','0']),('launch.py',['--run-id','001'])]:
   p=subprocess.run([sys.executable,str(D/script)]+args,cwd=D,capture_output=True,text=True,timeout=10)
   self.assertNotEqual(p.returncode,0);self.assertIn('final GPU admission locked',p.stderr)
if __name__=='__main__':unittest.main()
