import subprocess,sys,unittest
from pathlib import Path
import gpu_admission,single_arm
class Tests(unittest.TestCase):
 def test_gate_never_calls_model_or_gpu(self):
  with self.assertRaisesRegex(RuntimeError,'GPU locked'):gpu_admission.require_gpu()
  with self.assertRaisesRegex(RuntimeError,'GPU locked'):single_arm.run_one('DocVQA_VAL','16539','author','unused')
 def test_private_implementation_rejects_before_any_preflight_or_import(self):
  with self.assertRaisesRegex(RuntimeError,'GPU locked'):single_arm._draft_unadmitted_implementation('DocVQA_VAL','16539','author','unused')
 def test_both_clis_stay_locked(self):
  for path,extra in ((Path(single_arm.__file__),[]),(Path(__file__).parent/'controller.py',['--run-id','001'])):
   p=subprocess.run([sys.executable,str(path),*extra],capture_output=True,text=True,timeout=10)
   self.assertNotEqual(p.returncode,0);self.assertIn('GPU locked',p.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
