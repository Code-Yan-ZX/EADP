import subprocess,sys,os,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_gpu_is_hard_locked(self):
  import preflight_gate
  with self.assertRaises(RuntimeError):preflight_gate.gpu()
 def test_no_review_rejects_before_prior_reads(self):
  import preflight_gate
  if (D/'preflight_plan001.json').exists():self.skipTest('plan now exists')
  old=preflight_gate.prior
  try:
   preflight_gate.prior=lambda:(_ for _ in ()).throw(AssertionError('unexpected read'))
   with self.assertRaises(ValueError):preflight_gate.preflight()
  finally:preflight_gate.prior=old
 def test_same_name_fake_gate_rejected(self):
  code="import sys,types; m=types.ModuleType('_pixel_t020_runtime_preflight_gate');m.__file__='/tmp/fake.py';sys.modules[m.__name__]=m;import preflight_entry\ntry:preflight_entry.load(m.__name__,preflight_entry.D/'preflight_gate.py')\nexcept ValueError:pass\nelse:raise AssertionError('accepted foreign gate')"
  p=subprocess.run([sys.executable,'-c',code],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
  self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()
