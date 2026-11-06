import os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
import runtime_support as r
class Tests(unittest.TestCase):
 def test_hidden_gpu_and_inventory_before_any_model_io(self):
  with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
   with self.assertRaisesRegex(ValueError,'hidden GPUs'):r.preflight({})
  with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
   with self.assertRaisesRegex(ValueError,'inventory mismatch'):r.preflight({})
   with tempfile.TemporaryDirectory() as d:
    source=Path(d)/'source.py';source.write_text('actual')
    with self.assertRaisesRegex(ValueError,'inventory mismatch'):r.preflight({'source_sha256':{str(source):'0'*64}})
 def test_direct_cli_is_locked(self):
  x=subprocess.run([sys.executable,str(Path(r.__file__))],capture_output=True,text=True,timeout=10)
  self.assertNotEqual(x.returncode,0);self.assertIn('locked',x.stderr)
 def test_preloaded_foreign_module_rejected(self):
  fake=types.ModuleType('_wavefront_fake_input');fake.__file__='/tmp/foreign.py'
  with mock.patch.dict(sys.modules,{'_wavefront_fake_input':fake}):
   with self.assertRaisesRegex(ValueError,'foreign cached module'):r.reject_foreign('_wavefront_fake_input',Path(__file__))
if __name__=='__main__':unittest.main(verbosity=2)
