import subprocess,sys,unittest,os
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_real_bridge_module_import_without_model(self):
  import bridge
  loaded=bridge.prepare_cpu();self.assertEqual(set(loaded),{'worker','controller','host_runner','nvml_stages','check','model'})
 def test_cached_worker_contamination_rejected(self):
  code="import sys,types; m=types.ModuleType('_pixel_t020_runtime_worker');m.__file__='/tmp/foreign.py';sys.modules[m.__name__]=m;import bridge\ntry:bridge.prepare_cpu()\nexcept ValueError:pass\nelse:raise AssertionError('foreign module accepted')"
  p=subprocess.run([sys.executable,'-c',code],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
  self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()
