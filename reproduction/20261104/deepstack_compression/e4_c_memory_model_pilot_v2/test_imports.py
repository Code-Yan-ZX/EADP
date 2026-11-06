import os,subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_preloaded_foreign_modules_rejected(self):
  code="import sys,types,runpy;sys.path.insert(0,sys.argv[1]);m=types.ModuleType(sys.argv[2]);m.__file__='/tmp/foreign.py';sys.modules[sys.argv[2]]=m;sys.argv=sys.argv[3:];runpy.run_path(sys.argv[0],run_name='__probe__')"
  for module,target in (('imports','gate.py'),('identity_gate','gate.py'),('observe','model_run.py'),('summary','controller.py'),('latency_worker','controller.py'),('contract','sampler.py')):
   with self.subTest(module=module):
    r=subprocess.run([sys.executable,'-c',code,str(D),module,str(D/target)],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=20)
    self.assertNotEqual(r.returncode,0);self.assertIn('foreign',r.stderr)
 def test_worker_refuses_preloaded_fake_model_dependencies(self):
  code="import sys,types,runpy;sys.path.insert(0,sys.argv[1]);import gate;gate.review=lambda **kw:{};m=types.ModuleType('latency_worker');m.__file__='/tmp/foreign.py';sys.modules['latency_worker']=m;sys.argv=[sys.argv[2],'--out','unused','--preflight-only'];runpy.run_path(sys.argv[0],run_name='__main__')"
  r=subprocess.run([sys.executable,'-c',code,str(D),str(D/'worker.py')],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=20)
  self.assertNotEqual(r.returncode,0);self.assertIn('foreign cached module: latency_worker',r.stderr)
 def test_light_adapter_rejected_before_invocation(self):
  code="import sys,types;sys.path.insert(0,sys.argv[1]);m=types.ModuleType('light_adapter');m.__file__='/tmp/foreign.py';sys.modules['light_adapter']=m;import model_run;model_run.real_model(None,None,None,None,None,None,None,None,None)"
  r=subprocess.run([sys.executable,'-c',code,str(D)],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=20)
  self.assertNotEqual(r.returncode,0);self.assertIn('foreign cached module: light_adapter',r.stderr)
 def test_real_controller_sampler_imports_no_gpu(self):
  for mode in ('controller','sampler'):
   r=subprocess.run([sys.executable,str(D/'import_probe.py'),'--mode',mode],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=20)
   self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout.strip(),'import_probe_passed '+mode)
if __name__=='__main__':unittest.main(verbosity=2)
