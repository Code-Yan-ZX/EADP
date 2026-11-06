"""Exercise reference-first imports in CPU subprocesses; no model or GT."""
import os,subprocess,sys,unittest
from pathlib import Path
B=Path(__file__).resolve().parents[2];D=B/'e6/rres_eadp_timing_pilot_v4'
@unittest.skipUnless((D/'paired_runner.py').is_file(),'v4 not yet available')
class Tests(unittest.TestCase):
 def run_code(self,code):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  return subprocess.run([sys.executable,'-c',code,str(D)],cwd=B,env=env,text=True,capture_output=True,timeout=180)
 def test_real_references_then_request_module_and_measure(self):
  code='''import sys
from pathlib import Path
d=Path(sys.argv[1]);sys.path.insert(0,str(d))
import reference_inputs
r=reference_inputs.load()
assert len(r['rows'])==62 and len(r['references'])==124 and r['GT_opened'] is False
import qwen_request,process_core,process_worker,runner,paired_runner
assert Path(runner.__file__).name=='runner.py' and Path(runner.__file__).parent.name=='e4_c_latency'
assert Path(qwen_request.runner.__file__).resolve()==(d/'paired_runner.py').resolve()
assert process_core.runner is qwen_request.runner is paired_runner and process_worker.process_core is process_core
saved=[];ticks=iter([10,20,30,40])
class X: shape=(1,1)
def invoke(b):
 b(X(),None);b.generation_returned([42]);return 'text'
response,t=paired_runner.measure(invoke,lambda:None,clock=lambda:next(ticks),persist=saved.append)
assert response=='text' and saved==[[42]] and [t[k] for k in ('t0_ns','first_ns','generate_end_ns','response_end_ns')]==[10,20,30,40]
'''
  p=self.run_code(code);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
 def test_same_path_fake_module_is_rejected(self):
  code='''import sys,types
from pathlib import Path
d=Path(sys.argv[1]);sys.path.insert(0,str(d));fake=types.ModuleType('paired_runner');fake.__file__=str(d/'paired_runner.py')
def measure(*a,**kw):pass
def normalize(*a,**kw):pass
fake.measure=measure;fake.normalize=normalize;sys.modules['paired_runner']=fake
try:import qwen_request
except ValueError as e:assert 'foreign paired runner' in str(e)
else:raise AssertionError('same-path fake module accepted')
'''
  p=self.run_code(code);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
if __name__=='__main__':unittest.main()
