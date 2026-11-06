"""Independent actual identity worker import ordering, CPU-only."""
import os,subprocess,sys,unittest
from pathlib import Path
B=Path(__file__).resolve().parents[2];D=B/'e6'/os.environ.get('IDENTITY_TIMING_CANDIDATE','rres_eadp_identity3_proposal');T=B/'e6/rres_eadp_timing_pilot_v4'
@unittest.skipUnless((D/'timing_imports.py').is_file(),'identity loader not yet available')
class Tests(unittest.TestCase):
 def run_code(self,code):
  return subprocess.run([sys.executable,'-c',code,str(D),str(T)],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),text=True,capture_output=True,timeout=180)
 def test_reference_first_actual_worker_bindings(self):
  code='''import sys
from pathlib import Path
d,t=map(Path,sys.argv[1:]);sys.path[:0]=[str(d),str(t)]
import reference_inputs
r=reference_inputs.load()
import worker,runner
request,core,paired=worker.timing_imports.load()
assert Path(runner.__file__).parent.name=='e4_c_latency'
assert request.runner is paired and core.runner is paired and core.qwen_request is request
assert Path(paired.__file__).resolve()==(t/'paired_runner.py').resolve()
raw=[];ticks=iter([10,20,30,40])
class X:shape=(1,1)
def invoke(boundary):
 boundary(X(),None);boundary.generation_returned([7]);return 'text'
response,b=request.runner.measure(invoke,lambda:None,clock=lambda:next(ticks),persist=raw.append)
assert raw==[[7]] and response=='text' and b['first_ns']==20
assert len(r['rows'])==62 and len(r['references'])==124
'''
  p=self.run_code(code);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
 def test_preloaded_timing_runner_fails_closed(self):
  code='''import sys,types
from pathlib import Path
d,t=map(Path,sys.argv[1:]);sys.path[:0]=[str(d),str(t)]
import worker
sys.modules['paired_runner']=types.ModuleType('paired_runner')
try:worker.timing_imports.load()
except ValueError as e:assert 'preloaded' in str(e)
else:raise AssertionError('pollution accepted')
'''
  p=self.run_code(code);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
if __name__=='__main__':unittest.main()
