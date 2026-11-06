import os,subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_controller_and_sampler_locked_before_gpu(self):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
  code="import sys,runpy;sys.path.insert(0,sys.argv[1]);import gate;gate.review=lambda **kw:(_ for _ in ()).throw(ValueError('synthetic admission denied'));sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')"
  for args in ([str(D/'controller.py'),'--run-id','001'],[str(D/'sampler.py'),'--directory',str(D/'run_001')]):
   r=subprocess.run([sys.executable,'-c',code,str(D),*args],capture_output=True,text=True,env=env,timeout=30)
   self.assertNotEqual(r.returncode,0);self.assertIn('synthetic admission denied',r.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
