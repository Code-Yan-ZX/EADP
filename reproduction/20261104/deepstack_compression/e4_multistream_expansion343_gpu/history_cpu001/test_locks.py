import os,subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Locks(unittest.TestCase):
 def test_runid(self):
  r=subprocess.run([sys.executable,str(D/'launch.py'),'--run-id','002'],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15);self.assertNotEqual(r.returncode,0);self.assertIn('unexpected run id',r.stderr)
 def test_worker_gpu_locked(self):
  r=subprocess.run([sys.executable,str(D/'worker.py'),'--out',str(D/'run001')],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15);self.assertNotEqual(r.returncode,0);self.assertTrue('cpu_check001.json' in r.stderr or 'independent review missing' in r.stderr)
if __name__=='__main__':unittest.main()
