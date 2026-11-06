import os,subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Locks(unittest.TestCase):
 def test_runid(self):
  r=subprocess.run([sys.executable,str(D/'launch.py'),'--run-id','002'],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15);self.assertNotEqual(r.returncode,0);self.assertIn('unexpected run id',r.stderr)
 def test_worker_gpu_locked(self):
  r=subprocess.run([sys.executable,str(D/'worker.py'),'--out',str(D/'run001')],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15);self.assertNotEqual(r.returncode,0);self.assertTrue('cpu_check002.json' in r.stderr or 'independent review missing' in r.stderr)
 def test_formal_cannot_run_early(self):
  for script,out in [('snapshot.py','run001_sources'),('verify.py','verification001.json')]:
   r=subprocess.run([sys.executable,str(D/script),'--out',str(D/out)],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15);self.assertNotEqual(r.returncode,0);self.assertFalse((D/out).exists())
if __name__=='__main__':unittest.main()
