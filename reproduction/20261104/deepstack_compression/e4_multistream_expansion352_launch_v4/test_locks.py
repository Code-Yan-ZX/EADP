import os,subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Locks(unittest.TestCase):
 def test_run_id_rejected(self):
  p=subprocess.run([sys.executable,str(D/'launch.py'),'--run-id','002'],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15);self.assertNotEqual(p.returncode,0);self.assertIn('run id',p.stderr)
 def test_no_review_no_gpu(self):
  p=subprocess.run([sys.executable,str(D/'launch.py'),'--run-id','001'],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15);self.assertNotEqual(p.returncode,0)
 def test_formal_locked(self):
  for script,out in [('snapshot.py','run001_sources'),('verify.py','verification001.json')]:
   p=subprocess.run([sys.executable,str(D/script),'--out',str(D/out)],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15);self.assertNotEqual(p.returncode,0);self.assertFalse((D/out).exists())
if __name__=='__main__':unittest.main()
