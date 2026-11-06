import os,subprocess,sys,tempfile,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Locks(unittest.TestCase):
 def test_real_worker_without_review_rejects_before_cuda(self):
  with tempfile.TemporaryDirectory() as x:
   run=subprocess.run([sys.executable,str(D/'pilot_worker.py'),'--out',x,'--preflight-only'],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15)
   self.assertNotEqual(run.returncode,0);self.assertFalse((Path(x)/'worker_preflight.json').exists());self.assertTrue('cpu_check001.json' in run.stderr or 'independent review missing' in run.stderr)
 def test_preflight_rejects_unplanned_id(self):
  run=subprocess.run([sys.executable,str(D/'preflight_launch.py'),'--run-id','002'],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=15)
  self.assertNotEqual(run.returncode,0);self.assertIn('unexpected run id',run.stderr)
if __name__=='__main__':unittest.main()
