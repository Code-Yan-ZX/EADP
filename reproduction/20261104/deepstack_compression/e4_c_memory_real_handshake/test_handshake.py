"""No-GPU fail-closed checks for the real sampling handshake."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class TestHandshake(unittest.TestCase):
 def test_worker_preflight_really_imports_without_cuda(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'pre.json';env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
   p=subprocess.run([sys.executable,str(D/'worker.py'),'--preflight-only','--out',str(out)],env=env,capture_output=True,text=True,timeout=90)
   self.assertEqual(p.returncode,0,p.stderr);r=json.loads(out.read_text());self.assertEqual(r['status'],'passed');self.assertIs(r['cuda_initialized'],False);self.assertIs(r['gpu_used'],False);self.assertIs(r['model_loaded'],False)
 def test_direct_worker_refuses_before_cuda(self):
  with tempfile.TemporaryDirectory() as td:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='');p=subprocess.run([sys.executable,str(D/'worker.py'),'--out',td,'--nonce','0'*32],env=env,capture_output=True,text=True,timeout=90)
   self.assertNotEqual(p.returncode,0);self.assertFalse((Path(td)/'allocation.json').exists())
 def test_controller_requires_review_before_directory_or_gpu(self):
  p=subprocess.run([sys.executable,str(D/'controller.py'),'--run-id','001'],capture_output=True,text=True,timeout=90)
  self.assertNotEqual(p.returncode,0);self.assertFalse((D/'run_001').exists());self.assertNotIn('nvidia-smi',p.stderr)
 def test_sampler_rejects_unbound_output(self):
  with tempfile.TemporaryDirectory() as td:
   p=subprocess.run([sys.executable,str(D/'sampler.py'),'--directory',td],capture_output=True,text=True,timeout=90)
   self.assertNotEqual(p.returncode,0);self.assertFalse((Path(td)/'samples.jsonl').exists())
if __name__=='__main__':unittest.main(verbosity=2)
