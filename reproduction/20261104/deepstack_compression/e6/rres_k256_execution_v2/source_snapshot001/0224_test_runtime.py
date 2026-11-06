import os,subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/rres_k256_execution_v2'
class Tests(unittest.TestCase):
 def test_real_worker_generation_controls_and_locked_gate(self):
  r=subprocess.run([sys.executable,'-m','unittest','-v','test_execution'],cwd=E,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
  self.assertEqual(r.returncode,0,r.stdout+r.stderr)
 def test_real_cpu_launcher_refuses_without_plan_before_creating_container(self):
  out=D/'run001';self.assertFalse(out.exists())
  r=subprocess.run([sys.executable,str(D/'launch.py'),'--out',str(out)],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15)
  self.assertNotEqual(r.returncode,0);self.assertIn('plan001.json',r.stderr);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
