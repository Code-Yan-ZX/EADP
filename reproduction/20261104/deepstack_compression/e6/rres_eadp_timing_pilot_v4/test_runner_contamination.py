import os,subprocess,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_foreign_and_same_path_spoof_are_rejected(self):
  for claimed in ('/tmp/foreign-paired-runner.py',str(HERE/'paired_runner.py')):
   script='import sys,types;sys.path.insert(0,sys.argv[1]);m=types.ModuleType("paired_runner");m.__file__=sys.argv[2];m.measure=lambda *a,**k:None;m.normalize=lambda *a,**k:None;sys.modules["paired_runner"]=m;import qwen_request'
   p=subprocess.run([sys.executable,'-c',script,str(HERE),claimed],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},timeout=30)
   self.assertNotEqual(p.returncode,0);self.assertIn('foreign paired runner',p.stderr)
