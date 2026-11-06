import subprocess,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class PinningTests(unittest.TestCase):
 def test_preloaded_module_rejected_before_entry(self):
  code="import sys,types;sys.modules['paired']=types.ModuleType('paired');import score_oracle"
  p=subprocess.run([sys.executable,'-c',code],cwd=HERE,text=True,capture_output=True,timeout=30)
  self.assertNotEqual(p.returncode,0);self.assertIn('preloaded or changed scoring dependency: paired',p.stderr)
if __name__=='__main__':unittest.main()
