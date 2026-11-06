import sys,subprocess,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_no_prelaunch_proof_rejects_before_result_or_output(self):
  if (D/'analysis_plan001.json').exists() or (D/'analysis001.json').exists():self.skipTest('historical run exists')
  p=subprocess.run([sys.executable,str(D/'analysis_entry.py')],cwd=D,capture_output=True,text=True,timeout=10)
  self.assertNotEqual(p.returncode,0);self.assertIn('analysis source gate locked',p.stderr);self.assertFalse((D/'analysis001.json').exists())
if __name__=='__main__':unittest.main()
