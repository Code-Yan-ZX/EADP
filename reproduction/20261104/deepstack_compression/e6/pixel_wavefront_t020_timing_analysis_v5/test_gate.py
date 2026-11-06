import sys,subprocess,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_no_prelaunch_proof_rejects_before_result_or_output(self):
  if (D/'analysis_plan001.json').exists() or (D/'analysis001.json').exists():self.skipTest('historical run exists')
  p=subprocess.run([sys.executable,str(D/'analysis_entry.py')],cwd=D,capture_output=True,text=True,timeout=10)
  self.assertNotEqual(p.returncode,0);self.assertIn('analysis source gate locked',p.stderr);self.assertFalse((D/'analysis001.json').exists())
if __name__=='__main__':unittest.main()

class OrderingTests(unittest.TestCase):
 def test_no_formal_open_when_verifier_not_exited(self):
  import gate,tempfile
  from unittest.mock import patch
  from pathlib import Path
  with tempfile.TemporaryDirectory() as t:
   a=Path(t)/'a';i=Path(t)/'i';a.mkdir();i.mkdir();(a/'verification001.json').touch();(i/'pixel_wavefront_t020_timing_result_review.json').touch()
   with patch.object(gate,'A',a),patch.object(gate,'I',i),patch.object(gate,'terminal',side_effect=ValueError('not exited')),patch.object(gate,'read',side_effect=AssertionError('timing prematurely read')):
    with self.assertRaises(ValueError):gate.result_admit()
