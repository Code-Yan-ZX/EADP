import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gate
class Tests(unittest.TestCase):
 def test_gpu_locked(self):
  with self.assertRaises(RuntimeError):gate.admit_gpu()
 def test_absent_plan_refuses(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'D',Path(d)):
   with self.assertRaises(FileNotFoundError):gate.admit_preflight()
if __name__=='__main__':unittest.main()
