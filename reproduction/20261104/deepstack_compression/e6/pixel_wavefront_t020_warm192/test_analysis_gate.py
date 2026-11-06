import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import analysis
class AnalysisGateTest(unittest.TestCase):
 def test_missing_prelaunch_rejects_without_formal_read(self):
  with tempfile.TemporaryDirectory() as td,patch.object(analysis,'D',Path(td)):
   with self.assertRaises(FileNotFoundError):analysis.admit()
if __name__=='__main__':unittest.main()
