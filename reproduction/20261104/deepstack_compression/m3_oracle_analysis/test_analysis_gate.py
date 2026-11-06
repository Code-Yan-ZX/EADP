import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis_gate
class AnalysisGateTests(unittest.TestCase):
    def test_missing_independent_approval_is_locked(self):
        with tempfile.TemporaryDirectory() as d,patch.object(analysis_gate,'APPROVAL',Path(d)/'missing'):
            with self.assertRaisesRegex(ValueError,'locked'):analysis_gate.validate()
if __name__=='__main__':unittest.main()
