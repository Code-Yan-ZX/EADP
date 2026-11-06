import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_gate
class GateTests(unittest.TestCase):
    def test_incomplete_acceptance_stops_before_journal_or_gt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=SimpleNamespace(run=p,proof=p/'proof',verify_started=p/'start',verify_terminal=p/'end')
            with patch.object(score_gate.runtime_gate,'supervised'),patch.object(score_gate,'read',return_value={'status':'failed'}), \
                 patch.object(score_gate.verify,'lines',side_effect=AssertionError('prediction opened')):
                with self.assertRaisesRegex(ValueError,'acceptance'):score_gate.gate(a)
if __name__=='__main__':unittest.main()
