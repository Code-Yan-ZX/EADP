import sys
import unittest
from unittest.mock import patch
import score_panel as s

class GateTests(unittest.TestCase):
    def test_failed_source_review_never_reads_truth_or_records(self):
        args=['score_panel','--run',str(s.DEV/'run_001'),'--out',str(s.HERE/'never_created')]
        with patch.object(sys,'argv',args),patch.object(s,'source_review',side_effect=ValueError('unreviewed')),patch.object(s.e1_score,'read_truths') as truth,patch.object(s,'lines') as rows:
            with self.assertRaisesRegex(ValueError,'unreviewed'):s.main()
            truth.assert_not_called();rows.assert_not_called()
    def test_missing_full_panel_fails_before_predictions(self):
        with patch.object(s.gate,'terminal',side_effect=ValueError('missing full panel')),patch.object(s,'lines') as rows:
            with self.assertRaisesRegex(ValueError,'missing full panel'):s.accepted_prior(s.DEV/'run_001')
            rows.assert_not_called()
    def test_accepted_e1_score_sources_and_tampering(self):
        s.accepted_e1()
        original=s.gate.sha;target=s.metrics.VQA.resolve()
        def changed(path):
            return '0'*64 if __import__('pathlib').Path(path).resolve()==target else original(path)
        with patch.object(s.gate,'sha',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'E1 scoring source changed'):s.accepted_e1()
    def test_scoring_runtime_change_rejected(self):
        with patch.object(s.importlib.metadata,'version',return_value='different'):
            with self.assertRaisesRegex(ValueError,'E1 scoring runtime changed'):s.accepted_e1()
    def test_import_path_collision_rejected(self):
        for module in (s.gate,s.metrics,s.e1_score,s.core):
            with patch.object(module,'__file__','/wrong/collision.py'):
                with self.assertRaisesRegex(ImportError,'source mismatch'):s.validate_imports()
if __name__=='__main__':unittest.main()
