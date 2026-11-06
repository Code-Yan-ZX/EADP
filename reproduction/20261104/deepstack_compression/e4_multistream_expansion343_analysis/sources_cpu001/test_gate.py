import sys,unittest
from unittest import mock
from pathlib import Path
import gate
class TestGate(unittest.TestCase):
 def test_absent_prelaunch_blocks_before_prior_or_343(self):
  with mock.patch.object(gate,'prelaunch',side_effect=ValueError('absent review')),mock.patch.object(gate,'terminal') as term:
   with self.assertRaises(ValueError):gate.accepted()
   term.assert_not_called()
 def test_no_independent_proof_rejected(self):
  with mock.patch.object(gate,'read',return_value={}),mock.patch.object(gate,'terminal'),mock.patch.object(gate,'sha',return_value='fake'):
   with self.assertRaises((ValueError,KeyError)):gate.prelaunch()
 def test_dual_proofs_and_missing_records(self):
  expected={str(i) for i in range(686)};files={k:'digest' for k in expected};f={'status':'passed','images':343,'GT_opened':False,'scoring_performed':False,'files_sha256':files};q={'status':'passed','independent_review':True,'images':343,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'errors':[],'files_sha256':dict(files),'formal_sha256':'proof'}
  self.assertEqual(gate.validate_full_proofs(f,q,{'formal_sha256':'proof'},expected),files)
  for mut in ({**q,'images':342},{**q,'formal_sha256':'wrong'},{**q,'files_sha256':{}},{**q,'independent_review':False}):
   with self.assertRaises(ValueError):gate.validate_full_proofs(f,mut,{'formal_sha256':'proof'},expected)
  with self.assertRaises(ValueError):gate.validate_full_proofs({**f,'files_sha256':{}},{**q,'files_sha256':{}},{'formal_sha256':'proof'},expected)
if __name__=='__main__':unittest.main()
