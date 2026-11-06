import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gate
class GateTests(unittest.TestCase):
 def test_missing_approval_locks_panel(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'APPROVAL',Path(d)/'missing'):
   with self.assertRaisesRegex(ValueError,'locked'):gate.check(preflight_only=True)
 def test_deadline_rejects_insufficient_remaining_time(self):
  with patch.object(gate.time,'time',return_value=1794065414-100):
   with self.assertRaisesRegex(ValueError,'budget'):gate.check(preflight_only=True,require_future=True)
 def test_actual_smoke_proofs_reject_tampered_formal_binding(self):
  real=gate.read;target=gate.REVIEW/'selected_smoke001_review.json'
  def altered(path):
   result=real(path)
   if Path(path)==target:result=copy.deepcopy(result);result['formal_proof_sha256']='0'*64
   return result
  self.assertTrue(gate.upstream()['smoke_review_sha256'])
  with patch.object(gate,'read',side_effect=altered):
   with self.assertRaisesRegex(ValueError,'identity evidence'):gate.upstream()
if __name__=='__main__':unittest.main()
