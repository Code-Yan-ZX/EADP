import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import runtime_admission as a
class RuntimeAdmissionTest(unittest.TestCase):
 def test_missing_independent_review_fails_before_preflight(self):
  with tempfile.TemporaryDirectory() as d,patch.object(a,'REVIEW',Path(d)/'missing'):
   with self.assertRaisesRegex(ValueError,'preflight review required'):a.admit()
 def test_prior_preflight_provenance(self):
  proof=a.prior();self.assertEqual(proof['status'],'passed');self.assertFalse(proof['gpu_used']);self.assertEqual(len(proof['approval']['source_sha256']),59)
if __name__=='__main__':unittest.main()
