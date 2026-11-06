import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import preflight_admission as a
class Tests(unittest.TestCase):
 def test_missing_review_rejects_before_hash_or_gpu(self):
  with tempfile.TemporaryDirectory() as d,patch.object(a,'D',Path(d)),patch.object(a,'hashes',side_effect=AssertionError('hash read')):
   with self.assertRaisesRegex(ValueError,'review missing'):a.require_preflight()
if __name__=='__main__':unittest.main(verbosity=2)
