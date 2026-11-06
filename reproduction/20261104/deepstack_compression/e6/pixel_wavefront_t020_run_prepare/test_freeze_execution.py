import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import freeze_execution as f
class Tests(unittest.TestCase):
 def test_missing_real_preflight_result_blocks_before_source_access(self):
  with tempfile.TemporaryDirectory() as d,patch.object(f,'I',Path(d)),patch.object(f,'load',side_effect=AssertionError('source opened before proof')):
   with self.assertRaisesRegex(ValueError,'no-GPU result review required'):f.freeze()
if __name__=='__main__':unittest.main(verbosity=2)
