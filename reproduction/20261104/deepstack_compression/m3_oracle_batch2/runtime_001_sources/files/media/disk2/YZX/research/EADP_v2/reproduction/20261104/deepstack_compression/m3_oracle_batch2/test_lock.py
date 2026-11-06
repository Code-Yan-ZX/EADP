"""The batch-2 draft must fail before opening input or initializing CUDA."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import runtime_worker
class LockedEntry(unittest.TestCase):
 def test_default_and_preflight_are_both_locked_before_file_read(self):
  for extra in ([],['--preflight-only']):
   with self.subTest(extra=extra),patch.object(sys,'argv',['runtime_worker.py','--execute','--out','/unused',*extra]),patch.object(Path,'read_text',side_effect=AssertionError('unexpected file read')):
    with self.assertRaisesRegex(ValueError,'batch-2 locked'):runtime_worker.main()
if __name__=='__main__':unittest.main()
