import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gate
class GateTests(unittest.TestCase):
 def test_approval_missing_is_locked(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'APPROVAL',Path(d)/'none'):
   with self.assertRaisesRegex(ValueError,'locked'):gate.check(preflight_only=True)
 def test_deadline_cannot_be_extended(self):
  with patch.object(gate.time,'time',return_value=1794065414-100):
   with self.assertRaisesRegex(ValueError,'deadline'):gate.budget(True)
 def test_final_and_preflight_need_different_approval_inventory(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'approval.json';p.write_text('{}')
   with patch.object(gate,'APPROVAL',p):
    with self.assertRaisesRegex(ValueError,'inventory'):gate.check(preflight_only=False)
    with self.assertRaisesRegex(ValueError,'inventory'):gate.check(preflight_only=True)
if __name__=='__main__':unittest.main()
