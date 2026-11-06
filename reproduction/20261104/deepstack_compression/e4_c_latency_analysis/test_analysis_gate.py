import sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import analyze
class GateTest(unittest.TestCase):
 def test_missing_independent_review_rejected(self):
  with tempfile.TemporaryDirectory() as d,patch.object(analyze.gate,'PRE',Path(d)/'missing'):
   with self.assertRaisesRegex(ValueError,'prelaunch review'):analyze.gate.prelaunch()
 def test_main_stops_before_records_if_gate_fails(self):
  with patch.object(sys,'argv',['analyze.py','--out','unused']),patch.object(analyze.gate,'admit',side_effect=ValueError('not approved')),patch.object(analyze,'lines',side_effect=AssertionError('records opened')):
   with self.assertRaisesRegex(ValueError,'not approved'):analyze.main()
 def test_wrong_source_does_not_execute(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.py';p.write_text('raise RuntimeError("executed")')
   with self.assertRaisesRegex(ValueError,'source changed'):analyze.load_exact('fake_analysis',p,'0'*64)
 def test_import_collision_fails_before_execute(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'module.py';p.write_text('x=1');sys.modules['fake_analysis']=types.ModuleType('fake_analysis')
   try:
    with self.assertRaisesRegex(ValueError,'import collision'):analyze.load_exact('fake_analysis',p,analyze.gate.sha(p))
   finally:del sys.modules['fake_analysis']
if __name__=='__main__':unittest.main()
