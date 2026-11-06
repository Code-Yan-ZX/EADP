import os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import gate,preflight_entry,preflight_launch
class Tests(unittest.TestCase):
 def test_gpu_and_missing_review_fail_closed(self):
  with self.assertRaises(RuntimeError):gate.gpu()
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'D',Path(d)):
   with self.assertRaises(ValueError):gate.preflight()
 def test_real_launcher_does_not_create_output_before_approval(self):
  output=preflight_launch.D/'run_preflight001';fake=types.SimpleNamespace(preflight=lambda:(_ for _ in ()).throw(ValueError('blocked')))
  with patch.object(preflight_launch,'load',return_value=fake),patch.object(sys,'argv',['preflight_launch.py','--out',str(output)]),patch.object(Path,'mkdir') as mkdir:
   with self.assertRaisesRegex(ValueError,'blocked'):preflight_launch.main()
   mkdir.assert_not_called()
 def test_upstream_source_tampering_rejected(self):
  import hashlib
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'source';p.write_text('safe');proof={'source_sha256_after':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()}}
   self.assertEqual(gate.verified_prior(proof),proof['source_sha256_after']);p.write_text('changed')
   with self.assertRaisesRegex(ValueError,'source changed'):gate.verified_prior(proof)
 def test_reject_foreign_cached_model(self):
  x=types.ModuleType('_foreign_timing_test');x.__file__='/tmp/foreign.py'
  with patch.dict(sys.modules,{'_foreign_timing_test':x}):
   with self.assertRaises(ValueError):preflight_entry.load('_foreign_timing_test',Path(__file__))
if __name__=='__main__':unittest.main()
