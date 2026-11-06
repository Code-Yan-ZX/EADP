import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import worker, fixed_reference_gate as fixed
class Tests(unittest.TestCase):
 def tearDown(self):
  for name in ('_timing192_gpu_gate','_timing192_collision'):sys.modules.pop(name,None)
 def test_worker_missing_output_rejected_before_gpu(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(sys,'argv',['worker.py','--out',str(Path(temp)/'missing')]):
   with self.assertRaisesRegex(ValueError,'launcher output missing'):worker.main()
   self.assertFalse((Path(temp)/'missing').exists())
 def test_candidate_rejects_missing_plan_before_model_or_references(self):
  with tempfile.TemporaryDirectory() as temp:
   with self.assertRaises((ValueError,FileNotFoundError)):worker.execute_candidate(Path(temp))
   self.assertEqual(list(Path(temp).iterdir()),[])
 def test_foreign_preloaded_gate_rejected(self):
  with patch.dict(sys.modules,{'_timing192_collision':object()}):
   with self.assertRaisesRegex(ValueError,'preloaded'):worker.load('_timing192_collision','gate.py')
 def test_missing_888_proof_never_reads_records(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(fixed,'I',Path(temp)/'review'),patch.object(fixed,'A',Path(temp)/'formal'),patch.object(fixed,'RUN',Path(temp)/'run'):
   with self.assertRaisesRegex(ValueError,'acceptance missing'):fixed.load({})
if __name__=='__main__':unittest.main()
