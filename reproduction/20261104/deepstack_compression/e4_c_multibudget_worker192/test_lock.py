import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import worker, fixed_reference_gate as fixed
class Tests(unittest.TestCase):
 def tearDown(self):
  for name in ('_timing192_gpu_lock','_timing192_entry_lock','_timing192_collision'):sys.modules.pop(name,None)
 def test_gpu_and_preflight_are_unconditionally_locked(self):
  for flag in ([],['--preflight-only']):
   with tempfile.TemporaryDirectory() as temp,patch.object(sys,'argv',['worker.py','--out',str(Path(temp)/'out'),*flag]):
    with self.assertRaisesRegex(RuntimeError,'locked'):worker.main()
    self.assertFalse((Path(temp)/'out').exists())
   sys.modules.pop('_timing192_entry_lock',None)
 def test_candidate_locked_before_loading_model_or_references(self):
  with tempfile.TemporaryDirectory() as temp:
   with self.assertRaisesRegex(RuntimeError,'GPU locked'):worker.execute_candidate(Path(temp))
   self.assertEqual(list(Path(temp).iterdir()),[])
 def test_foreign_preloaded_gate_rejected(self):
  with patch.dict(sys.modules,{'_timing192_collision':object()}):
   with self.assertRaisesRegex(ValueError,'preloaded'):worker.load('_timing192_collision','gate.py')
 def test_missing_888_proof_never_reads_records(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(fixed,'I',Path(temp)/'review'),patch.object(fixed,'A',Path(temp)/'formal'),patch.object(fixed,'RUN',Path(temp)/'run'):
   with self.assertRaisesRegex(ValueError,'acceptance missing'):fixed.load({})
if __name__=='__main__':unittest.main()
