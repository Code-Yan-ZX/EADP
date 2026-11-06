import sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import preflight_entry as entry
class Tests(unittest.TestCase):
 def test_cached_foreign_module_rejected_before_call(self):
  for name in ('gpu_admission','upstream_identity','worker_entry','single_arm','schedule_contract'):
   fake=types.ModuleType(name);fake.__file__='/wrong/'+name+'.py'
   with patch.dict(sys.modules,{name:fake}):
    with self.assertRaisesRegex(ValueError,'foreign cached'):entry.load_exact(name,Path(entry.D)/(name+'.py'))
if __name__=='__main__':unittest.main(verbosity=2)
