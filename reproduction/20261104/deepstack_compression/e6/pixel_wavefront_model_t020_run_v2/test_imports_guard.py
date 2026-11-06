import sys,types,unittest
from unittest.mock import patch
from imports_guard import LATENCY,check
class Tests(unittest.TestCase):
 def test_preloaded_transitive_dependencies_rejected(self):
  for name,path in LATENCY.items():
   fake=types.ModuleType(name);fake.__file__='/wrong/'+name+'.py'
   with patch.dict(sys.modules,{name:fake}):
    with self.assertRaisesRegex(ValueError,'foreign module'):check({name:path})
 def test_missing_required_rejected(self):
  with patch.dict(sys.modules,{},clear=True):
   with self.assertRaisesRegex(ValueError,'missing imported'):check({'unknown':'/any'},True)
if __name__=='__main__':unittest.main(verbosity=2)
