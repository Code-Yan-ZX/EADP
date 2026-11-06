import importlib.util
from pathlib import Path
import subprocess,sys,unittest

HERE=Path(__file__).resolve().parent
class PinnedTests(unittest.TestCase):
 def test_preloaded_conflicting_adapter_refused(self):
  script="import sys,types; m=types.ModuleType('adapter'); m.__file__='/tmp/wrong.py';sys.modules['adapter']=m; import local_imports; local_imports.prepare()"
  p=subprocess.run([sys.executable,'-c',script],cwd=HERE,capture_output=True,text=True)
  self.assertNotEqual(p.returncode,0);self.assertIn('conflicting import adapter',p.stderr)
 def test_unique_contract_coexists_with_e4(self):
  script="import local_imports; a,b=local_imports.prepare();import smoke_contract; assert a.__file__.endswith('selected_smoke/stock_capture.py');assert b.__file__.endswith('selected_smoke/smoke_contract.py');assert smoke_contract.__file__.endswith('/e4/smoke_contract.py')"
  p=subprocess.run([sys.executable,'-c',script],cwd=HERE,capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)
 def test_preloaded_wrong_transitive_sources_are_refused(self):
  for name in ('skip_deepstack','completion','identity_worker','verify_k256'):
   script=f"import sys,types; m=types.ModuleType({name!r}); m.__file__='/tmp/wrong.py';sys.modules[{name!r}]=m; import local_imports; local_imports.prepare()"
   p=subprocess.run([sys.executable,'-c',script],cwd=HERE,capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0);self.assertIn('conflicting import '+name,p.stderr)
if __name__=='__main__':unittest.main()
