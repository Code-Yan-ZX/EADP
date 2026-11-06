import subprocess,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class ImportTests(unittest.TestCase):
 def test_poisoned_generic_full_runner_does_not_shadow_exact_local(self):
  code="import sys,types; m=types.ModuleType('full_runner');m.__file__='/tmp/wrong.py';sys.modules['full_runner']=m;from full_imports import load,HERE;r=load('eadp_selected_full_runner',HERE/'full_runner.py');assert r.__file__.endswith('selected_full/full_runner.py')"
  p=subprocess.run([sys.executable,'-c',code],cwd=HERE,text=True,capture_output=True);self.assertEqual(p.returncode,0,p.stderr)
 def test_poisoned_unique_module_fails_closed(self):
  code="import sys,types; m=types.ModuleType('eadp_selected_full_runner');m.__file__='/tmp/wrong.py';sys.modules[m.__name__]=m;from full_imports import load,HERE;load('eadp_selected_full_runner',HERE/'full_runner.py')"
  p=subprocess.run([sys.executable,'-c',code],cwd=HERE,text=True,capture_output=True);self.assertNotEqual(p.returncode,0);self.assertIn('source collision',p.stderr)
 def test_complete_pre_model_import_chain_ignores_generic_conflicts(self):
  code="""import sys,types
for name in ('full_runner','freeze_plan','gate','verify','stock_capture'):
 m=types.ModuleType(name);m.__file__='/tmp/wrong_'+name+'.py';sys.modules[name]=m
from full_imports import load,HERE,prepare
prepare()
for name in ('worker','preflight','launcher','verify'):
 module=load('eadp_full_test_'+name,HERE/(name+'.py'))
 assert module.__file__==str(HERE/(name+'.py'))
"""
  p=subprocess.run([sys.executable,'-c',code],cwd=HERE,text=True,capture_output=True);self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()
