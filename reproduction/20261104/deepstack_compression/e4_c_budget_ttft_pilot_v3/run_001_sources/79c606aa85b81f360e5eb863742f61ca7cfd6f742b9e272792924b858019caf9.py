import importlib.util,sys,tempfile,types,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('budget_preflight_tested',HERE/'preflight.py');preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(preflight)

class Tests(unittest.TestCase):
 def test_foreign_preloaded_module_rejected(self):
  name='_budget_test_foreign';old=types.ModuleType(name);old.__file__='/tmp/foreign.py';sys.modules[name]=old
  try:
   with self.assertRaisesRegex(ValueError,'foreign module'):preflight.load(name,HERE/'adapter.py')
  finally:del sys.modules[name]
 def test_source_mutation_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'source';path.write_text('changed')
   with self.assertRaisesRegex(ValueError,'source mismatch'):preflight.check_sources({'source_sha256':{str(path):'0'*64}})
if __name__=='__main__':unittest.main()
