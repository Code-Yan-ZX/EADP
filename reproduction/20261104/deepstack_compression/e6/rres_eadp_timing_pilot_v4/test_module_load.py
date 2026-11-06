import inspect,sys,tempfile,unittest
from pathlib import Path
import module_load
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
class Tests(unittest.TestCase):
 def test_actual_adapter_class_inspection_and_preloaded_rejection(self):
  name='_paired_adapter_test';path=ROOT/'e4_c_budget_pilot_v2/adapter.py'
  try:
   m=module_load.load(name,path);self.assertEqual(Path(inspect.getfile(m.BudgetLightAdapter)).resolve(),path.resolve())
   with self.assertRaises(ValueError):module_load.load(name,path)
  finally:sys.modules.pop(name,None)
 def test_exception_removes_partial_module(self):
  name='_paired_bad_test'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.py';p.write_text('raise RuntimeError("bad")\n')
   with self.assertRaises(RuntimeError):module_load.load(name,p)
   self.assertNotIn(name,sys.modules)
