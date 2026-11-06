import importlib.util,inspect,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import runtime_gate as gate
class Tests(unittest.TestCase):
 def test_gpu_locked(self):
  with self.assertRaisesRegex(RuntimeError,'GPU LOCKED'):gate.require_gpu()
 def test_source_tamper_rejected(self):
  p={'stage':'preflight_only','GPU_authorized':False,'GT_opened':False,'predictions_opened':True,'manifest_sha256':'ok','random_plan_sha256':'ok','source_sha256':{'/frozen':'expected'}}
  with patch.object(gate,'read',return_value=p),patch.object(gate,'sha',side_effect=lambda p:'bad' if str(p)=='/frozen' else 'ok'):
   with self.assertRaisesRegex(ValueError,'source changed'):gate.verify_plan()
 def test_missing_review_fails_closed(self):
  with patch.object(gate,'verify_plan',return_value={'source_sha256':{}}),patch.object(gate,'terminal',side_effect=FileNotFoundError):
   with self.assertRaises(FileNotFoundError):gate.require_review()
 def test_dynamic_real_class_registered_and_inspectable(self):
  spec=importlib.util.spec_from_file_location('random_pf_test2',Path(__file__).with_name('preflight_worker.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'realclass.py';path.write_text('class RandomAdapter:\n    pass\n');name='random_inspection_test_module';loaded=m.load_by_spec(name,path)
   try:
    self.assertEqual(Path(inspect.getfile(loaded.RandomAdapter)).resolve(),path.resolve())
    with self.assertRaisesRegex(ValueError,'preloaded'):m.load_by_spec(name,path)
   finally:sys.modules.pop(name,None)
 def test_foreign_selector_cache_rejected(self):
  spec=importlib.util.spec_from_file_location('random_pf_test',Path(__file__).with_name('preflight_worker.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  fake=types.ModuleType('selector');fake.__file__='/tmp/not_selector.py'
  with patch.dict(sys.modules,{'selector':fake}):
   with self.assertRaisesRegex(ValueError,'foreign cached'):m.checked('selector','/expected/selector.py')
if __name__=='__main__':unittest.main(verbosity=2)
