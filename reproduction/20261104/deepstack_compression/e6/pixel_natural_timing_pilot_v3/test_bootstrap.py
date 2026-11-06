import importlib.util,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_preloaded_same_name_refused_before_entry(self):
  for name in ('pilot_contract','observer','timer','runner','contract','input_preflight','identity_gate','real_worker','light_adapter','skip_deepstack','timing_launch','worker','preflight'):
   entries=['worker.py','preflight.py','preflight_launch.py'] if name not in ('worker','preflight') else (['preflight.py'] if name=='worker' else ['preflight_launch.py'])
   for entry in entries:
    code='import sys,types,runpy; m=types.ModuleType('+repr(name)+'); m.__file__="/wrong/'+name+'.py"; sys.modules['+repr(name)+']=m; runpy.run_path('+repr(str(HERE/entry))+',run_name="not_main")'
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
    self.assertNotEqual(r.returncode,0,(name,entry));self.assertIn('foreign preloaded module',r.stderr,(name,entry,r.stderr))
 def test_correct_module_pin_reuses_object(self):
  spec=importlib.util.spec_from_file_location('boot_test',HERE/'bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot)
  a=boot.pin('pilot_contract',HERE/'pilot_contract.py');self.assertIs(a,boot.pin('pilot_contract',HERE/'pilot_contract.py'))
if __name__=='__main__':unittest.main(verbosity=2)
