import subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class WorkerLockTests(unittest.TestCase):
 def test_worker_and_launcher_remain_locked_without_reading_inputs(self):
  with tempfile.TemporaryDirectory() as d:
   output=Path(d)/'never_created'
   for script,args in [('panel_worker.py',['--execute','--preflight-only','--out',str(output)]),('panel_launch.py',['--run-id','locked'])]:
    cmd=[sys.executable,str(HERE/script),*args,'--e1-acceptance','/missing','--e1-started','/missing','--e1-terminal','/missing']
    p=subprocess.run(cmd,text=True,capture_output=True,timeout=15)
    self.assertNotEqual(p.returncode,0);self.assertIn('GPU locked',p.stderr);self.assertFalse(output.exists())
 def test_verifier_import_chain_and_same_name_contamination(self):
  p=subprocess.run([sys.executable,str(HERE/'panel_verify.py'),'--help'],text=True,capture_output=True,timeout=15);self.assertEqual(p.returncode,0,p.stderr)
  source="import sys,importlib.util;from pathlib import Path;h=Path(sys.argv[1]);p=h.parent/'e1'/'contract.py';spec=importlib.util.spec_from_file_location('contract',p);m=importlib.util.module_from_spec(spec);sys.modules['contract']=m;spec.loader.exec_module(m);sys.path.insert(0,str(h));from full_imports import local;assert Path(local('contract').__file__).resolve()==(h/'contract.py').resolve();local('panel_worker');local('panel_runner')"
  p=subprocess.run([sys.executable,'-c',source,str(HERE)],text=True,capture_output=True,timeout=15);self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()
