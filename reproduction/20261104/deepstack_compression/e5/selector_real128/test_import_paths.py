"""Regress the preflight001 real import-chain collision."""
import os,subprocess,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_old_full_worker_preloaded_cannot_shadow_real_binding(self):
  code="""import runpy,sys,types;from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).parent));sys.modules['full_worker']=types.ModuleType('full_worker')
x=runpy.run_path(sys.argv[1],run_name='import_probe')
assert Path(x['bind_rows'].__code__.co_filename).resolve()==Path(sys.argv[2]).resolve()
"""
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1');r=subprocess.run([sys.executable,'-c',code,str(HERE/'worker.py'),str(HERE/'input_contract.py')],env=env,capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
 def test_poisoned_input_contract_rejected_before_model(self):
  code="""import runpy,sys,types;from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).parent));m=types.ModuleType('input_contract');exec('def bind_rows(*a): pass\\ndef compare_real(*a): pass',m.__dict__);sys.modules['input_contract']=m
try:runpy.run_path(sys.argv[1],run_name='import_probe')
except (ValueError,TypeError):sys.exit(0)
sys.exit(1)
"""
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1');r=subprocess.run([sys.executable,'-c',code,str(HERE/'worker.py')],env=env,capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__':unittest.main()
