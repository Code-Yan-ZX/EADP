"""Real frozen gate/source import order without model, CUDA, or container."""
import os,subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def run_isolated(self,code):return subprocess.run([sys.executable,'-c',code],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
 def test_real_gate_order_and_upstream_admission(self):
  code='''from pathlib import Path
import sys
from worker_entry import load
D=Path.cwd()
gate=load('_wavefront_t020_worker_gpu_gate',D/'gpu_admission.py')
try:gate.require_gpu()
except RuntimeError as e:assert 'GPU locked' in str(e)
else:raise AssertionError('unexpected GPU admission')
single=load('_wavefront_t020_worker_single_arm',D/'single_arm.py')
up=single.load('_wavefront_t020_upstream_admission',D/'upstream_admission.py')
prior=up.load_prior();proof=prior.require_preflight()
assert proof['source_sha256'] and Path(sys.modules['source_inventory'].__file__).resolve()==up.P/'source_inventory.py'
'''
  r=self.run_isolated(code);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
 def test_foreign_generic_inventory_rejected(self):
  code='''import sys,types
from pathlib import Path
from worker_entry import load
D=Path.cwd();load('_wavefront_t020_worker_gpu_gate',D/'gpu_admission.py')
x=types.ModuleType('source_inventory');x.__file__='/tmp/fake-source-inventory.py';sys.modules['source_inventory']=x
up=load('_wavefront_t020_upstream_admission',D/'upstream_admission.py')
try:up.load_prior()
except ValueError as e:assert 'foreign upstream inventory' in str(e)
else:raise AssertionError('pollution accepted')
'''
  r=self.run_isolated(code);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
if __name__=='__main__':unittest.main()
