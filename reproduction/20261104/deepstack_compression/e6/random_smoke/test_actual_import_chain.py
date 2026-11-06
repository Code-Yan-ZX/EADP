"""Load actual worker dependencies in a fresh no-GPU interpreter."""
import os,subprocess,sys,unittest
class Tests(unittest.TestCase):
 def test_real_imports_without_cuda(self):
  code=r'''
import sys
from pathlib import Path
import model_worker as w
sys.path[:0]=[str(w.RB),str(w.E4),str(w.LAT),str(w.D)]
latency=w.load('_random_latency_worker',w.LAT/'latency_worker.py')
for n,p in [('_random_e1_admission',w.B/'m3_oracle_exec/e1_admission.py'),('_random_e1_refs',w.D/'e1_refs.py'),('_random_manifest',w.D/'manifest_load.py'),('_random_inventory',w.D/'inventory.py'),('_random_identity_contract',w.D/'identity_contract.py'),('_random_smoke_core',w.D/'smoke_core.py'),('_random_diagnostic_run',w.D/'diagnostic_run.py'),('_random_model_core',w.D/'model_core.py'),('_random_selector',w.RB/'selector.py'),('_random_model_adapter',w.RB/'adapter.py'),('_random_budget_runner',w.E4/'budget_runner.py'),('_random_fixed_common',w.B/'e4_c_multibudget_fixed32/fixed_common.py')]:
 m=w.load(n,p)
 if Path(m.__file__).resolve()!=p.resolve():raise RuntimeError('bad actual source '+n)
import torch
if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized')
'''
  r=subprocess.run([sys.executable,'-c',code],cwd=os.path.dirname(__file__),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=60)
  self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__':unittest.main()
