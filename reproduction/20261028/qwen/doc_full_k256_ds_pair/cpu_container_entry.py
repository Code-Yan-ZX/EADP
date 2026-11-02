"""Synthetic tests and source guards in author container without GPU devices."""
import json,os,unittest
from pathlib import Path
import common as c
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not any(Path('/dev',n).exists() for n in ('nvidia0','nvidiactl','nvidia-uvm'))
p=c.read(c.PLAN);c.check(p,'container')
import torch
assert not torch.cuda.is_initialized() and torch.__version__.startswith('2.8.0')
root=str(Path(__file__).resolve().parent)
a=unittest.defaultTestLoader.discover(root,pattern='cpu*test*.py');b=unittest.defaultTestLoader.discover(root,pattern='test_verify_generation.py')
n=a.countTestCases();m=b.countTestCases();assert n>=8 and m>=1
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([a,b]));assert not torch.cuda.is_initialized()
for path,digest in p['files'].items():assert c.sha(path)==digest,path
print(json.dumps(dict(plan_sha256=c.sha(c.PLAN),success=result.wasSuccessful(),tests=result.testsRun,hook_tests=n,verifier_tests=m,torch_version=torch.__version__,GPU_devices_present=False,CUDA_initialized=False,model_loaded=False)))
raise SystemExit(0 if result.wasSuccessful() else 1)
