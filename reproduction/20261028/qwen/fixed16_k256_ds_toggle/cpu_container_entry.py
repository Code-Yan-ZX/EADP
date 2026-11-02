"""Run CPU draft tests in an explicitly no-device author container."""
import json,os,sys,unittest
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not any(Path('/dev',n).exists() for n in ('nvidia0','nvidiactl','nvidia-uvm'))
import common as c
plan=c.read(c.PLAN);c.check(plan,'container')
import torch
assert not torch.cuda.is_initialized() and torch.__version__.startswith('2.8.0')
suite=unittest.TestSuite([unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent),pattern='cpu*test*.py'),unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent),pattern='test_verify_terminal.py')])
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not torch.cuda.is_initialized()
for path,digest in plan['files'].items():assert c.sha(path)==digest,path
print(json.dumps(dict(plan_sha256=c.sha(c.PLAN),success=result.wasSuccessful(),tests=result.testsRun,torch_version=torch.__version__,GPU_devices_present=False,CUDA_initialized=False,model_loaded=False)))
raise SystemExit(0 if result.wasSuccessful() else 1)
