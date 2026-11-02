"""No-GPU device CPU preflight of frozen full input, model and source guards."""
import json,os,sys
from pathlib import Path
import common as c
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not any(Path('/dev',n).exists() for n in ('nvidia0','nvidiactl','nvidia-uvm'))
p=c.read(c.PLAN);c.check(p,'container');assert 'torch' not in sys.modules
print(json.dumps(dict(status='passed',plan_sha256=c.sha(c.PLAN),GPU_devices_present=False,torch_imported=False,model_loaded=False)))
