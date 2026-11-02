"""Read-only CPU container guard; no model, tensor runtime or NVIDIA device."""
import json
import os
from pathlib import Path
import sys
import common as c
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not any(Path('/dev',n).exists() for n in ('nvidia0','nvidiactl','nvidia-uvm'))
p=c.read(c.PLAN);c.plan_checks(p,'container')
assert 'torch' not in sys.modules
print(json.dumps(dict(status='passed',plan_sha256=c.sha(c.PLAN),GPU_devices_present=False,torch_imported=False,model_loaded=False)))
