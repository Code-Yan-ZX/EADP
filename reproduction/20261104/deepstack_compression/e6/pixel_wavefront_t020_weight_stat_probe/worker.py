"""Supplemental no-device bind-mount stat check; does not load a model."""
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import torch

D = Path(__file__).resolve().parent
B = D.parents[1]
GATE = B / 'e6/pixel_wavefront_t020_timing_final_v4/gpu_gate.py'
GATE_SHA = '21b2dde7c1c1a7877e26d9be6d466a6ad6c9519ad85e7d7413646ba0486c2412'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not torch.cuda.is_initialized()
assert hashlib.sha256(GATE.read_bytes()).hexdigest() == GATE_SHA
assert '_weight_stat_probe_gate' not in sys.modules
spec = importlib.util.spec_from_file_location('_weight_stat_probe_gate', GATE)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
result = gate.weight_preflight()
assert len(result['expected']) == 4 and not torch.cuda.is_initialized()
out = D / 'run001/worker.json'
with out.open('x') as f:
 json.dump({'status':'passed','gate_sha256':GATE_SHA,'weights':result,'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'GPU_used':False,'pid':os.getpid(),'hostname':os.uname().nodename,'self_cgroup':Path('/proc/self/cgroup').read_text()},f,indent=2)
 f.write('\n'); f.flush(); os.fsync(f.fileno())
time.sleep(5)
