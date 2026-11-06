import hashlib,importlib.util,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];path=B/'e6/rres_k256_execution/gpu_gate.py';spec=importlib.util.spec_from_file_location('_rres_gatecheck',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);approval=m.admit_gpu()
with (D/'result001.json').open('x') as f:json.dump({'status':'passed','approval':approval,'gpu_gate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'GT_opened':False,'GPU_queried':False,'model_loaded':False,'errors':[]},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
