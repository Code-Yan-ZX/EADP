"""Fresh-process sampler import probe. No nvidia-smi, model, CUDA, or GPU."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
import contract,sampler,barrier,host_evidence,sampler_io
for mod in (contract,sampler,barrier,host_evidence,sampler_io):
 if Path(mod.__file__).resolve().parent!=HERE:raise ValueError('wrong memory sampler module')
print(json.dumps({'status':'passed','contract_path':str(Path(contract.__file__).resolve()),'gpu_used':False,'model_loaded':False}),flush=True)
