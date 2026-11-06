"""Actual container imports in separate worker/sampler processes; no GPU."""
import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;LATENCY=HERE.parent/'e4_c_latency';sys.path.insert(0,str(LATENCY))
import contract as latency_contract
import admission,worker_preflight,preflight_entry,preflight_launch
from latency_worker import load_runtime
from real_worker import PLAN
if Path(latency_contract.__file__).resolve()!=(LATENCY/'contract.py').resolve():raise ValueError('wrong worker contract')
for module in (admission,worker_preflight,preflight_entry,preflight_launch):
 if Path(module.__file__).resolve().parent!=HERE:raise ValueError('wrong memory worker module')
torch,_native,_wrapper,versions=load_runtime({'plan':json.loads(PLAN.read_text())})
if torch.cuda.is_initialized():raise ValueError('CPU import initialized CUDA')
child=subprocess.run([sys.executable,str(HERE/'sampler_probe.py')],capture_output=True,text=True,timeout=60,check=True);sampler=json.loads(child.stdout)
if sampler.get('status')!='passed' or sampler.get('contract_path')!=str((HERE/'contract.py').resolve()) or sampler.get('gpu_used') is not False:raise ValueError('wrong sampler process import')
print(json.dumps({'status':'passed','cuda_initialized':False,'model_loaded':False,'predictions':False,'worker_contract':str(Path(latency_contract.__file__).resolve()),'sampler_contract':sampler['contract_path'],'versions':versions}),flush=True)
