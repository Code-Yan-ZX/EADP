"""No-GPU preparation for future memory worker; GPU entry remains locked.

Run in its own process, separate from the host sampler/controller, because
both the memory and latency directories contain a file named contract.py.
This helper has no CLI and does not itself establish Docker or supervision.
"""
import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
LATENCY=BASE/'e4_c_latency'

def validate_latency_contract(module):
 if Path(getattr(module,'__file__','/')).resolve()!=(LATENCY/'contract.py').resolve():raise ValueError('wrong contract module resolved in memory worker')

def prepare_memory(stage):
 if stage not in ('smoke','full'):raise ValueError('bad memory stage')
 sys.path.insert(0,str(LATENCY))
 import contract as latency_contract
 validate_latency_contract(latency_contract)
 from latency_gate import require_light_identity
 from latency_worker import prepare,load_runtime,sources
 from real_worker import sha
 spec=importlib.util.spec_from_file_location('eadp_memory_plan',HERE/'plan.py');memory_plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(memory_plan)
 identity=require_light_identity();prepared=prepare(stage);plans=memory_plan.split(prepared['schedule'],stage)
 torch,_native,_wrapper,versions=load_runtime(prepared)
 validate_latency_contract(sys.modules['contract'])
 if torch.cuda.is_initialized():raise ValueError('memory CPU preflight initialized CUDA')
 paths=list(sources())+[HERE/x for x in ('worker_preflight.py','plan.py','worker.py','test_worker_preflight.py')]
 source={str(p):sha(p) for p in paths}
 if any(sha(path)!=digest for path,digest in source.items()):raise ValueError('source changed during CPU preflight')
 return {'status':'passed','stage':stage,'scope':'CPU before model load; no GPU, predictions, GT or scoring','cuda_initialized':False,'model_loaded':False,
         'light_identity':identity,'memory_job_counts':{m:len(p['jobs']) for m,p in plans.items()},'source_sha256':source,
         'weight_stats_before':prepared['weights_before'],'versions':versions,'latency_contract_path':str(Path(latency_contract.__file__).resolve())}

def main():raise SystemExit('memory preflight CLI locked until independent CPU review and retained container controller')
if __name__=='__main__':main()
