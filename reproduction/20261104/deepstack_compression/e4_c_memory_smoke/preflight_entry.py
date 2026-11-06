"""No-GPU container worker preparation and isolated sampler import."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import cpu_gate,wiring

def sampler_import():
    folder=cpu_gate.BASE/'e4_c_memory_nvml';memory=cpu_gate.BASE/'e4_c_memory'
    code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import real_sampler,sampler,contract;assert Path(real_sampler.__file__).resolve()==Path(sys.argv[1]).resolve()/'real_sampler.py';assert Path(sampler.__file__).resolve()==Path(sys.argv[2]).resolve()/'sampler.py';assert Path(contract.__file__).resolve()==Path(sys.argv[2]).resolve()/'contract.py';print('isolated sampler imports passed')"
    p=subprocess.run([sys.executable,'-c',code,str(folder),str(memory)],capture_output=True,text=True,timeout=120,check=True)
    if p.stdout.strip()!='isolated sampler imports passed':raise ValueError('sampler import mismatch')
    return {'status':'passed','process_isolated':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=a.out.resolve()
    launch=json.loads((d/'launch.json').read_text());approval=launch['approval']
    if (approval.get('gpu_authorized') is not False or approval.get('source_sha256')!=cpu_gate.sources()
        or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):raise ValueError('not approved no-GPU preflight')
    context=wiring.prepare(approval);sampler=sampler_import()
    if context['torch'].cuda.is_initialized() or cpu_gate.sources()!=approval['source_sha256']:raise ValueError('CUDA/source changed')
    prepared=context['prepared'];result={'status':'passed','scope':'C-ON memory worker CPU preflight only',
       'gpu_authorized':False,'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'scoring_performed':False,
       'existing_DEV_predictions_opened':True,'existing_predictions_use':'input identity only','new_predictions_generated':False,
       'source_sha256':approval['source_sha256'],'sampler_import':sampler,'schedule_jobs':len(context['expected_schedule']['schedule']),
       'weight_stats_before':prepared['weights_before'],'e1_acceptance_sha256':prepared['e1_acceptance_sha256'],
       'anchor_sha256':prepared['anchor_sha256'],'versions':context['versions'],'approval_review_sha256':approval['review_sha256']}
    with (d/'worker_preflight.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
