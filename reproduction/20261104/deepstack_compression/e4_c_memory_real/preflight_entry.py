"""Real worker CPU preparation in retained container, never load model/CUDA."""
import argparse,json,os,sys
from pathlib import Path
import gate
from worker import prepare_cpu
import controller,guard

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    out=a.out.resolve(); launch=json.loads((out/'launch.json').read_text()); approval=launch['approval']
    if (approval.get('gpu_authorized') is not False or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
        or approval.get('source_sha256')!=gate.sources()):raise ValueError('not approved CPU-only launch')
    latency=gate.BASE/'e4_c_latency';sys.path.insert(0,str(latency))
    from latency_gate import require_light_identity
    from latency_worker import prepare,load_runtime
    from schedule import build
    value=prepare_cpu(stage='smoke',mode='C-ON',approved_sources=approval['source_sha256'],
                      require_identity=require_light_identity,prepare=prepare,load_runtime=load_runtime,
                      build=build,sha=gate.runtime.sha)
    prepared=value['prepared']
    if value['torch'].cuda.is_initialized() or gate.sources()!=approval['source_sha256']:raise ValueError('CPU preflight source/CUDA changed')
    result={'status':'passed','scope':'real NVML CPU worker preflight only','gpu_authorized':False,
            'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'scoring_performed':False,
            'existing_DEV_predictions_opened':True,'existing_predictions_use':'input identity only',
            'new_predictions_generated':False,
            'source_sha256':approval['source_sha256'],'schedule_jobs':len(value['expected_schedule']['schedule']),
            'weight_stats_before':prepared['weights_before'],'versions':value['versions'],
            'light_identity':value['identity'],'e1_acceptance_sha256':prepared['e1_acceptance_sha256'],
            'anchor_sha256':prepared['anchor_sha256'],'approval_review_sha256':approval['review_sha256']}
    with (out/'worker_preflight.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
