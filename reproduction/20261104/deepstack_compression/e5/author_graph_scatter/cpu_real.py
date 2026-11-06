"""CPU original vs one-statement variant on the archived K512 input."""
import hashlib,json,os,time,traceback
from pathlib import Path
import torch
import transform

HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
CASE=BASE/'e5/selector_real32/run_001/image_017/case_512.pt'
CASE_SHA='f651a859849d1e0c84119a7b708ffff794cbe8beaf60e0d9a8bf63d5ae24f95d'
OUT=HERE/'cpu_real001.json'

def tensor_sha(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
    result={'status':'failed','GT_opened':False,'scoring_performed':False,'cuda_initialized':False}
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized() or transform.digest(CASE)!=CASE_SHA:raise ValueError('CUDA or input identity invalid')
        author=transform.load_author();candidate,metadata=transform.build(author)
        data=torch.load(CASE,map_location='cpu',weights_only=True);imp=data['importance'];sim=data['similarity'];stored=data['ordered']
        if imp.shape!=(1,1024) or sim.shape!=(1,1024,1024) or stored.shape!=(1,512):raise ValueError('unexpected input tensors')
        before={'importance':tensor_sha(imp),'similarity':tensor_sha(sim)}
        with torch.inference_mode():
            t=time.monotonic();old,oldquota=author._greed_select_impl(imp,sim,512);old_seconds=time.monotonic()-t
            t=time.monotonic();new,newquota=candidate(imp,sim,512);new_seconds=time.monotonic()-t
        after={'importance':tensor_sha(imp),'similarity':tensor_sha(sim)}
        oldids=old.tolist();newids=new.tolist();archived=stored.tolist()
        equal=bool(torch.equal(old,new) and torch.equal(oldquota,newquota) and before==after)
        result.update(status='passed' if equal else 'failed',input_sha256=CASE_SHA,author_sha256=transform.AUTHOR_SHA,
            transform_sha256=transform.digest(HERE/'transform.py'),script_sha256=transform.digest(__file__),metadata=metadata,
            original_order=oldids,candidate_order=newids,original_quota=oldquota.tolist(),candidate_quota=newquota.tolist(),
            cpu_original_matches_candidate=equal,cpu_original_matches_archived_gpu=bool(torch.equal(old,stored)),
            original_wall_seconds=old_seconds,candidate_wall_seconds=new_seconds,input_tensors_sha256=before,
            input_tensors_unchanged=before==after,threads=2,cuda_initialized=torch.cuda.is_initialized(),
            scope='one archived real K512 tensor case on CPU; no CUDA capture or replay',
            limitations=['CPU original versus saved GPU order is separately reported; CPU/GPU reductions may differ',
                         'no GPU identity, GPU graph capture or model-level claim'])
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
