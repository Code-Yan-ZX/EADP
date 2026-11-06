"""Full 5940-job timing worker; reviewed two-stage admission before model/GPU."""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PILOT=HERE.parent/'e4_c_latency';sys.path.insert(0,str(PILOT))
from full_admission import admit,sources
from latency_worker import prepare,load_runtime,continue_stage
from real_worker import sha

def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();a.stage='full';out=a.out.resolve()
 approval=admit(preflight_only=a.preflight_only,host=False);launch=json.loads((out/'launch.json').read_text())
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=sha(__file__):raise ValueError('full launcher identity mismatch')
 prepared=prepare('full');actual={str(x):sha(x) for x in sources()}
 if actual!=approval['source_sha256'] or len(prepared['schedule']['schedule'])!=5940:raise ValueError('reviewed full sources/schedule changed')
 if a.preflight_only:
  torch,_native,_wrapper,versions=load_runtime(prepared)
  if torch.cuda.is_initialized():raise ValueError('CPU preflight initialized CUDA')
  save(out/'worker_preflight.json',{'status':'passed','jobs':5940,'cuda_initialized':False,'model_loaded':False,'predictions':False,'GT_opened':False,'scoring_performed':False,
       'full_source_sha256':actual,'pilot_identity':approval['pilot_identity'],'weight_stats_before':prepared['weights_before'],'versions':versions,'scope':'CPU before model load'})
  return
 save(out/'full_provenance.json',{'stage':'full','full_source_sha256':actual,'pilot_identity':approval['pilot_identity'],'budget_sha256':sha(HERE/'budget_plan.json'),'GT_opened':False,'scoring_performed':False})
 continue_stage(a,prepared,approval['pilot_identity']['light_identity'])
 if any(sha(path)!=digest for path,digest in actual.items()):raise ValueError('full source changed during execution')
if __name__=='__main__':main()
