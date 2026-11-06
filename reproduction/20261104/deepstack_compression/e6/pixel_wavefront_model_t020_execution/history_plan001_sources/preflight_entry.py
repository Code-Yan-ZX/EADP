"""Run execution-module import and GPU-lock checks in retained no-device container."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import preflight_admission
if Path(preflight_admission.__file__).resolve()!=D/'preflight_admission.py':raise ValueError('foreign admission')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/worker_preflight.json').resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('invalid no-device preflight')
 approval=preflight_admission.require_preflight()
 import torch
 if torch.cuda.is_initialized():raise ValueError('CUDA preinitialized')
 names=('gpu_admission','worker_entry','single_arm','controller_draft','bounded_runner_draft','schedule_contract','upstream_identity')
 for name in names:
  path=D/(name+'.py');spec=importlib.util.spec_from_file_location('_t020_preflight_'+name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  if Path(m.__file__).resolve()!=path:raise ValueError('module path')
 import upstream_identity;prior=upstream_identity.verify()
 import gpu_admission
 try:gpu_admission.require_gpu()
 except RuntimeError as e:
  if 'GPU locked' not in str(e):raise
 else:raise ValueError('GPU gate not locked')
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
 result={'status':'passed','approval':approval,'gpu_authorized':False,'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'module_paths':{x:str(D/(x+'.py')) for x in names},'upstream_review_sha256':prior['review_sha256']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
