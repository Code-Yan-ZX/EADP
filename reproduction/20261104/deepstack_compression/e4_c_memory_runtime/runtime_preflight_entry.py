"""Reviewed runtime worker preflight only; no GPU/model/generation."""
import argparse,importlib,json,os,subprocess,sys
from pathlib import Path
from runtime_admission import HERE,B,admit,source_hashes

def preflight(out,approval,prepare,import_runtime,sampler_import):
 out=Path(out).resolve();launch=json.loads((out/'launch.json').read_text())
 if launch.get('approval_bundle')!=approval or approval.get('gpu_authorized') is not False or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('not approved no-GPU preflight')
 if source_hashes()!=approval['source_sha256']:raise ValueError('runtime sources changed')
 previous=prepare('smoke')
 if previous.get('status')!='passed' or previous.get('cuda_initialized') is not False or previous.get('model_loaded') is not False:raise ValueError('worker CPU preflight failed')
 modules=import_runtime();sampler=sampler_import()
 import torch
 if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
 if source_hashes()!=approval['source_sha256']:raise ValueError('runtime sources changed during preflight')
 result={'status':'passed','gpu_authorized':False,'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'scoring_performed':False,'predictions':False,'approval_bundle':approval,'prior_worker_preflight':previous,'runtime_imports':modules,'sampler_import':sampler,'scope':'complete CPU runtime import and prior worker preparation only; no Docker sampler handshake or GPU'}
 with (out/'worker_preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return result

def import_runtime():
 names=('bridge','controller_adapter','controller_io','docker_guard','interruptible','lifecycle','sampler_supervisor','worker_barrier','worker_candidate')
 result={}
 for name in names:
  m=importlib.import_module(name);expected=(HERE/(name+'.py')).resolve()
  if Path(m.__file__).resolve()!=expected:raise ValueError('runtime module collision')
  result[name]=str(expected)
 return result

def sampler_import():
 code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import sampler_entry,contract;assert Path(sampler_entry.__file__).resolve()==Path(sys.argv[1]).resolve()/'sampler_entry.py';assert Path(contract.__file__).resolve()==Path(sys.argv[2]).resolve()/'contract.py';print('sampler import passed')"
 p=subprocess.run([sys.executable,'-c',code,str(HERE),str(B/'e4_c_memory')],capture_output=True,text=True,timeout=120,check=True)
 if p.stdout.strip()!='sampler import passed':raise ValueError('sampler import identity')
 return {'status':'passed','process_isolated':True,'stdout':p.stdout.strip()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();approval=admit()
 sys.path.insert(0,str(B/'e4_c_memory'));from worker_preflight import prepare_memory
 preflight(a.out,approval,prepare_memory,import_runtime,sampler_import)
if __name__=='__main__':main()
