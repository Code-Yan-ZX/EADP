"""PID1 EADP model worker; GPU reachable only after reviewed pre-CUDA handshake."""
import argparse,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;sys.path[:0]=[str(D),str(B/'e4_c_latency'),str(B/'e4')]
import gate
if Path(gate.__file__).resolve()!=D/'gate.py':raise ValueError('foreign memory gate')
_imports=sys.modules.get('imports')
if _imports is not None and (not getattr(_imports,'__file__',None) or Path(_imports.__file__).resolve()!=D/'imports.py'):raise ValueError('foreign imports module')
from imports import checked_import,check_existing,latency_bindings

def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def append(path,obj):
 with Path(path).open('a') as f:f.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def select(prepared):
 rows=[r for r in prepared['rows'] if r['task']=='DocVQA_VAL' and str(r['qid'])=='63709']
 if len(rows)!=1 or rows[0]['group_id']!='dd3c0ab5edc067ad04920cd66ea639ec85529b0ad6b6d25be7d423591bd59c80':raise ValueError('accepted anchor missing')
 path=B/'e4_c_timing/light_smoke_003/records.jsonl';formal=gate.read(B/'e4_c_timing/light_smoke_003_verification.json')
 if formal['evidence_sha256'].get(str(path.resolve()))!=gate.sha(path):raise ValueError('accepted anchor records changed')
 found=[]
 with path.open() as f:
  for line in f:
   item=json.loads(line);j=item['job']
   if (j.get('task'),str(j.get('qid')),j.get('path'),j.get('forced'))==('DocVQA_VAL','63709','C-ON',False):found.append(item['record'])
 if len(found)!=1 or any(found[0].get(k)!=rows[0].get(k) for k in ('task','group_id','image_sha256','message_sha256')):raise ValueError('C-ON anchor mismatch')
 return rows[0],found[0]

def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',required=True,type=Path);p.add_argument('--nonce');a=p.parse_args();out=a.out.resolve();gate.review(preflight_only=a.preflight_only,host=False)
 check_existing(latency_bindings())
 latency=checked_import('latency_worker',B/'e4_c_latency/latency_worker.py');prepare,load_runtime=latency.prepare,latency.load_runtime
 real=checked_import('model_run',D/'model_run.py');real_model,validate_result=real.real_model,real.validate_result
 identity=checked_import('identity_records',B/'e4/identity_records.py');tensor_meta=identity.tensor_meta
 check_existing({'skip_deepstack':B/'e4/skip_deepstack.py'})
 checked_import('light_adapter',B/'e4_c_latency/light_adapter.py')
 check_existing({'skip_deepstack':B/'e4/skip_deepstack.py'})
 check_existing(latency_bindings())
 import torch
 if torch.cuda.is_initialized():raise ValueError('unexpected pre-CUDA initialization')
 prepared=prepare('smoke');row,ref=select(prepared)
 author=Path(prepared['plan']['author_root']);check_existing({'model':author/'model/__init__.py','model.pruner':author/'model/pruner.py','vlmeval':author/'VLMEvalKit/vlmeval/__init__.py','vlmeval.vlm.qwen3_vl.model_fixed_res':author/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'})
 torch,pruner,wrapper_class,versions=load_runtime(prepared)
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner
 import inspect
 expected=(Path(prepared['plan']['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
 if Path(inspect.getfile(Qwen3VLChatCDPruner)).resolve()!=expected or torch.cuda.is_initialized():raise ValueError('actual wrapper path or CUDA state')
 spec=importlib.util.spec_from_file_location('eadp_realmodel_memory_barrier',B/'e4_c_memory_runtime/worker_barrier.py');barrier=importlib.util.module_from_spec(spec);spec.loader.exec_module(barrier)
 if Path(barrier.__file__).resolve()!=B/'e4_c_memory_runtime/worker_barrier.py':raise ValueError('barrier path')
 if a.preflight_only:
  for mode in ('controller','sampler'):
   p=subprocess.run([sys.executable,str(D/'import_probe.py'),'--mode',mode],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60)
   if p.returncode!=0 or p.stdout.strip()!='import_probe_passed '+mode:raise ValueError('actual '+mode+' import chain failed: '+p.stderr[-1000:])
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('preflight GPU not hidden')
  save(out/'worker_preflight.json',{'status':'passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,'predictions_opened':True,'prediction_scope':'E1 and accepted E4-C DEV journals for identity and exact output reference only','weights_verified':len(prepared['weights_before']),'anchor':{'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id']},'versions':versions,'source_sha256':gate.source_hashes()});return
 if os.getpid()!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or not a.nonce:raise ValueError('GPU mode requires reviewed PID1 worker')
 expected,release=barrier.handshake(out,a.nonce,torch.cuda.is_initialized,timeout=300)
 append(out/'worker_events.jsonl',{'event':'barrier_released','monotonic_ns':time.monotonic_ns(),'first_sample_ns':release['first_sample_ns']})
 recipe=prepared['plan']['recipe'];holder={}
 def wrapper(**kw):
  torch.cuda.set_per_process_memory_fraction(.9,0);return wrapper_class(**kw)
 def record(event):append(out/'worker_events.jsonl',event)
 model,response=real_model(torch,row,ref,recipe,pruner,wrapper,Qwen3VLChatCDPruner,record,time.monotonic_ns)
 identity=validate_result(response,ref,tensor_meta);save(out/'identity.json',identity)
 del response,model;torch.cuda.empty_cache();torch.cuda.synchronize();append(out/'worker_events.jsonl',{'event':'teardown_end','monotonic_ns':time.monotonic_ns()})
 for path,before in prepared['weights_before'].items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weights changed during run')
 save(out/'worker_complete.json',{'status':'completed','GT_opened':False,'model_loaded':True,'scoring_performed':False,'anchor':{'task':row['task'],'qid':str(row['qid'])},'versions':versions})
if __name__=='__main__':main()
