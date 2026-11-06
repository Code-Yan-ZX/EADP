"""Actual diagnostic worker; GPU requires reviewed launcher and proof."""
import argparse,hashlib,importlib.metadata,importlib.util,inspect,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;V2=BASE/'e4_c_budget_pilot_v2'
def load(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def check_gate(path):
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!='fdd588a9914cb624a4fb496158f4166a5b571a64db8a6dfacb8db316471920c6':raise ValueError('gate source changed before import')
 return path
gate=load('_budget_execution_gate',check_gate(HERE/'gate.py'))
def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
def event(out,name):
 with (out/'stages.jsonl').open('a') as f:f.write(json.dumps({'event':name,'unix':time.time()})+'\n');f.flush();os.fsync(f.fileno())

def read_identity(path,timeout=30,now=time.monotonic,sleep=time.sleep):
 deadline=now()+timeout;last=None
 while now()<deadline:
  try:return gate.read(path)
  except (FileNotFoundError,json.JSONDecodeError) as error:last=error;sleep(.1)
 raise ValueError('container identity not complete before deadline') from last

def verify_container(out,approval):
 launch=gate.read(out/'launch.json')
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=gate.sha(__file__) or launch.get('run_out')!=str(out):raise ValueError('launch identity mismatch')
 item=read_identity(out/'container_identity.json');cid=item.get('container_id');own=Path('/proc/self/cgroup').read_text()
 if not isinstance(cid,str) or len(cid)!=64 or cid not in own or cid not in item.get('host_cgroup','') or not (item.get('host_init') or {}).get('start_ticks'):raise ValueError('container identity handshake mismatch')
 return item

def preflight(out):
 """Exactly the same source/import/processor/weight path for CPU and GPU."""
 plan=gate.plan();prior=gate.prior();pre=load('_bulk_reference_preflight',HERE/'preflight.py')
 pre.prepare(plan,out/'processor_preflight.json',require_no_gpu=False)
 for name,path in [('identity_records',BASE/'e4/identity_records.py'),('decode_observer',BASE/'e4/decode_observer.py'),('off_adapter',BASE/'e4/off_adapter.py'),('budget_adapter',BASE/'e4/budget_adapter.py'),('fixed_common',BASE/'e4_c_multibudget_fixed32/fixed_common.py'),('reference_cache',HERE/'reference_cache.py'),('fixed_core',BASE/'e4_c_multibudget_fixed32/core.py'),('reference_arm',HERE/'reference_arm.py'),('reference_core',HERE/'core.py'),('reference_inputs',HERE/'references.py')]:load(name,path)
 v=gate.read(out/'processor_preflight.json')
 if v.get('status')!='passed' or v.get('cuda_initialized') is not False or v.get('model_loaded') is not False:raise ValueError('no-GPU preflight failed')
 gate.plan();return plan,prior,v

def run(out,approval,plan,prior,cpu):
 import torch
 import real_worker,reference_inputs,fixed_core,reference_arm,reference_core,identity_records
 author=gate.read(real_worker.PLAN);recipe=author['recipe'];expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
 if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('frozen author recipe changed')
 sys.path[:0]=[author['author_root'],str(Path(author['author_root'])/'VLMEvalKit')]
 from model import pruner
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
 if ({n:importlib.metadata.version(n) for n in author['expected_versions']}!=author['expected_versions'] or
     Path(pruner.__file__).resolve()!=(Path(author['author_root'])/'model/pruner.py').resolve() or
     Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=(Path(author['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve() or
     torch.cuda.get_device_name(0)!='NVIDIA A40'):raise ValueError('actual model/runtime identity mismatch')
 refs=reference_inputs.load_references()
 save(out/'frozen.json',{'stage':'222-input fixed32 reference generation; no latency claim','approval':approval,'prior_review_sha256':prior['review_sha256'],
      'source_sha256':plan['source_sha256'],'weights_before':cpu['weight_stats'],'E4_formal_sha256':refs['formal_sha256'],
      'E4_review_sha256':refs['review_sha256'],'E4_records_sha256':refs['records_sha256'],
      'processor_preflight_sha256':gate.sha(out/'processor_preflight.json'),'GT_opened':False,'scoring_performed':False,'predictions_opened':True})
 event(out,'model_loading');wrapper=Qwen3VLChatEADP(**recipe)
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('model dtype/backend changed')
 for row in refs['rows']:
  real_worker.preflight_one(torch,wrapper,row,{'processor':row['processor_expected'],'image_start':row['image_start']})
 event(out,'model_ready')
 def generate(w,row):
  with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
 target=out/'diagnostic';target.mkdir(exist_ok=False)
 result=reference_core.run(torch,wrapper,pruner,refs,target,generate,reference_arm.run_arm,fixed_core,identity_records.tensor_meta)
 gate.plan()
 for path,before in cpu['weight_stats'].items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weights changed during diagnostic')
 again=reference_inputs.load_references()
 for key in ('formal_sha256','review_sha256','records_sha256'):
  if again[key]!=refs[key]:raise ValueError('accepted reference changed during run')
 save(out/'worker_complete.json',dict(result,status='passed',source_unchanged=True,weights_stat_unchanged=True,
      diagnostic_complete_sha256=gate.sha(target/'worker_complete.json'),GT_opened=False,scoring_performed=False,timing_claims=False))
 event(out,'complete')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args();out=args.out.resolve()
 if not out.is_dir():raise ValueError('launcher output missing')
 approval=None
 if not args.preflight_only:approval=gate.admit_gpu()
 verify_container(out,approval)
 event(out,'cpu_preflight_started');plan,prior,cpu=preflight(out);event(out,'cpu_preflight_completed')
 if args.preflight_only:
  save(out/'worker_preflight.json',{'status':'passed','plan_sha256':gate.sha(HERE/'execution_plan001.json'),'prior_review_sha256':prior['review_sha256'],
       'processor_preflight_sha256':gate.sha(out/'processor_preflight.json'),'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'cuda_initialized':False,'model_loaded':False})
  return
 run(out,approval,plan,prior,cpu)
if __name__=='__main__':main()
