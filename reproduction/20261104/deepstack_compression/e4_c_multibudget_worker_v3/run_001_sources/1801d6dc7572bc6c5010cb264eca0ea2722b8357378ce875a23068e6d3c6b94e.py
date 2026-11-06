"""32-image timing worker continuation. No GPU authority before future gates."""
import argparse,importlib.metadata,importlib.util,inspect,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent

def load(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
admission=load('_multibudget_timing_admission',HERE/'admission.py');preflight=load('_multibudget_timing_preflight',HERE/'preflight.py')
runner=load('_multibudget_timing_runner',HERE/'runner.py');core=load('_multibudget_timing_core',HERE/'worker_core.py')
reference_gate=load('_multibudget_fixed_reference_gate',HERE/'fixed_reference_gate.py')

def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def read_identity(path,timeout=30,now=time.monotonic,sleep=time.sleep):
 deadline=now()+timeout;last=None
 while now()<deadline:
  try:return admission.read(path)
  except (FileNotFoundError,json.JSONDecodeError) as error:last=error;sleep(.1)
 raise ValueError('container identity not complete before deadline') from last
def verify_container(out,approval):
 launch=admission.read(out/'launch.json')
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=admission.sha(__file__) or launch.get('run_out')!=str(out) or launch.get('time_namespace')!=os.readlink('/proc/self/ns/time'):raise ValueError('timing launch identity mismatch')
 item=read_identity(out/'container_identity.json');cid=item.get('container_id')
 if not isinstance(cid,str) or len(cid)!=64 or cid not in Path('/proc/self/cgroup').read_text() or cid not in item.get('host_cgroup','') or not (item.get('host_init') or {}).get('start_ticks'):raise ValueError('container identity handshake mismatch')

def run_after_admission(out,plan,cpu,approval):
 # These modules were path-pinned by the actual no-GPU processor preflight.
 import torch,real_worker
 refs_mod=load('_multi_budget_references',HERE/'references.py');refs=refs_mod.load_references()
 fixed=reference_gate.load();sample_plan=admission.read(BASE/'e4_c_multibudget_expansion/pilot_plan001.json')
 sample_plan['_validated_full_cohort']=admission.read(BASE/'e4_c_latency_full/full_001/frozen.json')['schedule']['cohort']
 contract=load('_multi_budget_contract',BASE/'e4_c_multibudget_expansion/pilot_contract.py');core.prepare(sample_plan,refs['rows'],refs['records'],contract)
 generation=load('_multi_budget_generation',BASE/'e4_c_budget_ttft_pilot_v3/generation_identity.py')
 generation.validate(plan['generation_source_sha256'],admission.sha)
 adapter=load('multibudget_light_adapter',BASE/'e4_c_budget_pilot_v2/adapter.py')
 author=admission.read(real_worker.PLAN);recipe=author['recipe'];expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
 if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('author recipe mismatch')
 sys.path[:0]=[author['author_root'],str(Path(author['author_root'])/'VLMEvalKit')]
 from model import pruner
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
 from transformers.generation.stopping_criteria import StoppingCriteriaList
 if ({n:importlib.metadata.version(n) for n in author['expected_versions']}!=author['expected_versions'] or Path(pruner.__file__).resolve()!=(Path(author['author_root'])/'model/pruner.py').resolve() or Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=(Path(author['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve() or torch.cuda.get_device_name(0)!='NVIDIA A40'):raise ValueError('model/runtime identity mismatch')
 save(out/'frozen.json',{'stage':'32-image exploratory multibudget timing','approval':approval,'source_sha256':plan['source_sha256'],'schedule':sample_plan['jobs'],'weight_stats_before':cpu['weight_stats'],'E4_records_sha256':refs['records_sha256'],'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'memory_claims':False})
 core.append(out/'stages.jsonl',{'event':'model_loading','unix':time.time()});wrapper=Qwen3VLChatEADP(**recipe)
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('timing backend mismatch')
 generation.validate(plan['generation_source_sha256'],admission.sha);generation.validate_model_class(wrapper.model)
 for row in refs['rows']:real_worker.preflight_one(torch,wrapper,row,{'processor':row['processor_expected'],'image_start':row['image_start']})
 import gc
 gc.collect();torch.cuda.synchronize();core.append(out/'stages.jsonl',{'event':'model_ready','unix':time.time()})
 def author_generate(w,row):
  with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
 def execute(row,budget,protocol,panel,raw):
  return runner.execute(torch,wrapper,pruner,row,budget,protocol,panel,adapter.BudgetLightAdapter,StoppingCriteriaList,author_generate,raw,generation.validate_model_class)
 result=core.run(sample_plan,refs['rows'],refs['records'],fixed,contract,execute,lambda x:core.append(out/'journal.jsonl',x))
 admission.plan();generation.validate(plan['generation_source_sha256'],admission.sha)
 for path,before in cpu['weight_stats'].items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight file changed')
 if refs_mod.load_references()['records_sha256']!=refs['records_sha256']:raise ValueError('accepted E4 references changed')
 save(out/'worker_complete.json',dict(result,source_unchanged=True,weights_stat_unchanged=True,journal_sha256=admission.sha(out/'journal.jsonl')))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--preflight-only',action='store_true');args=p.parse_args();out=args.out.resolve()
 if not out.is_dir():raise ValueError('launcher output missing')
 approval=None if args.preflight_only else admission.admit_gpu()
 verify_container(out,approval);plan=admission.plan()
 cpu=preflight.prepare(plan,out/'processor_preflight.json',require_no_gpu=args.preflight_only)
 save(out/'worker_preflight.json',{'status':'passed','plan_sha256':admission.sha(HERE/'runtime_plan001.json'),'processor_preflight_sha256':admission.sha(out/'processor_preflight.json'),'jobs':cpu['jobs'],'inputs':cpu['inputs_checked'],'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'cuda_initialized':False,'model_loaded':False})
 if not args.preflight_only:run_after_admission(out,plan,cpu,approval)
if __name__=='__main__':main()
