"""Prepared TTFT worker. GPU is unconditionally locked in admission.py."""
import argparse,importlib.metadata,importlib.util,inspect,json,os,sys,time
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
identity=load('_budget_ttft_identity',HERE/'identity_gate.py');admission=load('_budget_ttft_admission',HERE/'admission.py')
contract=load('_budget_ttft_contract',HERE/'contract.py');candidate=load('_budget_ttft_candidate',HERE/'worker_candidate.py');runner=load('_budget_ttft_runner',HERE/'runner.py')
generation=load('_budget_ttft_generation_identity',HERE/'generation_identity.py')
def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def prepare(out):
 runtime=admission.plan();accepted=identity.admit_identity();plan=json.loads((HERE/'pilot_plan001.json').read_text())
 if identity.sha(HERE/'contract.py')!=plan['schedule_source_sha256'] or identity.sha(BASE/'e6/pixel_natural_timing_pilot_v4/plan001.json')!=plan['anchor_plan_sha256']:raise ValueError('frozen schedule/input source changed')
 contract.validate(plan['jobs'],plan['anchors'])
 pre=load('_budget_ttft_processor_preflight',V2/'preflight.py');before=sys.argv
 try:sys.argv=[str(V2/'preflight.py'),'--out',str(out/'processor_preflight.json')];pre.main()
 finally:sys.argv=before
 import torch,budget_reference
 if torch.cuda.is_initialized():raise ValueError('CPU preflight unexpectedly initialized CUDA')
 refs=budget_reference.load_references();contract.validate(plan['jobs'],refs['rows'])
 if len(refs['records'])!=12:raise ValueError('same-arm references incomplete')
 generation_paths=generation.validate(runtime['generation_source_sha256'],identity.sha)
 admission.plan()
 return {'runtime':runtime,'identity':accepted,'schedule':plan['jobs'],'refs':refs,'preflight':json.loads((out/'processor_preflight.json').read_text()),'torch':torch,'generation_paths':generation_paths}
def verify_container(out,approval):
 launch=json.loads((out/'launch.json').read_text())
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=identity.sha(__file__) or launch.get('run_out')!=str(out) or launch.get('time_namespace')!=os.readlink('/proc/self/ns/time'):raise ValueError('timing launch identity mismatch')
 path=out/'container_identity.json';deadline=time.monotonic()+30
 while not path.exists() and time.monotonic()<deadline:time.sleep(.1)
 item=json.loads(path.read_text());cid=item.get('container_id')
 if not isinstance(cid,str) or len(cid)!=64 or cid not in Path('/proc/self/cgroup').read_text() or cid not in item.get('host_cgroup','') or not (item.get('host_init') or {}).get('start_ticks'):raise ValueError('container identity handshake mismatch')
def run_after_future_admission(out,prepared,approval):
 """Call only after final independent review and scheduled GPU release."""
 import budget_reference,real_worker,timer,adapter
 torch=prepared['torch'];author=json.loads(real_worker.PLAN.read_text());recipe=author['recipe']
 expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
 if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('author recipe mismatch')
 sys.path[:0]=[author['author_root'],str(Path(author['author_root'])/'VLMEvalKit')]
 from model import pruner
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
 from transformers.generation.stopping_criteria import StoppingCriteriaList
 if ({n:importlib.metadata.version(n) for n in author['expected_versions']}!=author['expected_versions'] or
     Path(pruner.__file__).resolve()!=(Path(author['author_root'])/'model/pruner.py').resolve() or
     Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=(Path(author['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve() or
     torch.cuda.get_device_name(0)!='NVIDIA A40'):raise ValueError('timing runtime identity mismatch')
 refs=prepared['refs'];save(out/'frozen.json',{'stage':'exploratory three-anchor TTFT','approval':approval,'source_sha256':prepared['runtime']['source_sha256'],
  'schedule':prepared['schedule'],'identity':prepared['identity'],'weight_stats_before':prepared['preflight']['weight_stats'],
  'E4_records_sha256':refs['records_sha256'],'GT_opened':False,'scoring_performed':False,'predictions_opened':True,
  'memory_claims':False,'health_clock':'CLOCK_MONOTONIC with time namespace; full supervisor health windows'})
 candidate.append(out/'stages.jsonl',{'event':'model_loading','unix':time.time()});wrapper=Qwen3VLChatEADP(**recipe)
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('timing model backend mismatch')
 generation.validate(prepared['runtime']['generation_source_sha256'],identity.sha);generation.validate_model_class(wrapper.model)
 for row in refs['rows']:
  real_worker.preflight_one(torch,wrapper,row,{'processor':row['processor_expected'],'image_start':row['image_start']})
 import gc
 gc.collect();torch.cuda.synchronize();candidate.append(out/'stages.jsonl',{'event':'model_ready','unix':time.time()})
 def author_generate(w,row):
  with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
 result=candidate.execute_schedule(refs['rows'],refs['records'],prepared['schedule'],contract,runner,torch,wrapper,pruner,adapter.BudgetLightAdapter,timer.measure,StoppingCriteriaList,author_generate,out/'journal.jsonl',generation.validate_model_class)
 admission.plan()
 generation.validate(prepared['runtime']['generation_source_sha256'],identity.sha)
 for path,before in prepared['preflight']['weight_stats'].items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight file changed')
 again=budget_reference.load_references()
 if again['records_sha256']!=refs['records_sha256']:raise ValueError('accepted reference changed')
 save(out/'worker_complete.json',dict(result,status='passed',source_unchanged=True,weights_stat_unchanged=True,journal_sha256=identity.sha(out/'journal.jsonl')))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args();out=args.out.resolve()
 approval=None
 if not args.preflight_only:approval=admission.admit_gpu()
 if not out.is_dir():raise ValueError('output directory missing')
 verify_container(out,approval)
 prepared=prepare(out)
 save(out/'worker_preflight.json',{'status':'passed','identity':prepared['identity'],'jobs':len(prepared['schedule']),
      'processor_preflight_sha256':identity.sha(out/'processor_preflight.json'),'plan_sha256':identity.sha(HERE/'pilot_plan001.json'),'runtime_plan_sha256':identity.sha(HERE/'runtime_plan001.json'),
      'generation_paths':prepared['generation_paths'],'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'cuda_initialized':False,'model_loaded':False})
 if not args.preflight_only:run_after_future_admission(out,prepared,approval)
if __name__=='__main__':main()
