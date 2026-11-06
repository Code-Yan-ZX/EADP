"""192-image timing worker; future GPU execution requires independent evidence.

Missing plans, accepted 888 references or reviews fail closed.
"""
import argparse,hashlib,importlib.metadata,importlib.util,inspect,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
PIN={'gate.py':'ecfe5d7ef2c09932fe59e2836f0a8c9c1b34b44268d6329d1ddccea43fa63d3f','core.py':'5abf852da74f9042b9ffb674737014572e460d9245d3b0dd00573f0ca3636192','fixed_reference_gate.py':'17254049539f1db8f1a78cab606f9190342267f06d221f9c3bc1bf59e99e85f0','../e4_c_multibudget_worker_v3/runner.py':'e52b253571c430fc5a75660d43a94ae10b6bbfe5fd7308dc5e8c40f5a50a5ccb','../e4_c_multibudget_references192/references.py':'5088d92e0afbc4d9aaf01b1bf419f04b00e07d910b739a723e13c477470d9388','../e4_c_budget_ttft_pilot_v3/generation_identity.py':'6e1a50340bf9a01de6d73603b3e123881cc195099bb6de48471b42995ba08518','../e6/pixel_natural_timing_pilot_v4/bootstrap.py':'ab81e2f48f995403afc21ab6ab4875c8b4ae4e9bcca72be063ab8fc42a53fdbc'}
PIN['../e4_c_budget_pilot_v2/adapter.py']='9ad1e377a8c390b9f8d224ec423e0cd0317b4c391eaf9e2c4ebfbfc864170c53'
PIN['gpu_gate.py']='f31b9de0a7ebc7e3769e9581dc45b9fbeba03e36e624776d29e45bc9667ce9f2'
PIN['cpu_gate.py']='d10e6a6237a92622f159ffa81accdf8dad5c55800ff659dd2cf4fbb1c060d870'
PIN['../e4_c_multibudget_references192/preflight.py']='7f20121c54ef75d3d8b504c3b7e0b7deab7afeedcf27c849dc52402be41dc051'
def load(name,relative):
 path=(D/relative).resolve()
 if hashlib.sha256(path.read_bytes()).hexdigest()!=PIN[relative] or name in sys.modules:raise ValueError('unreviewed/preloaded module '+name)
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:del sys.modules[name];raise
 return module
def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def append(path,obj):
 with Path(path).open('a') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def execute_candidate(out):
 # This locked function must be reviewed again if gate.py or any pin changes.
 gate=load('_timing192_gpu_gate','gpu_gate.py');receipt=gate.admit(host=False)
 verify_container(out,receipt)
 pre=load('_timing192_ref_gpu_preflight','../e4_c_multibudget_references192/preflight.py')
 reference_plan=json.loads((B/'e4_c_multibudget_references192/execution_plan001.json').read_text());cpu=pre.prepare(reference_plan,out/'processor_preflight.json',require_no_gpu=False)
 if cpu.get('inputs_checked')!=222 or cpu.get('references_checked')!=888 or cpu.get('cuda_initialized') is not False or cpu.get('model_loaded') is not False:raise ValueError('runtime input preflight mismatch')
 # The future receipt must be bound by the gate to real supervised exits,
 # source/weight identities, 888 accepted references and the frozen proposal.
 if not isinstance(receipt,dict) or receipt.get('status')!='passed':raise ValueError('missing independently verified admission')
 proposal=json.loads((B/'e4_c_multibudget_expansion192/proposal002.json').read_text());boot=load('_timing192_bootstrap','../e6/pixel_natural_timing_pilot_v4/bootstrap.py');boot.pin_all()
 refs_module=load('_timing192_refs','../e4_c_multibudget_references192/references.py');fixed_gate=load('_timing192_fixed_gate','fixed_reference_gate.py');core=load('_timing192_core','core.py');runner=load('_timing192_runner','../e4_c_multibudget_worker_v3/runner.py');generation=load('_timing192_generation','../e4_c_budget_ttft_pilot_v3/generation_identity.py')
 refs=refs_module.load_references();fixed=fixed_gate.load(proposal);core.prepare(proposal,refs['rows'],refs['records'],fixed)
 # No CUDA use, model construction or new generation occurs before the above.
 import torch,real_worker
 from transformers.generation.stopping_criteria import StoppingCriteriaList
 author=json.loads(Path(real_worker.PLAN).read_text());recipe=author['recipe'];expected={'visual_token_num':256,'alpha':.5,'beta':2.,'min_pixels':1048576,'max_pixels':1048576,'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
 if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('recipe mismatch')
 sys.path[:0]=[author['author_root'],str(Path(author['author_root'])/'VLMEvalKit')]
 from model import pruner
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
 if {n:importlib.metadata.version(n) for n in author['expected_versions']}!=author['expected_versions'] or Path(pruner.__file__).resolve()!=(Path(author['author_root'])/'model/pruner.py').resolve() or Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=(Path(author['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve() or torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('model/runtime mismatch')
 multibudget_light_adapter=load('multibudget_light_adapter','../e4_c_budget_pilot_v2/adapter.py')
 generation_plan=json.loads((B/'e4_c_budget_ttft_pilot_v3/runtime_plan001.json').read_text());generation.validate(generation_plan['generation_source_sha256'],lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
 save(out/'frozen.json',{'stage':'192-image exploratory timing candidate','approval':receipt,'schedule':proposal['jobs'],'E4_records_sha256':refs['records_sha256'],'weight_stats_before':cpu['weight_stats'],'GT_opened':False,'scoring_performed':False,'predictions_opened':True})
 wrapper=Qwen3VLChatEADP(**recipe)
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend mismatch')
 generation.validate_model_class(wrapper.model)
 for row in refs['rows']:real_worker.preflight_one(torch,wrapper,row,{'processor':row['processor_expected'],'image_start':row['image_start']})
 import gc;gc.collect();torch.cuda.synchronize()
 def author_generate(w,row):
  with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
 def execute(row,budget,protocol,panel,raw):return runner.execute(torch,wrapper,pruner,row,budget,protocol,panel,multibudget_light_adapter.BudgetLightAdapter,StoppingCriteriaList,author_generate,raw,generation.validate_model_class)
 result=core.run(proposal,refs['rows'],refs['records'],fixed,execute,lambda e:append(out/'journal.jsonl',e))
 gate.plan();generation.validate(generation_plan['generation_source_sha256'],lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
 for path,before in cpu['weight_stats'].items():
  stat=Path(path).stat()
  if [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]!=before:raise ValueError('weights changed')
 if refs_module.load_references()['records_sha256']!=refs['records_sha256']:raise ValueError('accepted E4 records changed')
 save(out/'worker_complete.json',dict(result,status='passed',source_unchanged=True,weights_stat_unchanged=True,journal_sha256=hashlib.sha256((out/'journal.jsonl').read_bytes()).hexdigest()))
def verify_container(out,approval):
 deadline=time.monotonic()+30;identity=None
 while time.monotonic()<deadline:
  try:identity=json.loads((out/'container_identity.json').read_text());break
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
 if identity is None:raise ValueError('container identity missing')
 launch=json.loads((out/'launch.json').read_text());cid=identity.get('container_id')
 if launch.get('approval_bundle')!=approval or launch.get('run_out')!=str(out) or launch.get('worker_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or launch.get('time_namespace')!=os.readlink('/proc/self/ns/time') or not isinstance(cid,str) or len(cid)!=64 or cid not in Path('/proc/self/cgroup').read_text() or cid not in identity.get('host_cgroup','') or not (identity.get('host_init') or {}).get('start_ticks'):raise ValueError('GPU container identity mismatch')
def cpu_preflight(out,approval,cpu_gate):
 # The launcher writes and fsyncs both records; bounded retry handles atomicity.
 deadline=time.monotonic()+30;identity=None
 while time.monotonic()<deadline:
  try:identity=json.loads((out/'container_identity.json').read_text());break
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
 if identity is None:raise ValueError('container identity missing')
 launch=json.loads((out/'launch.json').read_text());cid=identity.get('container_id')
 if launch.get('approval_bundle')!=approval or launch.get('run_out')!=str(out) or launch.get('worker_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or not isinstance(cid,str) or len(cid)!=64 or cid not in Path('/proc/self/cgroup').read_text() or cid not in identity.get('host_cgroup','') or not (identity.get('host_init') or {}).get('start_ticks'):raise ValueError('CPU container identity mismatch')
 pre=load('_timing192_ref_cpu_preflight','../e4_c_multibudget_references192/preflight.py')
 reference_plan=json.loads((B/'e4_c_multibudget_references192/execution_plan001.json').read_text());v=pre.prepare(reference_plan,out/'processor_preflight.json',require_no_gpu=True)
 if v.get('status')!='passed' or v.get('inputs_checked')!=222 or v.get('references_checked')!=888 or v.get('cuda_initialized') is not False or v.get('model_loaded') is not False or v.get('GT_opened') is not False:raise ValueError('CPU processor preflight mismatch')
 cpu_gate.plan();save(out/'worker_preflight.json',{'status':'passed','stage':'192 timing worker CPU-only preflight','plan_sha256':approval['plan_sha256'],'processor_preflight_sha256':hashlib.sha256((out/'processor_preflight.json').read_bytes()).hexdigest(),'inputs':222,'jobs':7920,'existing_natural_references':888,'fixed32_new_references_opened':False,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'prediction_scope':'existing accepted E1/E4 DEV mixed predictions parsed for identity only','cuda_initialized':False,'model_loaded':False,'GPU_authorized':False})
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.preflight_only:
  cg=load('_timing192_cpu_gate','cpu_gate.py');approval=cg.admit();out=a.out.resolve()
  if not out.is_dir():raise ValueError('launcher output missing')
  cpu_preflight(out,approval,cg);return
 out=a.out.resolve()
 if not out.is_dir():raise ValueError('launcher output missing')
 execute_candidate(out)
if __name__=='__main__':main()
