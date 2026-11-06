"""Three-anchor, same-process original-vs-graph warm TTFT pilot.

This is exploratory, not the preregistered full M1 experiment. It uses the
existing E4-C timing boundary and P3/DS-on intervention. No scores or GT.
"""
import argparse,importlib.util,json,os,random,sys,time
from pathlib import Path
ROOT=Path('/media/disk2/YZX/research/EADP_v2');BASE=ROOT/'reproduction/20261104/deepstack_compression';HERE=BASE/'e5/graph_ttft_pilot_v2'
sys.path[:0]=[str(ROOT/'_reproduction/author_eadp/Qwen_vl'),str(BASE/'e4_c_latency'),str(BASE/'e5/fused_selector'),str(BASE/'e5/graph_selector'),str(BASE/'e6/ziprerank_qtext_smoke_v2')]
from real_worker import sha,source_and_weights,preflight_one
from input_preflight import load_accepted_e1_references
from runner import execute_one
spec=importlib.util.spec_from_file_location('zip_diag_helper',BASE/'e6/ziprerank_qtext_smoke_v2/worker.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
save,progress=diag.save,diag.progress

def weight_stats(weights):
 result=diag.weight_stats(weights)
 for row in result:row['ctime_ns']=Path(row['path']).stat().st_ctime_ns
 return result

def preflight(plan):
 for path,expected in plan['sha256'].items():
  if sha(path)!=expected:raise ValueError('source mismatch: '+path)
 rows=[json.loads(x) for x in (BASE/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];out=[]
 for anchor in plan['anchors']:
  matches=[x for x in rows if x['task']==anchor['task'] and str(x['qid'])==anchor['qid']]
  if len(matches)!=1:raise ValueError('anchor uniqueness')
  x=matches[0]
  if x['group_id']!=anchor['group_id'] or x['message_sha256']!=anchor['message_sha256'] or sha(x['image_path'])!=anchor['image_sha256']:raise ValueError('anchor mismatch')
  out.append(x)
 return out

def validate_import_paths(mapping):
 for module,expected in mapping.items():
  if Path(module.__file__).resolve()!=Path(expected).resolve():raise ValueError('timing module path mismatch')


def durable_generation(original,emit):
 def call(*args,**kwargs):
  result=original(*args,**kwargs)
  emit(result.detach().cpu().tolist())
  return result
 return call


def compare_pair(a,b):
 if {a['job']['method'],b['job']['method']}!={'original','graph'}:raise ValueError('pair must contain both methods')
 keys=('phase','repeat','task','qid','protocol')
 if any(a['job'][k]!=b['job'][k] for k in keys):raise ValueError('pair identity mismatch')
 same_order=a['ordered_indices']==b['ordered_indices'];same_tokens=a['result']['output_ids']==b['result']['output_ids']
 return {'job':{k:a['job'][k] for k in keys},'ordered_indices_equal':same_order,'output_ids_equal':same_tokens,
         'eligible_for_equal_output_timing':same_order and same_tokens}


def make_schedule(anchors):
 # Same rows and protocol for both methods; seeded random ordering; never drop.
 rng=random.Random(20261105);out=[]
 for phase,repeats in [('warmup',1),('measured',5)]:
  for rep in range(repeats):
   tasks=list(anchors);rng.shuffle(tasks)
   for row in tasks:
    methods=['original','graph'];rng.shuffle(methods)
    for method in methods:out.append({'phase':phase,'repeat':rep,'task':row['task'],'qid':row['qid'],'method':method,'protocol':'fixed32'})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 plan=json.loads((HERE/'plan001.json').read_text());anchors=preflight(plan)
 import torch,transformers
 from model import pruner as native
 from graph import GraphSelector
 from kernel import fused_greedy
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP
 import inspect,graph,kernel,runner,real_worker,input_preflight
 if Path(native.__file__).resolve()!=(ROOT/'_reproduction/author_eadp/Qwen_vl/model/pruner.py').resolve():raise ValueError('author pruner path mismatch')
 if Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=(ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve():raise ValueError('wrapper path mismatch')
 if Path(graph.__file__).resolve()!=(BASE/'e5/graph_selector/graph.py').resolve() or Path(kernel.__file__).resolve()!=(BASE/'e5/fused_selector/kernel.py').resolve():raise ValueError('graph/kernel path mismatch')
 if transformers.__version__!='4.57.1':raise ValueError('transformers mismatch')
 validate_import_paths({runner:BASE/'e4_c_latency/runner.py',real_worker:BASE/'e4_c_latency/real_worker.py',input_preflight:BASE/'e4_c_latency/input_preflight.py'})
 weights=json.loads((ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json').read_text());stats=weight_stats(weights)
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
  for item in weights['files']:
   if sha(item['path'])!=item['expected_lfs_sha256']:raise ValueError('weight SHA mismatch')
  if weight_stats(weights)!=stats:raise ValueError('weights changed while hashing')
  save(a.out,{'status':'passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,'source_plan_sha256':sha(HERE/'plan001.json'),
              'torch':torch.__version__,'transformers':transformers.__version__,'weight_sha256_verified':4,'weight_stats':stats,'schedule':make_schedule(plan['anchors'])});return
 if stats!=json.loads((HERE/'preflight001.json').read_text())['weight_stats']:raise ValueError('weights changed')
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('GPU mismatch')
 a.out.mkdir(exist_ok=False);progress(a.out/'progress.jsonl',{'event':'worker_started'})
 save(a.out/'started.json',{'plan_sha256':sha(HERE/'plan001.json'),'GT_opened':False,'scoring':False})
 e1=BASE/'e1';allrows,refs=load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json')
 lookup={(r['task'],str(r['qid'])):r for r in allrows};author_plan=json.loads((ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())
 wrapper=Qwen3VLChatEADP(**author_plan['recipe'])
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend mismatch')
 for row in anchors:
  key=(row['task'],str(row['qid']));accepted=lookup[key]
  if any(row.get(k)!=accepted.get(k) for k in ('group_id','message_sha256','image_sha256')):raise ValueError('accepted E1 identity mismatch')
  row['image_start']=refs[key]['image_start'];preflight_one(torch,wrapper,row,refs[key])
 torch.cuda.synchronize();s=torch.rand((1024,1024),device='cuda');w=torch.rand((1024,),device='cuda')
 with torch.inference_mode():
  fused_greedy(s,w,256);torch.cuda.synchronize();before={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
  start=time.perf_counter();graph=GraphSelector(s,w,256);torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
  after={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
 save(a.out/'graph_setup.json',{'capture_wall_ms':elapsed,'before_bytes':before,'after_bytes':after,'setup_excluded_from_timing':True})
 bykey={(r['task'],str(r['qid'])):r for r in anchors};original=native._greed_select_impl;pairs={};comparisons=[]
 for job in make_schedule(plan['anchors']):
  row=bykey[(job['task'],str(job['qid']))];observed=[]
  def intercepted(importance,similarity,token_num):
   if importance.shape!=(1,1024) or similarity.shape!=(1,1024,1024) or token_num!=256:raise ValueError('unsupported actual selector shape')
   if job['method']=='graph':
    indices=graph(similarity[0],importance[0]).unsqueeze(0);quotas=torch.full((1,),256,dtype=torch.long,device=indices.device);result=(indices,quotas)
   else:result=original(importance,similarity,token_num)
   observed.append(result[0]);return result
  progress(a.out/'progress.jsonl',{'event':'job_started','job':job});native._greed_select_impl=intercepted
  had_generate='generate' in wrapper.model.__dict__;old_generate=wrapper.model.__dict__.get('generate');original_generate=wrapper.model.generate
  wrapper.model.generate=durable_generation(original_generate,lambda ids:progress(a.out/'generated_raw.jsonl',{'job':job,'output_ids':ids}))
  try:
   with torch.inference_mode():result=execute_one(wrapper,native,torch,row,'C-ON',job['protocol'],torch.cuda.synchronize,
      on_partial=lambda part:progress(a.out/'partial.jsonl',{'job':job,**part}))
   progress(a.out/'generated.jsonl',{'job':job,'result':result})
   if len(observed)!=1:raise ValueError('expected exactly one selector call')
   record={'job':job,'status':'completed','ordered_indices':observed[0].reshape(-1).tolist(),'result':result}
  except Exception as exc:
   record={'job':job,'status':'failed','error_type':type(exc).__name__,'error':str(exc)}
   progress(a.out/'progress.jsonl',{'event':'failed','record':record});raise
  finally:
   native._greed_select_impl=original
   if had_generate:wrapper.model.generate=old_generate
   else:delattr(wrapper.model,'generate')
  progress(a.out/'records.jsonl',record);progress(a.out/'progress.jsonl',{'event':'job_completed','job':job})
  pair_key=(job['phase'],job['repeat'],job['task'],job['qid'],job['protocol'])
  if pair_key in pairs:
   comparison=compare_pair(pairs.pop(pair_key),record);comparisons.append(comparison);progress(a.out/'comparisons.jsonl',comparison)
  else:pairs[pair_key]=record
 if pairs or len(comparisons)!=18:raise ValueError('incomplete comparison inventory')
 if weight_stats(weights)!=stats:raise ValueError('weights changed')
 save(a.out/'worker_complete.json',{'status':'completed','jobs':len(make_schedule(plan['anchors'])),'no_GT':True,'scoring':False,
  'comparisons':len(comparisons),'unequal_comparisons':sum(not c['eligible_for_equal_output_timing'] for c in comparisons),
  'scope':'three anchors, fixed32, warm setup excluded; any mismatch excludes equal-output aggregate timing claim; raw output fsync happens after first-token boundary and affects end-to-end time; no extrapolation to production or accuracy'})
if __name__=='__main__':main()
