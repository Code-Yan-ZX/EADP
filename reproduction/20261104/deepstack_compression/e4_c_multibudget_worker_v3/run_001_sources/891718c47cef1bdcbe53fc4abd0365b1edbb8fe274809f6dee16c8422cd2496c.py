"""No-GPU processor/dependency preflight for the proposed 32-image pilot.

Requires a frozen source plan from a separately reviewed launcher.  This file
cannot authorize or invoke any model or GPU execution.
"""
import argparse,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[2]
AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def pin(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign import '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def check_sources(plan):
 sources=plan['source_sha256']
 if not sources:raise ValueError('empty source inventory')
 for path,digest in sources.items():
  if sha(path)!=digest:raise ValueError('source changed '+path)
def prepare(plan,out,require_no_gpu=True):
 if require_no_gpu:os.environ['CUDA_VISIBLE_DEVICES']=''
 check_sources(plan)
 boot=pin('_multi_budget_bootstrap',BASE/'e6/pixel_natural_timing_pilot_v4/bootstrap.py');boot.pin_all()
 core=pin('_multi_budget_core',HERE/'worker_core.py');refs_module=pin('_multi_budget_references',HERE/'references.py')
 contract=pin('_multi_budget_contract',BASE/'e4_c_multibudget_expansion/pilot_contract.py')
 runner=pin('_multi_budget_runner',HERE/'runner.py')
 generation=pin('_multi_budget_generation',BASE/'e4_c_budget_ttft_pilot_v3/generation_identity.py')
 if not callable(runner.execute):raise ValueError('timing primitive missing')
 import torch,transformers,input_preflight,real_worker
 from transformers import AutoProcessor,AutoConfig
 sys.path[:0]=[str(AUTHOR)]
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 if torch.cuda.is_initialized() or Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=(AUTHOR/'vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve() or transformers.__version__!='4.57.1':raise ValueError('unexpected processor/runtime identity')
 sample_plan=json.loads((BASE/'e4_c_multibudget_expansion/pilot_plan001.json').read_text())
 if sha(BASE/'e4_c_multibudget_expansion/pilot_plan001.json')!=plan['pilot_plan_sha256']:raise ValueError('sample plan changed')
 frozen=json.loads((BASE/'e4_c_latency_full/full_001/frozen.json').read_text())
 sample_plan['_validated_full_cohort']=frozen['schedule']['cohort']
 refs=refs_module.load_references();core.prepare(sample_plan,refs['rows'],refs['records'],contract)
 runtime=json.loads((BASE/'e4_c_budget_ttft_pilot_v3/runtime_plan001.json').read_text());generation_paths=generation.validate(runtime['generation_source_sha256'],sha)
 author_plan=json.loads(real_worker.PLAN.read_text());weights=real_worker.source_and_weights(author_plan)
 model=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
 processor=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 wrapper=object.__new__(Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
 checked=[]
 for row in refs['rows']:
  actual=wrapper._processor_inputs(wrapper._build_messages(row['message'],dataset=row['task']))
  input_preflight.validate_processor(torch,actual,row['processor_expected'],config.image_token_id,row['image_start'])
  if not bool(torch.all(actual['attention_mask']==1)):raise ValueError('padded batch violates delta contract')
  checked.append({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'image_start':row['image_start'],'processor':row['processor_expected']})
  del actual
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized in no-GPU preflight')
 check_sources(plan)
 result={'status':'passed','stage':'no-GPU input and processor preflight only; no GPU authority','sources_checked':len(plan['source_sha256']),
         'pilot_plan_sha256':plan['pilot_plan_sha256'],'generation_paths':generation_paths,'weight_stats':weights,'weight_payloads_hashed':4,
         'inputs_checked':len(checked),'processor_inputs':checked,'references_checked':len(refs['records']),'jobs':len(sample_plan['jobs']),
         'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,
         'prediction_scope':'previously accepted E1 mixed DEV journal and E4 DEV natural outputs for processor and same-arm identity',
         'fixed32_model_identity':'not run; mandatory before GPU timing','GPU_authorized':False}
 with Path(out).open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 result=prepare(json.loads(args.plan.read_text()),args.out);print(json.dumps({'status':result['status'],'inputs':result['inputs_checked'],'jobs':result['jobs']}),flush=True)
