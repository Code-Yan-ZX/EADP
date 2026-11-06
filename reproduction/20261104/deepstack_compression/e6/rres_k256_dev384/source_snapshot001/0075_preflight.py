"""Four-budget diagnostic CPU preflight. No GPU admission or model loading."""
import argparse, importlib.util, inspect, json, os, sys
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[2]
PILOT=BASE/'e6/pixel_natural_timing_pilot_v4'
AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit'

def load(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:del sys.modules[name];raise
 return module

def check_sources(plan):
 import hashlib
 for name,digest in plan['source_sha256'].items():
  h=hashlib.sha256()
  with open(name,'rb') as f:
   for block in iter(lambda:f.read(1048576),b''):h.update(block)
  if h.hexdigest()!=digest:raise ValueError('source mismatch '+name)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 plan=json.loads((HERE/'preflight_plan001.json').read_text());check_sources(plan)
 boot=load('_budget_preflight_bootstrap',PILOT/'bootstrap.py');boot.pin_all()
 for name,path in [('adapter',HERE/'adapter.py'),('budget_capture',HERE/'budget_capture.py'),('budget_arm',HERE/'budget_arm.py'),
                   ('budget_compare',HERE/'budget_compare.py'),('diagnostic_core',HERE/'diagnostic_core.py'),
                   ('off_adapter',BASE/'e4/off_adapter.py'),('budget_adapter',BASE/'e4/budget_adapter.py'),
                   ('identity_records',BASE/'e4/identity_records.py'),('decode_observer',BASE/'e4/decode_observer.py'),
                   ('budget_runner',BASE/'e4/budget_runner.py')]:load(name,path)
 candidate=load('budget_worker_candidate',HERE/'worker_candidate.py')
 candidate.force_code(sys.modules['budget_arm'].run_arm);candidate.force_code(sys.modules['budget_runner'].run_arm)
 reference=load('budget_reference',HERE/'reference.py')
 refs=reference.load_references()
 import torch,transformers,input_preflight,real_worker
 from transformers import AutoProcessor,AutoConfig
 sys.path[:0]=[str(AUTHOR)]
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 if torch.cuda.is_initialized():raise ValueError('CUDA already initialized')
 if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=(AUTHOR/'vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve():raise ValueError('foreign wrapper')
 if transformers.__version__!='4.57.1':raise ValueError('unexpected transformers')
 author_plan=json.loads(real_worker.PLAN.read_text());weights=real_worker.source_and_weights(author_plan)
 model=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
 processor=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 wrapper=object.__new__(Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
 anchors=[]
 for row in refs['rows']:
  actual=wrapper._processor_inputs(wrapper._build_messages(row['message'],dataset=row['task']))
  input_preflight.validate_processor(torch,actual,row['processor_expected'],config.image_token_id,row['image_start'])
  if not bool(torch.all(actual['attention_mask']==1)):raise ValueError('delta derivation requires unpadded batch')
  anchors.append({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'image_start':row['image_start'],'attention_mask_all_one':True})
  del actual
 if torch.cuda.is_initialized():raise ValueError('unexpected CUDA initialization')
 check_sources(plan)
 result={'status':'passed','scope':'no-GPU model-before-load processor/dependency preflight; GPU remains locked',
         'plan_sha256':reference.sha(HERE/'preflight_plan001.json'),'sources_checked':len(plan['source_sha256']),
         'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,
         'prediction_scope':'accepted E1 mixed DEV journal and accepted E4 DEV outputs for input and same-arm identity',
         'same_arm_references':len(refs['records']),'anchors':anchors,'weight_stats':weights,'weight_payloads_hashed':4,
         'force_processor_code_identities_resolved':2,
         'torch':torch.__version__,'transformers':transformers.__version__,
         'delta_contract':'batch 1 with unpadded attention: full delta = max(full position)+1-original length; adjusted=full delta+1024-K',
         'keep_all_contract':'K1024 uses identity indices and zero greedy calls; real-model verification pending'}
 with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','anchors':len(anchors),'references':len(refs['records'])}),flush=True)

if __name__=='__main__':main()
