"""Unmodified author forced-prefix control: no model forward hooks or patches."""
import argparse,hashlib,inspect,json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/media/disk2/YZX/research/EADP_v2');BASE=ROOT/'reproduction/20261104/deepstack_compression';HERE=BASE/'e6/ziprerank_nohook_v2';ZIP=BASE/'e6/external_sources/ziprerank_f367b09';AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
sys.path[:0]=[str(AUTHOR),str(ZIP)]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(path,obj):
 with open(path,'x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def progress(path,event):
 with open(path,'a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
def weight_stats(weights):
 out=[]
 for x in weights['files']:
  p=Path(x['path']);s=p.stat()
  if s.st_size!=x['bytes'] or x['sha256']!=x['expected_lfs_sha256']:raise ValueError('weight identity mismatch')
  out.append({'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev})
 if len(out)!=4:raise ValueError('weight count')
 return out
def preflight(plan):
 for p,h in plan['sha256'].items():
  if sha(p)!=h:raise ValueError('source mismatch: '+p)
 rows=[json.loads(x) for x in (BASE/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];a=plan['anchor'];found=[r for r in rows if r['task']==a['task'] and str(r['qid'])==a['qid']]
 if len(found)!=1:raise ValueError('anchor uniqueness')
 r=found[0];d=hashlib.sha256(json.dumps(r['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
 if d!=r['message_sha256'] or d!=a['message_sha256'] or sha(r['image_path'])!=a['image_sha256'] or [x['type'] for x in r['message']]!=['image','text']:raise ValueError('anchor fingerprint')
 return r
def hook_counts(model):
 import torch.nn.modules.module as module
 return sum(len(x._forward_pre_hooks)+len(x._forward_hooks) for x in model.modules())+len(module._global_forward_pre_hooks)+len(module._global_forward_hooks)
def run_arm(model,inputs,torch,processor,force_cls,name,enabled,ratio,out):
 model.model.rope_deltas=None;model.set_qi_early_enabled(enabled);model.set_qi_early_keep_ratio(ratio)
 if hook_counts(model)!=0:raise ValueError('model forward hooks present')
 seq=inputs['input_ids'].shape[1];token=processor.tokenizer.encode(' the',add_special_tokens=False)[0]
 class ForceToken(force_cls):
  def __call__(self,input_ids,scores):
   f=torch.full_like(scores,-float('inf'));f[:,token]=0;return f
 save(out/(name+'_started.json'),{'arm':name,'enabled':enabled,'keep_ratio':ratio,'model_forward_hooks':0});progress(out/'progress.jsonl',{'event':'arm_started','arm':name})
 try:
  with torch.inference_mode():
   result=model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=33,min_new_tokens=33,logits_processor=[ForceToken()],return_dict_in_generate=True)
  if torch.cuda.is_available():torch.cuda.synchronize()
  tokens=result.sequences[0,seq:].tolist();save(out/(name+'_generated.json'),{'arm':name,'tokens':tokens,'forced_token_id':token});progress(out/'progress.jsonl',{'event':'generated','arm':name})
  if hook_counts(model)!=0:raise ValueError('model or global forward hooks appeared')
  status={'arm':name,'status':'completed','forced_tokens_match':tokens==[token]*33,'generated_token_count':len(tokens),'model_forward_hooks_after':hook_counts(model)}
 except Exception as e:
  status={'arm':name,'status':'failed','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'model_forward_hooks_after':hook_counts(model)}
  save(out/(name+'_exception.json'),status);progress(out/'progress.jsonl',{'event':'exception','arm':name,'error_type':type(e).__name__})
 save(out/(name+'.json'),status);progress(out/'progress.jsonl',{'event':'arm_recorded','arm':name,'status':status['status']});return status
def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads((HERE/'plan001.json').read_text());row=preflight(plan)
 import torch,transformers
 from transformers import AutoProcessor,LogitsProcessor
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 from models.qwen3vl_with_qi_early import Qwen3VLWithQIEarly
 if Path(inspect.getfile(Qwen3VLWithQIEarly)).resolve()!=ZIP/'models/qwen3vl_with_qi_early.py':raise ValueError('author path')
 weights=json.loads((ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json').read_text());stats=weight_stats(weights)
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('CPU initialized CUDA')
  for x in weights['files']:
   if sha(x['path'])!=x['expected_lfs_sha256']:raise ValueError('weight SHA')
  if weight_stats(weights)!=stats:raise ValueError('weights changed')
  save(a.out,{'status':'passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,'source_plan_sha256':sha(HERE/'plan001.json'),'torch':torch.__version__,'transformers':transformers.__version__,'weight_sha256_verified':4,'weight_stats':stats});return
 previous=json.loads((HERE/'preflight001.json').read_text())
 if stats!=previous['weight_stats'] or torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('weight/GPU identity')
 a.out.mkdir(exist_ok=False);progress(a.out/'progress.jsonl',{'event':'worker_started'});save(a.out/'started.json',{'plan_sha256':sha(HERE/'plan001.json'),'task':row['task'],'qid':row['qid'],'GT_opened':False,'scoring':False})
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);model=Qwen3VLWithQIEarly.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval()
 wrapper=object.__new__(Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cuda',dtype=torch.bfloat16)
 inputs=wrapper._processor_inputs(wrapper._build_messages(row['message'],dataset=row['task']));result=[]
 for name,enabled,ratio in [('disabled',False,1.0),('keep_all',True,1.0),('keep_256',True,.25)]:result.append(run_arm(model,inputs,torch,processor,LogitsProcessor,name,enabled,ratio,a.out))
 if weight_stats(weights)!=stats or any(sha(p)!=h for p,h in plan['sha256'].items()):raise ValueError('source or weights changed during run')
 save(a.out/'worker_complete.json',{'status':'diagnostic_completed','arm_statuses':{x['arm']:x['status'] for x in result},'no_GT':True,'warning':'completion records attempts, not success; no hooks or timing claims'})
if __name__=='__main__':main()
