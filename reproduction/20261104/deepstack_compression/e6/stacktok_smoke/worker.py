"""Three-anchor local Qwen3 StackTok identity smoke; independently gated.

Directly loads the unchanged upstream selector module; this is not the author's
Qwen3 integration. Prior DEV mixed prediction journals are opened for input
identity only. No reference answers or scoring are used.
"""
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
sys.path[:0]=[str(BASE/'e4_c_latency'),str(BASE/'e6/stacktok_audit/prototype'),str(BASE/'e4_native'),str(BASE/'e4')]
from admission import admit,sources,A
from latency_worker import prepare,load_runtime,save
from real_worker import sha
from local_adapter import user_question

def upstream_selector():
 path=(A/'stacktok/core/selector.py').resolve();name='_stacktok_original_selector_571bdc6'
 if sha(path)!='329e3a80cef25ec09f159954a4dd30e7ec7a7dbfe03f394e15727a6e8f3e42cc' or name in sys.modules:raise ValueError('author selector source or namespace mismatch')
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:del sys.modules[name];raise
 if Path(module.__file__).resolve()!=path:raise ValueError('author selector module path')
 return module.StackTokSelector

def anchors(prepared):
 expected=json.loads((HERE/'plan001.json').read_text())['anchors'];out=[]
 for item in expected:
  matches=[r for r in prepared['rows'] if r['task']==item['task'] and str(r['qid'])==item['qid']]
  if len(matches)!=1 or any(matches[0].get(k)!=item[k] for k in ('group_id','image_sha256','message_sha256')):raise ValueError('anchor identity mismatch')
  out.append(matches[0])
 return out

def emit(path,obj):
 with path.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def ensure_path_ready(path,pair):
 if path=='local_stacktok':
  if len(pair)!=2:raise ValueError('native keepall control incomplete')
  from arm import compare_keepall
  compare_keepall(*pair)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();out=a.out.resolve()
 approval=admit(preflight_only=a.preflight_only,host=False);launch=json.loads((out/'launch.json').read_text())
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=sha(__file__):raise ValueError('launcher or approval identity mismatch')
 source={str(x.resolve()):sha(x) for x in sources()}
 if source!=approval['source_sha256']:raise ValueError('source changed')
 prepared=prepare('smoke');rows=anchors(prepared);torch,_native,wrapper_class,versions=load_runtime(prepared);Selector=upstream_selector()
 import arm,stock_capture,verify,inspect
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 expected_wrapper=(Path(prepared['plan']['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
 if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=expected_wrapper:raise ValueError('native FixedRes source mismatch')
 for module,name in ((arm,'arm.py'),(stock_capture,'stock_capture.py'),(verify,'verify.py')):
  if Path(module.__file__).resolve()!=HERE/name:raise ValueError('actual generation import path mismatch: '+name)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(prepared['plan']['recipe']['model_path'],local_files_only=True)
 words=[]
 for row in rows:
  question=user_question(row['message']);ids=tok(question.split(),is_split_into_words=True,return_tensors='pt')['input_ids']
  if ids.ndim!=2 or ids.shape[0]!=1 or not ids.numel():raise ValueError('tokenizer word split invalid')
  words.append({'task':row['task'],'qid':str(row['qid']),'tokens':int(ids.numel()),'bos_token_id':tok.bos_token_id})
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
  save(out/'worker_preflight.json',{'status':'passed','images':3,'source_sha256':source,'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'predictions_opened':True,'prediction_scope':'E1 DEV mixed journal parsed solely to bind input identity','weight_stats_before':prepared['weights_before'],'versions':versions,'tokenization':words,'upstream_selector_path':str(Path(sys.modules[Selector.__module__].__file__).resolve())});return
 from arm import run_local,compare_keepall,validate_recipe,validate_fingerprints
 from stock_capture import run_native
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 torch.cuda.set_per_process_memory_fraction(.9,0);recipe=dict(prepared['plan']['recipe']);recipe['visual_token_num']=1024
 start=time.monotonic()
 try:
  w=wrapper_class(**recipe)
  if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend mismatch')
 except BaseException as e:
  save(out/'model_load_failure.json',{'status':'failed','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'model_load_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'GT_opened':False});raise
 eos=w.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or [])
 if not eos:raise ValueError('native generation EOS missing')
 fixed=tok.encode(' the',add_special_tokens=False)[0];save(out/'frozen.json',{'anchors':json.loads((HERE/'plan001.json').read_text())['anchors'],'source_sha256':source,'weight_stats_before':prepared['weights_before'],'versions':versions,'model_load_seconds':time.monotonic()-start,'GPU_allocator_fraction':.9,'GT_opened':False,'scoring_performed':False,'predictions_opened':True});(out/'arrays').mkdir()
 def generate(model,row):
  with torch.inference_mode():return Qwen3VLChatFixedRes.generate_inner_transformers_fixed(model,row['message'],dataset=row['task'])
 def generated(rec):emit(out/'generated.jsonl',rec);emit(out/'journal.jsonl',{'event':'generated','task':rec['task'],'qid':rec['qid'],'path':rec['path'],'forced':rec['forced']})
 progress={'steps':0}
 def heartbeat(_m,_args,_kwargs,_out):
  progress['steps']+=1
  if progress['steps']%32==0:emit(out/'journal.jsonl',{'event':'decode_progress','steps':progress['steps'],'unix':time.time()})
 progress_hook=w.model.model.language_model.register_forward_hook(heartbeat,with_kwargs=True)
 try:
  count=0
  for row in rows:
   for forced in (False,True):
    pair=[]
    for path in ('stock_native','local_keepall','local_stacktok'):
     ensure_path_ready(path,pair)
     emit(out/'journal.jsonl',{'event':'started','task':row['task'],'qid':str(row['qid']),'path':path,'forced':forced,'unix':time.time()})
     key=hashlib.sha256(json.dumps([row['task'],str(row['qid']),path,forced],separators=(',',':')).encode()).hexdigest()
     def archive(tensors):
      import numpy as np
      dst=out/'arrays'/(key+'.npz');data={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
      with dst.open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
      return {'kind':'full_arrays','path':str(dst),'sha256':sha(dst)}
     token=fixed if forced else None
     if path=='stock_native':rec=run_native(torch,w,row,generate,generated,archive,forced_token=token)
     else:
      k=1024 if path=='local_keepall' else 256;rec=run_local(torch,w,row,Selector(target_vision_tokens=k),k,generate,generated,archive,forced_token=token)
     validate_recipe(rec,recipe,forced,eos)
     full=rec['processor']['input_ids']['shape'][1];validate_fingerprints(rec['cache_fingerprints'],forced,full-(0 if path!='local_stacktok' else 768))
     emit(out/'records.jsonl',rec);count+=1;emit(out/'journal.jsonl',{'event':'recorded','task':row['task'],'qid':str(row['qid']),'path':path,'forced':forced})
     if path in ('stock_native','local_keepall'):pair.append(rec)
     if path=='local_keepall':
      compare_keepall(*pair);emit(out/'comparisons.jsonl',{'task':row['task'],'qid':str(row['qid']),'forced':forced,'status':'passed'})
  if count!=18:raise ValueError('generation inventory incomplete')
  for path,before in prepared['weights_before'].items():
   s=Path(path).stat()
   if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weights changed')
  if source!={str(x.resolve()):sha(x) for x in sources()}:raise ValueError('source changed during run')
  progress_hook.remove()
  save(out/'worker_complete.json',{'status':'completed','generations':count,'comparisons':6,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'finished_unix':time.time()})
 except BaseException as e:
  progress_hook.remove()
  emit(out/'journal.jsonl',{'event':'failed','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});raise
if __name__=='__main__':main()
