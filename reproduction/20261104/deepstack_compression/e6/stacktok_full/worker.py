"""K256 local Qwen3 StackTok DEV panel, immutable 384 rows; no GT/score."""
import argparse,importlib.util,inspect,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];sys.path[:0]=[str(HERE),str(BASE/'e4_c_latency'),str(BASE/'e6/stacktok_audit/prototype'),str(BASE/'e4')]
from admission import admit,sources,A
from latency_worker import prepare,load_runtime,save
from real_worker import sha
from full_refs import e4_records,validate_reference
from full_arm import run_local

def author_selector():
 path=(A/'stacktok/core/selector.py').resolve();name='_stacktok_original_selector_full_571bdc6'
 if sha(path)!='329e3a80cef25ec09f159954a4dd30e7ec7a7dbfe03f394e15727a6e8f3e42cc' or name in sys.modules:raise ValueError('author selector identity')
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
 try:spec.loader.exec_module(mod)
 except BaseException:del sys.modules[name];raise
 if Path(mod.__file__).resolve()!=path:raise ValueError('author selector path')
 return mod.StackTokSelector

def append(path,row):
 with Path(path).open('a') as f:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def execute_rows(rows,refs,out,run_one):
 if len(rows)!=384 or len(refs)!=384 or len({(r['task'],str(r['qid']),r['group_id']) for r in rows})!=384:raise ValueError('expected 384 distinct inputs')
 count=0
 for row in rows:
  key=(row['task'],str(row['qid']));append(out/'journal.jsonl',{'event':'started','task':key[0],'qid':key[1],'group_id':row['group_id'],'unix':time.time()})
  generated_count=[0];returned_count=[0]
  def returned(rec):
   returned_count[0]+=1
   if returned_count[0]!=1 or (rec.get('task'),str(rec.get('qid')),rec.get('group_id'))!=(key[0],key[1],row['group_id']):raise ValueError('returned output identity')
   append(out/'returned.jsonl',rec);append(out/'journal.jsonl',{'event':'returned','task':rec['task'],'qid':str(rec['qid'])})
  def generated(rec):
   generated_count[0]+=1
   if generated_count[0]!=1 or (rec.get('task'),str(rec.get('qid')),rec.get('group_id'))!=(key[0],key[1],row['group_id']):raise ValueError('generated output identity')
   append(out/'generated.jsonl',rec);append(out/'journal.jsonl',{'event':'generated','task':rec['task'],'qid':str(rec['qid'])})
  try:
   rec=run_one(row,refs[key],returned,generated)
   if returned_count!=[1] or generated_count!=[1]:raise ValueError('generation inventory')
   validate_reference(rec,refs[key]);append(out/'records.jsonl',rec);append(out/'journal.jsonl',{'event':'completed','task':key[0],'qid':key[1],'group_id':row['group_id']});count+=1
  except BaseException as e:
   append(out/'journal.jsonl',{'event':'failed','task':key[0],'qid':key[1],'group_id':row['group_id'],'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()});raise
 return count

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();out=a.out.resolve();approval=admit(preflight_only=a.preflight_only,host=False);launch=json.loads((out/'launch.json').read_text());source={str(x.resolve()):sha(x) for x in sources()}
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=sha(__file__) or source!=approval['source_sha256']:raise ValueError('source or launch mismatch')
 prepared=prepare('smoke');rows=prepared['rows'];refs=e4_records(rows);torch,_native,wrapper_class,versions=load_runtime(prepared);Selector=author_selector()
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 wrapper_path=(Path(prepared['plan']['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
 if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=wrapper_path:raise ValueError('stock wrapper path')
 import full_adapter,full_arm,full_refs
 if any(Path(m.__file__).resolve()!=HERE/name for m,name in ((full_adapter,'full_adapter.py'),(full_arm,'full_arm.py'),(full_refs,'full_refs.py'))):raise ValueError('wrong full module path')
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
  save(out/'worker_preflight.json',{'status':'passed','questions':len(rows),'source_sha256':source,'weights_verified':len(prepared['weights_before']),'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'predictions_opened':True,'prediction_scope':'E1 and E4 DEV mixed journals parsed exclusively for input and visual identity','versions':versions});return
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 torch.cuda.set_per_process_memory_fraction(.9,0);recipe=dict(prepared['plan']['recipe']);recipe['visual_token_num']=1024;start=time.monotonic()
 try:
  w=wrapper_class(**recipe)
  if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend mismatch')
 except BaseException as e:
  save(out/'model_load_failure.json',{'status':'failed','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'model_load_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});raise
 save(out/'frozen.json',{'source_sha256':source,'plan_sha256':sha(HERE/'plan.json'),'questions':384,'model_load_seconds':time.monotonic()-start,'versions':versions,'weight_stats_before':prepared['weights_before'],'GT_opened':False,'scoring_performed':False,'predictions_opened':True})
 def generate(model,row):
  with torch.inference_mode():return Qwen3VLChatFixedRes.generate_inner_transformers_fixed(model,row['message'],dataset=row['task'])
 from identity_records import tensor_meta
 def archive(tensors):return {'kind':'summary_only','tensors':{k:tensor_meta(v) for k,v in tensors.items()}}
 progress={'steps':0}
 def heartbeat(_m,_args,_kwargs,_out):
  progress['steps']+=1
  if progress['steps']%32==0:append(out/'journal.jsonl',{'event':'decode_progress','steps':progress['steps'],'unix':time.time()})
 hook=w.model.model.language_model.register_forward_hook(heartbeat,with_kwargs=True)
 def run_one(row,ref,returned,generated):
  w.model.model.rope_deltas=None
  return run_local(torch,w,row,Selector(target_vision_tokens=256),256,generate,returned,generated,archive,ref,recipe)
 try:
  if execute_rows(rows,refs,out,run_one)!=384:raise ValueError('full generation count')
  for path,before in prepared['weights_before'].items():
   s=Path(path).stat()
   if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight changed')
  if source!={str(x.resolve()):sha(x) for x in sources()}:raise ValueError('source changed')
  hook.remove();save(out/'worker_complete.json',{'status':'completed','questions':384,'generations':384,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'finished_unix':time.time()})
 except BaseException:hook.remove();raise
if __name__=='__main__':main()
