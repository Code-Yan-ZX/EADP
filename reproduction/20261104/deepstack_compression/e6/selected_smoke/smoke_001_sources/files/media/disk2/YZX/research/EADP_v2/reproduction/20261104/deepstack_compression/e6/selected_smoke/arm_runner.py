"""Real diagnostic arms. No loader, no GT, no launch authority."""
import hashlib,json,os,time
from pathlib import Path
from local_imports import prepare
from generation_contract import controls

class CacheFingerprints:
 def __init__(self,wrapper,all_steps):self.wrapper=wrapper;self.all_steps=all_steps;self.fingerprints=[]
 def __enter__(self):
  from identity_records import tensor_meta
  def capture(_module,_args,_kwargs,out):
   if self.fingerprints and not self.all_steps:return
   cache=out.past_key_values
   if cache is None or len(cache.layers)!=36:raise ValueError('expected 36 actual KV layers')
   self.fingerprints.append([{'key':tensor_meta(x.keys),'value':tensor_meta(x.values)} for x in cache.layers])
  self.handle=self.wrapper.model.model.language_model.register_forward_hook(capture,with_kwargs=True)
  return self
 def __exit__(self,*_args):self.handle.remove()

def append(path,record):
 with Path(path).open('a') as f:
  f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def run(torch,wrapper,author_class,pruner_class,method,rows,recipe,out):
 stock,contract=prepare();from baseline_runner import run_arm
 if type(wrapper) is not author_class or type(wrapper.pruner) is not pruner_class:
  raise ValueError('wrapper or pruner is not exact imported author class')
 if method not in ('div','cd') or len(rows)!=3 or [r['task'] for r in rows]!=['DocVQA_VAL','TextVQA_VAL','AI2D_TEST']:
  raise ValueError('unexpected method or anchor inventory')
 out=Path(out);(out/'arrays').mkdir(exist_ok=True)
 scalars={k:recipe[k] for k in ('top_p','top_k','temperature','repetition_penalty')};scalars.update(do_sample=False,max_new_tokens=2048)
 encoded=wrapper.processor.tokenizer.encode(' the',add_special_tokens=False)
 if not encoded:raise ValueError('forced diagnostic token unavailable')
 token=int(encoded[0]);completed={}
 def invoke(w,row):
  with torch.inference_mode():return w.generate_inner_transformers_fixed(row['message'],dataset=row['task'])
 def on_generated(record):append(out/'generated.jsonl',record)
 for row in rows:
  row=dict(row,split='DEV')
  for forced in ((False,True) if row['task']=='DocVQA_VAL' else (False,)):
   pair=[]
   for path in ('author_stock','adapted'):
    # A crash inside generate may expose an unknown partial prefix before its
    # final token IDs can be journalled. Persist that arm as started first.
    append(out/'events.jsonl',{'event':'arm_started','method':method,'task':row['task'],'qid':str(row['qid']),
                               'group_id':row['group_id'],'path':path,'forced':forced,'unix':time.time()})
    key=hashlib.sha256(json.dumps([method,row['task'],str(row['qid']),path,forced],separators=(',',':')).encode()).hexdigest()
    def archive(tensors):
     import numpy as np
     dst=out/'arrays'/(key+'.npz');payload={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
     with dst.open('xb') as f:np.savez(f,**payload);f.flush();os.fsync(f.fileno())
     h=hashlib.sha256()
     with dst.open('rb') as f:
      for data in iter(lambda:f.read(1048576),b''):h.update(data)
     return {'kind':'full_arrays','path':str(dst.resolve()),'sha256':h.hexdigest()}
    wrapper.model.model.rope_deltas=None
    with CacheFingerprints(wrapper,forced) as kv:
     if path=='author_stock':
      rec=stock.run_stock(torch,wrapper,method,row,invoke,on_generated,archive,scalars,token if forced else None)
     else:
      original=wrapper.model.generate;actual_controls={}
      def guarded(*args,**kw):
       actual={k:kw.get(k) for k in scalars};wanted=dict(scalars)
       if forced:wanted['max_new_tokens']=33
       if actual!=wanted or (forced and (kw.get('min_new_tokens')!=33 or kw.get('eos_token_id')!=[] or not kw.get('logits_processor'))):
        raise ValueError('actual adapted generation recipe differs before generate')
       actual_controls.update(controls(wrapper.model.generation_config,kw,forced))
       return original(*args,**kw)
      wrapper.model.generate=guarded
      try:
       rec=run_arm(torch,wrapper,method,row,invoke,lambda r:on_generated(dict(r,method=method,path='adapted',generation_controls=dict(actual_controls))),
                   archive,forced_token=token if forced else None,capture_prefill_logits=True)
      finally:wrapper.model.generate=original
      rec['path']='adapted';rec['generation_controls']=actual_controls
    rec['cache_fingerprints']=kv.fingerprints;contract.validate_cache(rec)
    append(out/'records.jsonl',rec);pair.append(rec);completed[(row['task'],path,forced)]=rec
    append(out/'events.jsonl',{'event':'arm_complete','method':method,'task':row['task'],'qid':str(row['qid']),
                                'group_id':row['group_id'],'path':path,'forced':forced,'unix':time.time()})
   contract.compare_pair(*pair)
   if forced:
    contract.compare_stock_forced(completed[(row['task'],'author_stock',False)],pair[0])
    contract.compare_forced(completed[(row['task'],'adapted',False)],pair[1])
   append(out/'comparisons.jsonl',{'method':method,'task':row['task'],'qid':str(row['qid']),'forced':forced,'status':'passed'})
 return {'method':method,'generations':8,'pairs':4,'forced_token':token}
