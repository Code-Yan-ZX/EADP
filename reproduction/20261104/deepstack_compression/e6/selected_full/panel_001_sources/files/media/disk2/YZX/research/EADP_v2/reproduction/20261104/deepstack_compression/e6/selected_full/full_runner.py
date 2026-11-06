"""All-DEV Div/CD stock vs P3/DS natural arms, summary-only. No launch/GT."""
import json,os,time
from pathlib import Path
from full_imports import prepare
def append(path,row):
 with Path(path).open('a') as f:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def run_method(torch,wrapper,author_class,pruner_class,method,rows,recipe,out,expected_controls):
 stock,contract=prepare();from baseline_runner import run_arm
 from identity_records import tensor_meta
 from generation_contract import controls
 if type(wrapper) is not author_class or type(wrapper.pruner) is not pruner_class or method not in ('div','cd'):
  raise ValueError('author class/selector mismatch')
 if (len(rows)!=384 or len({(r['task'],str(r['qid']),r['group_id']) for r in rows})!=384 or
     set(expected_controls)!={'DocVQA_VAL','TextVQA_VAL','AI2D_TEST'}):raise ValueError('full DEV input/control inventory')
 out=Path(out);scalars={k:recipe[k] for k in ('top_p','top_k','temperature','repetition_penalty')};scalars.update(do_sample=False,max_new_tokens=2048)
 def invoke(w,row):
  with torch.inference_mode():return w.generate_inner_transformers_fixed(row['message'],dataset=row['task'])
 def archive(tensors):return {'kind':'summary_only','tensors':{k:tensor_meta(x) for k,x in tensors.items()}}
 def generated(record):append(out/'generated.jsonl',record)
 for row in rows:
  row=dict(row,split='DEV');pair=[]
  for path in ('author_stock','adapted'):
   ident={'method':method,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'path':path,'forced':False}
   append(out/'events.jsonl',dict(ident,event='arm_started',unix=time.time()));wrapper.model.model.rope_deltas=None
   if path=='author_stock':rec=stock.run_stock(torch,wrapper,method,row,invoke,generated,archive,scalars,expected_controls=expected_controls[row['task']])
   else:
    original=wrapper.model.generate;actual_controls={}
    def guarded(*args,**kw):
     if {k:kw.get(k) for k in scalars}!=scalars:raise ValueError('generation recipe differs before generate')
     actual_controls.update(controls(wrapper.model.generation_config,kw,False))
     if actual_controls!=expected_controls[row['task']]:raise ValueError('actual adapted controls differ before generate')
     return original(*args,**kw)
    wrapper.model.generate=guarded
    try:rec=run_arm(torch,wrapper,method,row,invoke,lambda r:generated(dict(r,method=method,path='adapted',generation_controls=dict(actual_controls))),archive)
    finally:wrapper.model.generate=original
    rec['path']='adapted';rec['generation_controls']=actual_controls
   if rec['arrays']['kind']!='summary_only':raise ValueError('full panel must not write high-dimensional arrays')
   append(out/'records.jsonl',rec);pair.append(rec);append(out/'events.jsonl',dict(ident,event='arm_completed',unix=time.time()))
  contract.compare_pair(*pair)
  append(out/'pairs.jsonl',{'method':method,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'status':'passed'})
 return {'method':method,'questions':384,'generations':768}
