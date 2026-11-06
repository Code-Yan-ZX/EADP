"""Fixed 384 DEV × three-seed natural generation, no execution authority."""
import json,os
from pathlib import Path
SEEDS=(20261104,20261105,20261106)
def schedule(rows,choices):
 if len(rows)!=384 or any(r.get('split')!='DEV' for r in rows):raise ValueError('expected complete DEV inventory')
 keys={(r['task'],str(r['qid'])) for r in rows}
 if len(keys)!=384 or len(choices)!=1152:raise ValueError('duplicate/incomplete inventory')
 out=[]
 for row in rows:
  key=(row['task'],str(row['qid']))
  for seed in SEEDS:
   r=choices.get((*key,seed))
   if not isinstance(r,dict) or any(str(r.get(k))!=str(row.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('frozen input/seed mismatch')
   ids=r.get('spatial_ids')
   if not isinstance(ids,list) or len(ids)!=256 or ids!=sorted(set(ids)) or any(type(i) is not int or not 0<=i<1024 for i in ids):raise ValueError('bad fixed selection')
   out.append((row,seed))
 if len(out)!=1152:raise ValueError('incomplete schedule')
 return out
def emit(path,obj):
 with Path(path).open('a',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
def run(rows,choices,refs,out,run_one,before_complete):
 jobs=schedule(rows,choices);out=Path(out)
 if not out.is_dir() or any((out/name).exists() for name in ('journal.jsonl','generated.jsonl','records.jsonl','worker_complete.json')):raise ValueError('output must be new')
 if len(refs)!=384 or any(len(refs.get((r['task'],str(r['qid'])),[]))!=1 for r in rows):raise ValueError('E1 reference inventory')
 for row,seed in jobs:
  key=(row['task'],str(row['qid']),seed);job={'task':key[0],'qid':key[1],'group_id':row['group_id'],'seed':seed,'forced':False};emit(out/'journal.jsonl',{'event':'started',**job});seen=[]
  def on_generated(raw):
   if seen:raise ValueError('duplicate generation')
   emit(out/'generated.jsonl',{'scheduled':job,'raw':raw});seen.append(True)
   if not isinstance(raw,dict) or any(raw.get(k)!=v for k,v in job.items()) or not isinstance(raw.get('output_ids'),list) or not raw['output_ids']:raise ValueError('generated exposure identity')
  try:
   result=run_one(row,refs[key[:2]][0],seed,on_generated)
   if len(seen)!=1 or not isinstance(result,dict) or any(result.get(k)!=v for k,v in job.items()) or result.get('spatial_ids')!=choices[key]['spatial_ids']:raise ValueError('arm identity/selection')
   if result.get('arrays',{}).get('kind')!='summary_only':raise ValueError('full panel must use summaries only')
   emit(out/'records.jsonl',result);emit(out/'journal.jsonl',{'event':'recorded',**job})
  except BaseException as e:
   emit(out/'journal.jsonl',{'event':'failed',**job,'error_type':type(e).__name__,'error':str(e)});raise
 before_complete()
 complete={'status':'completed','questions':384,'natural_generations':1152,'seeds':list(SEEDS),'GT_opened':False,'scoring_performed':False,'timing_eligible':False,'arrays':'summary_only'}
 with (out/'worker_complete.json').open('x') as f:json.dump(complete,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return complete
