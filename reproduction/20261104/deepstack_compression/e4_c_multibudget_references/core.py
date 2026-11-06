"""Bulk DEV-only fixed32 references: 62 identities x 4K; never performance.

No model load/GPU entrypoint. Requires reviewed, bounded launcher, complete
source/input/weights admission and completed independent 12-pair identity gate.
"""
import json,os,re
from pathlib import Path
BUDGETS=(128,256,512,1024)
def schedule(rows):
 if len(rows)!=62 or len({(r['task'],str(r['qid'])) for r in rows})!=62 or len({(r['task'],r['group_id']) for r in rows})!=62 or any(r.get('split')!='DEV' for r in rows):raise ValueError('frozen 62 DEV image identities required')
 return [(r,k) for r in sorted(rows,key=lambda r:(r['task'],str(r['qid']))) for k in BUDGETS]
def validate(rec,panel,fixed_core):
 fixed_core.compare_panel_prefill(rec,panel)
 if rec.get('protocol')!='fixed32' or rec.get('forced') is not False or rec.get('min_new_tokens')!=32 or rec.get('max_new_tokens')!=32 or rec.get('generation_scalars')!=dict(panel['generation_scalars'],max_new_tokens=32) or rec.get('eos_ids')!=panel['eos_ids'] or not isinstance(rec.get('answer'),str):raise ValueError('fixed32 reference actual recipe')
 ids=rec.get('output_ids');eos=rec['eos_ids'];trace=rec.get('decode_trace');k=rec['budget']
 if not isinstance(ids,list) or len(ids)!=32 or any(type(v) is not int for v in ids) or not eos or any(v in eos for v in ids[:-1]):raise ValueError('fixed32 output incomplete')
 if not isinstance(trace,list) or len(trace)!=32:raise ValueError('prefill plus 31 decode required')
 seq=rec['original_length']-1024+k;delta=rec['adjusted_delta'][0][0]
 for i,s in enumerate(trace):
  before=0 if i==0 else seq+i-1;after=seq+i
  expected_position=rec['prefill_position_ids'] if i==0 else [[[seq+i-1+delta]]]*3
  expected_cache=list(range(seq)) if i==0 else [seq+i-1]
  before_ok=s.get('layer_cache_lengths') in ([],[0]*36) if i==0 else s.get('layer_cache_lengths')==[before]*36
  if s.get('position_ids')!=expected_position or s.get('cache_position')!=expected_cache or s.get('rope_delta')!=[[delta]] or not before_ok or s.get('layer_cache_lengths_after')!=[after]*36:raise ValueError('fixed32 position/cache trajectory')
 fps=rec.get('cache_fingerprints')
 if not isinstance(fps,list) or len(fps)!=1 or len(fps[0])!=36:raise ValueError('prefill KV fingerprint missing')
 for layer in fps[0]:
  if set(layer)!={'key','value'}:raise ValueError('bad KV pair')
  for v in layer.values():
   if set(v)!={'dtype','shape','sha256'} or v['dtype']!='torch.bfloat16' or v['shape']!=[1,8,seq,128] or re.fullmatch('[0-9a-f]{64}',v['sha256']) is None:raise ValueError('bad prefill KV fingerprint')
 if (rec.get('arrays') or {}).get('kind')!='summary_only':raise ValueError('reference array evidence type')
 return True

def run(torch,wrapper,native,refs,out,generate,fresh_run,fixed_core,tensor_meta):
 rows=refs['rows'];jobs=schedule(rows);records=refs['records'];out=Path(out)
 if not out.is_dir() or any((out/x).exists() for x in ('journal.jsonl','records.jsonl','worker_complete.json')):raise ValueError('reused or missing output')
 for row,k in jobs:
  p=records[(row['task'],str(row['qid']),k)]
  if p.get('budget')!=k or p.get('ds_on') is not True or any(str(p.get(x))!=str(row.get(x)) for x in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('E4 accepted recipe mismatch')
 def append(path,obj):
  with path.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 def summary(tensors):return {'kind':'summary_only','tensors':{k:tensor_meta(v) for k,v in tensors.items()}}
 for index,(row,k) in enumerate(jobs):
  job={'order':index,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'protocol':'fixed32'}
  panel=records[(row['task'],str(row['qid']),k)];wrapper.visual_token_num=k;wrapper.pruner.visual_token_num=k
  append(out/'journal.jsonl',{'event':'arm_started','job':job})
  def raw(value):append(out/'journal.jsonl',{'event':'raw_generation','job':job,'raw':value})
  try:
   rec=fresh_run(torch,wrapper,native,row,k,True,generate,raw,summary,capture_prefill_logits=True,accepted_panel=panel)
   append(out/'journal.jsonl',{'event':'arm_captured','job':job,'record':rec})
   validate(rec,panel,fixed_core)
   append(out/'records.jsonl',rec);append(out/'journal.jsonl',{'event':'arm_completed','job':job})
  except BaseException as error:
   append(out/'journal.jsonl',{'event':'arm_failed','job':job,'error_type':type(error).__name__,'error':str(error)});raise
 result={'status':'passed','inputs':62,'budgets':4,'generations':248,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'prefill_KV':'36 layer fingerprints','decode_KV':'31 subsequent forwards: positions and length per layer only; no KV content archived'}
 with (out/'worker_complete.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return result
if __name__=='__main__':raise SystemExit('GPU locked: reference execution has no admitted launcher or prelaunch proof')
