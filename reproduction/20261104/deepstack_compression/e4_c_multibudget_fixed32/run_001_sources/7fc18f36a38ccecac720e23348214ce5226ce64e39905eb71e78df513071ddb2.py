"""Fixed32 endpoint diagnostic continuation, locked until independently admitted.

No entrypoint, model load, GT or GPU authority.  Caller owns source pinning,
processor/weight identity, exclusive device, timeout and retained exit evidence.
"""
import json,os
from pathlib import Path

BUDGETS=(128,256,512,1024)
PREFILL_FIELDS=('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','greedy_calls','processor','main_visual','pruned_visual','image_start','original_length','full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','injected_mask','injected_deepstack','block_calls','main_merger_calls','ds_calls')

def compare_panel_prefill(actual,panel):
 if panel.get('actual_injection_verified') is not True:raise ValueError('accepted E4 injection proof missing')
 for name in PREFILL_FIELDS:
  if name not in actual or name not in panel or actual[name]!=panel[name]:raise ValueError('accepted E4 prefill mismatch: '+name)
 a=actual.get('decode_trace');b=panel.get('decode_trace')
 if not isinstance(a,list) or not a or not isinstance(b,list) or not b or a[0]!=b[0]:raise ValueError('accepted E4 prefill cache/position mismatch')
 # Existing full-panel first logits were not archived; only the independently
 # captured fixed32 fresh/light logits can be compared to one another.
 return True
def schedule(rows):
 if len(rows)!=3 or len({r['task'] for r in rows})!=3 or len({r['group_id'] for r in rows})!=3 or any(r.get('split')!='DEV' for r in rows):raise ValueError('three distinct DEV image-group anchors required')
 return [(row,k,side) for row in rows for k in BUDGETS for side in ('fresh','light')]

def run(torch,wrapper,native,refs,out,generate,fresh_run,light_run,compare,sha):
 """On actual completed generation fsync raw, then capture diagnostics.

 This deliberately stores no cache arrays: KV equality means equal 32x36
 cryptographic fingerprints, not archived or elementwise-reconstructed KV.
 """
 rows=refs['rows'];jobs=schedule(rows);records=refs['records'];out=Path(out)
 if not out.is_dir() or any((out/x).exists() for x in ('journal.jsonl','arrays','worker_complete.json')):raise ValueError('reused or missing output directory')
 for row in rows:
  for k in BUDGETS:
   p=records[(row['task'],str(row['qid']),k)]
   if p.get('budget')!=k or p.get('ds_on') is not True or any(str(p.get(x))!=str(row.get(x)) for x in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('accepted E4 recipe/input mismatch')
 (out/'arrays').mkdir();journal=out/'journal.jsonl';completed=0;prior={}
 def emit(item):
  with journal.open('a') as f:f.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 def archive_factory(job):
  import numpy as np
  path=out/'arrays'/f"{job['task']}_{job['qid']}_k{job['budget']}_{job['side']}.npz"
  def archive(tensors):
   data={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
   with path.open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
   return {'kind':'full_arrays','path':str(path),'sha256':sha(path)}
  return archive
 for row,k,side in jobs:
  job={'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'side':side,'protocol':'fixed32'}
  panel=records[(row['task'],str(row['qid']),k)];wrapper.visual_token_num=k;wrapper.pruner.visual_token_num=k
  emit({'event':'arm_started','job':job})
  def raw(value):emit({'event':'raw_generation','job':job,'raw':value})
  try:
   if side=='fresh':rec=fresh_run(torch,wrapper,native,row,k,True,generate,raw,archive_factory(job),capture_prefill_logits=True,accepted_panel=panel)
   else:rec=light_run(torch,wrapper,native,row,k,generate,raw,archive_factory(job),accepted_panel=panel)
   emit({'event':'arm_completed','job':job,'record':rec});completed+=1
   compare_panel_prefill(rec,panel)
   key=(row['task'],str(row['qid']),k)
   if side=='fresh':prior[key]=rec
   else:
    result=compare.compare_metadata(rec,prior[key]);arrays=compare.compare_archives(rec,prior[key])
    emit({'event':'pair_verified','job':job,'comparison':result,'arrays':arrays})
  except BaseException as exc:
   emit({'event':'arm_failed','job':job,'error_type':type(exc).__name__,'error':str(exc)});raise
 if len(jobs)!=24 or completed!=24:raise ValueError('incomplete fixed32 denominator')
 result={'status':'passed','pairs':12,'generations':24,'per_arm_output_tokens':32,'decode_forwards_after_prefill':31,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'KV_scope':'32 by 36 layers actual KV fingerprints only; full KV arrays not archived'}
 with (out/'worker_complete.json').open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 return result

if __name__=='__main__':raise SystemExit('fixed32 GPU path locked: no admitted launcher or independent prelaunch review')
