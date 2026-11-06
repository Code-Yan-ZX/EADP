"""Unadmitted GPU identity continuation. No runnable entrypoint.

The caller must provide a separately reviewed launch/identity/cleanup gate,
freeze this file, and supervise the retained Docker before calling this code.
This is an instrumented endpoint diagnostic, never a latency measurement.
"""
import json,os,time
from pathlib import Path

def append_journal(path,record):
 with Path(path).open('a') as f:
  f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def write_complete(path,record):
 with Path(path).open('x') as f:
  json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def run_after_future_admission(torch,wrapper,native,refs,out,generate,forced_token,
                               fresh_runner,light_runner,cache_fingerprints,core,compare,sha):
 """Use only after future reviewed admission; GPU scheduling belongs to root."""
 out=Path(out)
 if not out.is_dir() or any((out/name).exists() for name in ('journal.jsonl','arrays','worker_complete.json')):raise ValueError('reused output')
 (out/'arrays').mkdir();journal=out/'journal.jsonl'
 def emit(record):append_journal(journal,record)
 def archive_factory(job,side):
  import numpy as np
  name=f"{job['task']}_{job['qid']}_k{job['budget']}_{'forced' if job['forced'] else 'natural'}_{side}.npz"
  def archive(tensors):
   dst=out/'arrays'/name
   data={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
   with dst.open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
   return {'kind':'full_arrays','path':str(dst),'sha256':sha(dst)}
  return archive
 def fresh(row,budget,token,raw,archive):
  wrapper.visual_token_num=budget;wrapper.pruner.visual_token_num=budget
  with cache_fingerprints(wrapper,token is not None) as kv:
   result=fresh_runner(torch,wrapper,native,row,budget,True,generate,raw,archive,token,capture_prefill_logits=True)
  result['cache_fingerprints']=kv.fingerprints
  return result
 def light(row,budget,token,raw,archive):
  wrapper.visual_token_num=budget;wrapper.pruner.visual_token_num=budget
  return light_runner(torch,wrapper,native,row,budget,generate,raw,archive,token)
 result=core.run(refs['rows'],refs['records'],forced_token,fresh,light,archive_factory,emit,compare.compare_metadata,compare.compare_archives)
 if result.get('generations')!=32 or result.get('natural_pairs')!=12 or result.get('forced_pairs')!=4:raise ValueError('incomplete diagnostic')
 write_complete(out/'worker_complete.json',dict(result,finished_unix=time.time(),GT_opened=False,scoring_performed=False,timing_claims=False))
 return result

if __name__=='__main__':raise SystemExit('GPU worker locked: no admitted launcher or independent prelaunch approval')
