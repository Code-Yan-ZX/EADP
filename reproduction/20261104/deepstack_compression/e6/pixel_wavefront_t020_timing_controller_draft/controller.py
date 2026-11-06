"""Injected controller candidate; no GPU entry, no launcher or authority."""
import json,os
from pathlib import Path

def append(path,item):
 with Path(path).open('a') as f:json.dump(item,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())

def execute(jobs,references,journal,runner,validate):
 """One retained job per call. A failed job is recorded and stops the run."""
 journal=Path(journal)
 if journal.exists() or len(jobs)!=24 or [x['order'] for x in jobs]!=list(range(24)):raise ValueError('unplanned timing inventory')
 seen=set();results=[]
 for job in jobs:
  key=(job['task'],str(job['qid']),job['arm']);ref=references[key];append(journal,{'event':'started','job':job})
  try:
   item=runner(job);cid=item.get('container_id')
   if not isinstance(cid,str) or len(cid)!=64 or cid in seen or any(c not in '0123456789abcdef' for c in cid):raise ValueError('invalid or reused container')
   seen.add(cid)
   if item.get('returncode')!=0 or item.get('container_exit_code')!=0 or item.get('container_stopped_verified') is not True or item.get('timed_out') is not False or item.get('interrupted') is not None:raise ValueError('job container failed')
   record=item.get('record');metrics=validate(job,record,ref)
   append(journal,{'event':'completed','job':job,'container_id':cid,'record':record,'metrics':metrics});results.append(item)
  except BaseException as exc:
   append(journal,{'event':'failed','job':job,'error_type':type(exc).__name__,'error':str(exc)});raise
 if len(results)!=24 or len(seen)!=24:raise ValueError('incomplete timing')
 append(journal,{'event':'all_completed','jobs':24,'warmup':6,'measurement':18});return results
if __name__=='__main__':raise SystemExit('GPU locked: actual runner and independent approvals are not implemented')
