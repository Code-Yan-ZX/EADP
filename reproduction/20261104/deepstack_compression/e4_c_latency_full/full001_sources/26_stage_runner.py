"""Deterministic latency stage replay with durable output and failure records."""
import os
from schedule import build
from runner import execute_one

def execute_stage(schedule,rows,excluded,stage,execute,journal):
 if schedule!=build(rows,excluded,stage):raise ValueError('timing stage differs from frozen schedule')
 bykey={(r['task'],str(r['qid'])):r for r in rows}
 if len(bykey)!=384 or any(r.get('split')!='DEV' for r in rows):raise ValueError('DEV input inventory')
 jobs=schedule['schedule']
 for job in jobs:
  row=bykey.get((job['task'],str(job['qid'])))
  if row is None or row.get('group_id')!=job['group_id']:raise ValueError('schedule/input mismatch')
 for job in jobs:
  row=bykey[(job['task'],str(job['qid']))]
  def partial(raw):journal({'status':'generated_partial','job':job,'worker_pid':os.getpid(),'result':raw})
  try:value=execute(row,job['mode'],job['protocol'],partial)
  except BaseException as exc:
   journal({'status':'failed','job':job,'worker_pid':os.getpid(),'error_type':type(exc).__name__,'error':str(exc)});raise
  journal({'status':'generated','job':job,'worker_pid':os.getpid(),'measurement_role':'latency_interleaved','result':value})
  if value.get('mode')!=job['mode'] or value.get('protocol')!=job['protocol']:raise ValueError('actual execution identity mismatch')
  journal({'status':'completed','job':job,'worker_pid':os.getpid(),'measurement_role':'latency_interleaved'})
 return len(jobs)

def execute_model(schedule,rows,excluded,stage,wrapper,native,torch,journal):
 def one(row,mode,protocol,partial):
  if type(row.get('image_start')) is not int:raise ValueError('input visual span missing')
  with torch.inference_mode():return execute_one(wrapper,native,torch,row,mode,protocol,torch.cuda.synchronize,partial)
 return execute_stage(schedule,rows,excluded,stage,one,journal)
