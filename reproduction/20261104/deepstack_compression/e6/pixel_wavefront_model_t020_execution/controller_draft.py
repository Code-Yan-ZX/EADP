"""Injected six-arm controller prototype. No GPU/container entry is provided.

A future reviewed bounded runner must supply authenticated container identity,
real exit state and separately recorded source/weight evidence. Tests here use
only a fake runner and do not constitute model or GPU admission.
"""
import json,os
from pathlib import Path
import schedule_contract as contract

def emit(path,value):
 with Path(path).open('a') as f:
  f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def execute_with_injected_runner(images,journal,runner):
 jobs=contract.schedule(images);journal=Path(journal)
 if journal.exists():raise ValueError('existing identity journal')
 history=[]
 for job in jobs:
  # This event precedes the runner call so a killed process exposes the arm.
  emit(journal,{'event':'arm_invoked','job':job})
  try:
   result=runner(job)
   if not isinstance(result,dict):raise ValueError('runner result not structured')
   cid=result.get('container_id');started={'event':'started','job':job,'container_id':cid}
   complete={'event':'completed','job':job,'container_id':cid,'returncode':result.get('returncode'),
    'container_exit_code':result.get('container_exit_code'),'container_stopped_verified':result.get('container_stopped_verified'),
    'timed_out':result.get('timed_out'),'interrupted':result.get('interrupted'),'result_status':result.get('result_status')}
   history.extend((started,complete));emit(journal,{'event':'arm_result','job':job,'result':result})
   # Check each arm immediately, and never silently continue past failed work.
   # Container uniqueness is checked across the full actual history below.
   if (not isinstance(cid,str) or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid)
       or cid in {x['container_id'] for x in history[:-2:2]}
       or complete['returncode']!=0 or complete['container_exit_code']!=0
       or complete['container_stopped_verified'] is not True or complete['timed_out'] is not False
       or complete['interrupted'] is not None or complete['result_status']!='passed'):
    raise ValueError('identity arm failed or container reused')
  except BaseException as exc:
   emit(journal,{'event':'arm_failed','job':job,'error_type':type(exc).__name__,'error':str(exc)})
   raise
 contract.validate_history(jobs,history)
 emit(journal,{'event':'all_arms_completed','images':3,'jobs':6})
 return history

def main():raise SystemExit('GPU locked: real bounded runner, plan and independent review not implemented')
if __name__=='__main__':main()
